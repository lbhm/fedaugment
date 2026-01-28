import operator
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from loguru import logger
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

from fedaugment.augmentations.evaluator_base import BaseEvaluator
from fedaugment.augmentations.retrieval_metrics import (
    RetrievalMetrics,
    compute_match_statistics,
    compute_retrieval_metrics,
)
from fedaugment.config import HNSWConfig
from fedaugment.projections.dataset import EmbeddingDataset

# Large cost value to disallow assignments in Hungarian algorithm
DISALLOWED = 1e9


class UnionDiscoveryEvaluator(BaseEvaluator):
    """Evaluator for table-level union discovery using Starmie-style bipartite matching.

    This evaluator implements table union discovery, which identifies tables that can be
    vertically combined (unioned) with a query table. The evaluation process follows the
    Starmie methodology:

    1. **Query Processing**: Each query represents an entire table (all its columns)
    2. **Candidate Retrieval**: For each query column, use HNSW to find similar candidate columns
    3. **Table Scoring**: Group candidates by table and compute unionability scores via
       maximum-weight bipartite matching between query and candidate column embeddings
    4. **Evaluation**: Rank candidate tables and compute table-level retrieval metrics

    Input Format Assumptions:
        - candidate_dataset: Column-level embeddings with IDs as "<table_name>::<column_name>"
        - query_dataset: Same format as candidate dataset for query table columns
        - query_path: File with single column "query_table" listing query table identifiers
        - ground_truth_path: File with columns "query_table" and "candidate_table"
          specifying which tables can be unioned
    """

    def __init__(
        self,
        candidate_dataset: EmbeddingDataset[Any],
        query_dataset: EmbeddingDataset[Any],
        query_path: Path,
        ground_truth_path: Path,
        checkpoint_path: Path | None,
        hnsw_config: HNSWConfig,
        filter_ground_truth: bool = True,
        device: Literal["auto", "cpu", "cuda"] | torch.device = "auto",
        candidate_knn_per_col: int = 50,
    ) -> None:
        """Initializes the UnionDiscoveryEvaluator.

        Args:
            candidate_dataset: Dataset containing the candidate column embeddings and column
                identifiers for the dataset collection.
            query_dataset: Dataset containing the query column embeddings and column
                identifiers.
            query_path: Path to a CSV or Parquet file containing join queries.
            ground_truth_path: Path to a CSV or Parquet file containing the ground truth for join
                queries.
            checkpoint_path: Path to a projection model checkpoint. Used to align query embeddings.
                If None, query embeddings are taken as-is from the query_dataset.
            hnsw_config: Configuration parameters for the HNSW index.
            filter_ground_truth: Whether to filter the ground truth to only include candidate
                columns found in the dataset collection.
            device: Computation device for torch tensors.
            candidate_knn_per_col: Number of nearest neighbor columns to retrieve per query column.
                This controls the recall vs. efficiency trade-off in candidate generation.
                Typical values: 50 (standard), 4 (fast evaluation).
        """
        super().__init__(
            candidate_dataset,
            query_dataset,
            query_path,
            ground_truth_path,
            checkpoint_path,
            hnsw_config,
            filter_ground_truth,
            device,
        )
        self.candidate_knn_per_col = candidate_knn_per_col

        # Validate and load query table specifications
        if "query_table" not in self.query_df.columns:
            raise ValueError("query_path must contain a `query_table` column")
        self.query_tables = self.query_df["query_table"].unique()

        # Validate and process ground truth data
        if not {"query_table", "candidate_table"}.issubset(set(self.ground_truth_df.columns)):
            raise ValueError(
                "ground_truth_path must contain `query_table` and `candidate_table` columns"
            )

        # Filter ground truth to only include query tables available in candidate dataset
        if filter_ground_truth:
            self.ground_truth_df = self._filter_gt(
                self.ground_truth_df,
                pl.Series(values=self.candidate_ids, dtype=pl.String),
                include_col_level=False,  # Table-level filtering for union discovery
            )

        # Preprocess ground truth for fast lookup: query_table -> set(candidate_table)
        self.ground_truth: dict[str, set[str]] = defaultdict(set)
        for q, c in self.ground_truth_df.rows():
            self.ground_truth[q].add(c)

        logger.info(
            "Loaded ground truth for {} query tables. Ground truth size distribution: {}",
            len(self.ground_truth),
            Counter(len(v) for v in self.ground_truth.values()),
        )

        # Pre-group candidate columns by table for efficient table-level scoring
        # Format: table_name -> list of (column_id, normalized_embedding)
        self.candidate_table_columns = self._group_candidate_columns_by_table(
            self.CANDIDATE_VIEW_IDX
        )

    def evaluate(self, k_vals: list[int], view_idx: int | Literal["iterate"]) -> pl.DataFrame:
        """Evaluate table-level union discovery using bipartite matching.

        For each query table, this method:
        1. Retrieves candidate columns via HNSW search across all query columns
        2. Groups candidates by their source tables
        3. Computes unionability scores using maximum-weight bipartite matching
        4. Ranks candidate tables by score and computes retrieval metrics

        Args:
            k_vals: List of k values for computing top-k retrieval metrics (e.g., [1, 5, 10]).
            view_idx: Index of the query embedding view to use for projection.
                 If set to "iterate", the evaluator uses a different view per query.

        Returns:
            DataFrame with columns: query_table, MAP@k, R@k, P@k for each k in k_vals.
            Each row contains metrics for one query table's union discovery results.
        """
        if isinstance(view_idx, int) and not (0 <= view_idx < self.query_dataset.num_views):
            max_idx = self.query_dataset.num_views - 1
            raise ValueError(f"Invalid view_idx {view_idx}: must be in range [0, {max_idx}]")

        # Prepare result schema
        col_names = RetrievalMetrics.get_column_names(k_vals, "union")

        # Group query columns by table for efficient processing
        query_table_columns = self._group_query_columns_by_table(view_idx)

        # Evaluate each query table independently
        result_list: list[list[str | float]] = []
        n_skipped = 0
        logger.info("Starting evaluation of {} queries.", len(self.query_df))
        for query_table in tqdm(
            self.query_tables,
            desc="Evaluating query tables",
            total=len(self.query_tables),
            leave=False,
            unit="query",
            mininterval=1.0,
            dynamic_ncols=True,
        ):
            query_cols = query_table_columns.get(query_table, [])
            if not query_cols:
                logger.warning("No columns/embeddings found for query table {}", query_table)
                n_skipped += 1
                continue

            # Stack and normalize query column embeddings for batch processing
            query_embs = torch.vstack([emb for _, emb in query_cols]).to(self.device)  # (m, D)

            # Step 1: Collect candidate tables via HNSW search across query columns
            # This reduces the search space from all tables to a manageable subset
            candidate_tables = self._collect_candidate_tables_via_hnsw(query_cols)

            # Step 2: Score each candidate table using bipartite matching
            scored_tables: list[tuple[str, float]] = []  # list of (table_name, unionability_score)
            for cand_table in candidate_tables:
                cand_cols = self.candidate_table_columns.get(cand_table, [])
                if not cand_cols:
                    continue

                # Stack and normalize candidate column embeddings
                cand_embs = torch.vstack([emb for _, emb in cand_cols]).to(self.device)
                score = self._compute_unionability_score(query_embs, cand_embs)
                scored_tables.append((cand_table, score))

            # Step 3: Rank candidate tables by unionability score (descending)
            scored_tables.sort(key=operator.itemgetter(1), reverse=True)
            ranked_tables = [table_name for table_name, _ in scored_tables]
            gt_tables = list(self.ground_truth.get(query_table, []))

            # Step 4: Compute retrieval metrics for each k value
            result_row: list[str | float] = [query_table]
            for k in k_vals:
                # Compute match statistics and metrics
                stats = compute_match_statistics(ranked_tables, set(gt_tables), k)
                metrics = compute_retrieval_metrics(stats)
                result_row.extend(metrics.to_list())

            result_list.append(result_row)

        results = pl.DataFrame(result_list, schema=col_names, orient="row")
        if n_skipped > 0:
            logger.warning(
                "Skipped {} of {} queries (missing embeddings or ground truth).",
                n_skipped,
                len(self.query_df),
            )

        logger.info("Completed evaluation of {} queries.", len(results))
        return results

    def _group_candidate_columns_by_table(
        self, view_idx: int
    ) -> dict[str, list[tuple[str, torch.Tensor]]]:
        """Group candidate column embeddings by their source table.

        This preprocessing step organizes candidate columns by table to enable efficient
        table-level scoring during evaluation. Each table's columns are normalized and converted to
        PyTorch tensors.

        Args:
            view_idx: Index of the embedding view to extract from the dataset.

        Returns:
            Dictionary mapping table names to lists of (column_id, normalized_embedding) tuples.
            Each embedding is L2-normalized for cosine similarity computation.
        """
        table_map: dict[str, list[tuple[str, torch.Tensor]]] = defaultdict(list)
        embeddings = self.candidate_dataset.get_view(view_idx)

        for i, col_id in enumerate(self.candidate_ids):
            # Extract table name from column ID (format: "table_name::column_name")
            table_name = col_id.split("::")[0]

            # Convert numpy embedding to normalized PyTorch tensor
            emb = torch.from_numpy(embeddings[i]).reshape(-1).float()
            emb = F.normalize(emb, dim=0)  # L2 normalize for cosine similarity

            table_map[table_name].append((col_id, emb))

        return table_map

    @torch.no_grad()
    def _group_query_columns_by_table(
        self, view_idx: int | Literal["iterate"]
    ) -> dict[str, list[tuple[str, torch.Tensor]]]:
        """Group query column embeddings by their source table with optional projection.

        This method processes query embeddings by:
        1. Extracting embeddings from the specified view
        2. Applying projection model alignment if available
        3. L2-normalizing embeddings for cosine similarity
        4. Grouping by table name for table-level evaluation

        Args:
            view_idx: Index of the query embedding view to use for projection.
                If set to "iterate", the evaluator uses a different view per query.

        Returns:
            Dictionary mapping table names to lists of (column_id,
            projected_normalized_embedding) tuples.
        """
        table_map: dict[str, list[tuple[str, torch.Tensor]]] = defaultdict(list)

        # Process each query column embedding
        for i, col_id in enumerate(self.query_ids):
            # Extract table name from column ID
            table_name = col_id.split("::")[0]
            current_view_idx = (
                i % self.query_dataset.num_views if view_idx == "iterate" else view_idx
            )

            # Load and prepare embedding tensor
            emb = (
                torch.from_numpy(self.query_dataset.get_view(current_view_idx, i))
                .reshape(-1)
                .float()
                .to(self.device)
            )

            # Apply projection model if available to align with candidate space
            if self.projection_model is not None:
                emb = self.projection_model.project(emb.unsqueeze(0), current_view_idx).squeeze(0)

            # L2 normalize for cosine similarity computation
            emb = F.normalize(emb, dim=0)
            table_map[table_name].append((col_id, emb))

        return table_map

    def _collect_candidate_tables_via_hnsw(
        self, query_cols: list[tuple[str, torch.Tensor]]
    ) -> set[str]:
        """Collect candidate tables by retrieving similar columns via HNSW search.

        This method implements the first stage of candidate generation:
        1. For each query column, use HNSW to find k most similar candidate columns
        2. Extract the source table of each similar column
        3. Return the union of all candidate tables

        This approach balances recall (by considering multiple query columns) with
        efficiency (by limiting the number of candidates to score).

        Args:
            query_cols: List of (column_id, normalized_embedding) tuples for the query table.

        Returns:
            Set of candidate table names that contain at least one similar column
            to some query column.
        """
        candidate_tables: set[str] = set()

        # For each query column, find similar candidate columns via HNSW
        for _, emb in query_cols:
            # Prepare query embedding for HNSW (requires float32 numpy array)
            query_emb = emb.unsqueeze(0).cpu().numpy().astype(np.float32)

            # Retrieve k nearest neighbor columns (limited by dataset size)
            k = min(self.candidate_knn_per_col, len(self.candidate_ids))

            neighbor_indices = self._query_index(query_emb, k=k)
            neighbor_indices = neighbor_indices.flatten()

            # Extract source tables from similar columns
            for neighbor_idx in neighbor_indices:
                neighbor_col_id = self.candidate_ids[neighbor_idx]
                neighbor_table = neighbor_col_id.split("::")[0]
                candidate_tables.add(neighbor_table)

        return candidate_tables

    def _compute_unionability_score(
        self, query_embs: torch.Tensor, cand_embs: torch.Tensor, tau: float = 0.1
    ) -> float:
        """Compute table unionability score using Starmie-style bipartite matching.

        This method implements the core unionability scoring algorithm:
        1. Compute pairwise cosine similarities between query and candidate columns
        2. Apply similarity threshold to filter weak connections
        3. Use Hungarian algorithm to find maximum-weight bipartite matching
        4. Return average similarity of matched column pairs

        The bipartite matching ensures one-to-one column alignment, preventing
        multiple query columns from matching the same candidate column.

        Args:
            query_embs: Query column embeddings of shape (m, D), L2-normalized.
            cand_embs: Candidate column embeddings of shape (n, D), L2-normalized.
            tau: Similarity threshold for filtering weak column connections.
                 Typical values: 0.1 (lenient), 0.7 (strict). Default from Starmie paper.

        Returns:
            Unionability score as the average cosine similarity of optimally matched
            column pairs. Returns 0.0 if no valid matches exist above the threshold.
        """
        # Step 1: Compute pairwise cosine similarity matrix
        similarity_matrix = query_embs @ cand_embs.T  # (m, n)
        sim_np = similarity_matrix.cpu().numpy()

        # Step 2: Apply similarity threshold to filter weak connections
        # Similarities below tau are set to 0 (no connection)
        masked_sim = np.where(sim_np >= tau, sim_np, 0.0)

        # Step 3: Check if any valid connections exist
        if masked_sim.max() == 0:
            return 0.0  # No column pairs meet the similarity threshold

        # Step 4: Convert to minimum-cost assignment problem for Hungarian algorithm
        # The algorithm minimizes cost, so we convert maximum similarity to minimum cost
        max_sim = masked_sim.max()
        cost_matrix = np.where(
            masked_sim > 0,
            max_sim - masked_sim,  # Valid connections: convert to cost
            DISALLOWED,  # Invalid connections: prohibitively high cost
        )

        # Step 5: Solve optimal assignment using Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Step 6: Compute average similarity of matched pairs
        total_score = 0.0
        valid_matches = 0
        for r, c in zip(row_indices, col_indices, strict=True):
            if masked_sim[r, c] > 0:  # Only count valid matches above threshold
                total_score += float(masked_sim[r, c])
                valid_matches += 1

        # Return average similarity of matched column pairs
        return total_score / len(row_indices) if len(row_indices) > 0 else 0.0
