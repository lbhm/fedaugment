from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from loguru import logger
from tqdm.auto import tqdm

from fedaugment.augmentations.evaluator_base import BaseEvaluator
from fedaugment.augmentations.retrieval_metrics import (
    RetrievalMetrics,
    compute_match_statistics,
    compute_retrieval_metrics,
)
from fedaugment.config import HNSWConfig
from fedaugment.projections.dataset import EmbeddingDataset
from fedaugment.types import StrArray
from fedaugment.utils import sanitize_col_name


class JoinDiscoveryEvaluator(BaseEvaluator):
    """Evaluator for joinable table discovery using aligned embeddings and an HNSW index.

    This class evaluates the performance of joinable table discovery by:
    1. Loading query tables and their ground truth join candidates from a benchmark
    2. Projecting query column embeddings into the shared/aligned embedding space
    3. Searching for nearest neighbors in a pre-built HNSW index containing all column embeddings
    4. Calculating standard retrieval metrics (Recall@k, Precision@k, Hits@k, MRR@k)
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
        exclude_self_joins: bool = True,
    ) -> None:
        """Initializes the JoinDiscoveryEvaluator.

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
            exclude_self_joins: Whether to exclude columns from the same table as the query when
                computing metrics. Default True to prevent query tables from matching themselves.
        """
        # Initialize base evaluator which handles dataset loading, HNSW construction,
        # loading projection model and basic pre-processing.
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
        self.exclude_self_joins = exclude_self_joins

        # Expected schema: "query_table", "query_column"
        if not {"query_table", "query_column"}.issubset(set(self.query_df.columns)):
            raise ValueError("query_path must contain `query_table` and `query_column` columns")
        # Expected schema: "query_table", "candidate_table", "query_column", "candidate_column"
        if not {"query_table", "candidate_table", "query_column", "candidate_column"}.issubset(
            set(self.ground_truth_df.columns)
        ):
            raise ValueError(
                "ground_truth_path must contain `query_table`, `candidate_table`, "
                "`query_column`, `candidate_column` columns"
            )

        # Filter ground truth to only include query columns available in candidate dataset
        if filter_ground_truth:
            self.ground_truth_df = self._filter_gt(
                self.ground_truth_df,
                pl.Series(values=self.candidate_ids, dtype=pl.String),
                include_col_level=True,  # Column-level filtering for join discovery
            )

        # Preprocess ground truth for fast lookup (column-level)
        self.ground_truth_ids: defaultdict[str, set[str]] = defaultdict(set)
        for query_table, candidate_table, query_col, candidate_col in self.ground_truth_df.rows():
            query_id = self._get_column_id(query_table, query_col)
            candidate_id = self._get_column_id(candidate_table, candidate_col)
            self.ground_truth_ids[query_id].add(candidate_id)

        logger.info(
            "Loaded ground truth for {} queries. Ground truth size distribution: {}",
            len(self.ground_truth_ids),
            Counter(len(v) for v in self.ground_truth_ids.values()),
        )

    def evaluate(self, k_vals: list[int], view_idx: int | Literal["iterate"]) -> pl.DataFrame:
        """Evaluate joinable table discovery performance on the benchmark dataset.

        For each query column, retrieves the top-k nearest neighbors from the dataset collection
        and compares them against the ground truth joinable column candidates.

        Args:
            k_vals: Number of top candidates to retrieve for each query.
            view_idx: Index of the embedding view/model used to project query embeddings to the
                aligned space. If set to "iterate", the evaluator uses a different view per query.

        Returns:
            A Polars DataFrame containing the join evaluation metrics for each query.
        """
        if isinstance(view_idx, int) and not (0 <= view_idx < self.query_dataset.num_views):
            max_idx = self.query_dataset.num_views - 1
            raise ValueError(f"Invalid view_idx {view_idx}: must be in range [0, {max_idx}]")

        # Prepare result schema
        col_names = RetrievalMetrics.get_column_names(k_vals, "join")

        result_list: list[list[str | float]] = []
        n_skipped = 0
        logger.info("Starting evaluation of {} queries.", len(self.query_df))
        for i, (query_table, query_col) in tqdm(
            enumerate(self.query_df.rows()),
            desc="Processing queries",
            total=len(self.query_df),
            leave=False,
            unit="query",
            mininterval=1.0,
            dynamic_ncols=True,
        ):
            current_view_idx = (
                i % self.query_dataset.num_views if view_idx == "iterate" else view_idx
            )
            query_id = self._get_column_id(query_table, query_col)
            query_emb = self._get_query_embedding(query_id, current_view_idx)
            if query_emb is None:
                logger.warning("Query column {} not found in query dataset", query_id)
                n_skipped += 1
                continue

            gt_ids = self.ground_truth_ids[query_id]
            if not gt_ids:
                logger.warning("No ground truth found for query {}", query_id)
                n_skipped += 1
                continue

            result_row = self._process_query(query_id, query_emb, gt_ids, k_vals)
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

    def _process_query(
        self, query_id: str, query_emb: torch.Tensor, gt_ids: set[str], k_vals: list[int]
    ) -> list[str | float]:
        """Process a single query and return metrics for all k values."""
        result_row: list[str | float] = [query_id]
        for k in k_vals:
            if self.exclude_self_joins:
                pred_ids = self._get_k_non_self_joins(query_emb, query_id, k)
            else:
                pred_idx = self._query_index(query_emb.detach().cpu().numpy(), k=k)
                pred_ids = self.candidate_ids[pred_idx]  # (1, k)

            # Compute match statistics and metrics
            stats = compute_match_statistics(pred_ids.flatten().tolist(), gt_ids, k)
            metrics = compute_retrieval_metrics(stats)
            result_row.extend(metrics.to_list())
        return result_row

    def _get_k_non_self_joins(self, query_emb: torch.Tensor, query_id: str, k: int) -> StrArray:
        """Fetch results in batches until k non-self-join predictions are found.

        This method robustly handles cases where a table has many columns by iteratively querying
        the HNSW index with increasing batch sizes until k non-self-join results are obtained.

        Args:
            query_emb: The query embedding tensor.
            query_id: The identifier of the query column.
            k: The number of non-self-join predictions to retrieve.

        Returns:
            A 2D NumPy array of shape (1, k) (or fewer if not enough non-self candidates exist)
            containing the filtered prediction IDs.
        """
        query_arr = query_emb.detach().cpu().numpy()
        query_table_name = query_id.rsplit("::", 1)[0]
        filtered_ids: list[str] = []
        batch_size = max(k * 2, 50)
        fetched = 0

        while len(filtered_ids) < k:
            search_k = batch_size + fetched

            pred_idx = self._query_index(query_arr, k=search_k)
            if len(pred_idx) == 0 or len(pred_idx[0]) == fetched:
                break

            pred_ids = self.candidate_ids[pred_idx.flatten()[fetched:]]
            # TODO: This loop looks inefficient; consider vectorized filtering
            for pred_id in pred_ids:
                pred_table = pred_id.rsplit("::", 1)[0]
                if pred_table != query_table_name:
                    filtered_ids.append(pred_id)
                    if len(filtered_ids) >= k:
                        break

            fetched = search_k

        return np.array([filtered_ids[:k]], dtype=object)

    def _get_column_id(self, table: str, column: str) -> str:
        """Construct the column identifier used in the datasets."""
        return f"{table}::{sanitize_col_name(column)}"

    @torch.no_grad()
    def _get_query_embedding(self, query_id: str, view_idx: int) -> torch.Tensor | None:
        """Retrieve and optionally project the embedding for a query column.

        Args:
            query_id: Column ID of the query.
            view_idx: Index of the embedding view/model used to project the query embedding.

        Returns:
            Query embedding tensor of shape (1, D), or None if not found.
        """
        query_idx = np.asarray(self.query_ids == query_id).nonzero()[0]  # only first dimension
        if len(query_idx) == 0:
            return None
        if len(query_idx) > 1:
            raise ValueError(f"Multiple embeddings found for query column ID: {query_id}")

        query_emb = (
            torch.from_numpy(self.query_dataset.get_view(view_idx, query_idx[0]))
            .reshape(1, -1)
            .to(self.device)
        )  # (1, D_i)
        if self.projection_model is not None:
            query_emb = self.projection_model.project(query_emb, view_idx)
        return F.normalize(query_emb, dim=1)  # (1, D)
