import abc
import time
from pathlib import Path
from typing import Any, ClassVar, Literal

import hnswlib
import numpy as np
import polars as pl
import torch
from loguru import logger
from numpy.typing import ArrayLike, NDArray

from fedaugment.config import HNSWConfig
from fedaugment.projections import load_projection_model
from fedaugment.projections.dataset import EmbeddingDataset


class BaseEvaluator(abc.ABC):
    """Abstract base class for table augmentation evaluators.

    This class provides common functionality for evaluating table augmentation tasks by:
    1. Loading candidate and query embedding datasets
    2. Building an HNSW index for efficient similarity search over candidate embeddings
    3. Loading projection models to align query embeddings into the candidate space
    4. Preprocessing query tables and ground truth data
    5. Filtering ground truth to only include candidates present in the dataset collection

    Subclasses must implement the `evaluate` method to perform task-specific evaluation
    (e.g., join discovery, union discovery) and return metrics as a Polars DataFrame.
    """

    CANDIDATE_VIEW_IDX: ClassVar[int] = 0
    """Index of the embedding view to use from the candidate dataset.

    We assume that the candidate dataset contains aligned embeddings in a single view
    (typically view 0) and ignore all other views. This view contains embeddings that
    have been projected into a common space by a trained projection model.
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
    ) -> None:
        """Initialize the evaluator with datasets and configuration.

        Args:
            candidate_dataset: Dataset containing candidate column embeddings and identifiers.
                These embeddings should already be aligned/projected into a common space.
            query_dataset: Dataset containing query column embeddings and identifiers.
                These will be projected using the checkpoint model if provided.
            query_path: Path to file containing query specifications (format depends on subclass).
            ground_truth_path: Path to file containing ground truth annotations.
            checkpoint_path: Path to projection model checkpoint for aligning query embeddings.
                If None, query embeddings are used as-is without projection.
            hnsw_config: Configuration parameters for the HNSW search index.
            filter_ground_truth: Whether to filter ground truth to only include candidates
                that exist in the candidate dataset collection.
            device: Computation device for PyTorch operations ("auto", "cpu", "cuda", or
                device object).
        """
        super().__init__()

        # Store dataset references and extract column identifiers
        self.candidate_dataset = candidate_dataset
        self.candidate_ids = self.candidate_dataset.column_ids
        self.query_dataset = query_dataset
        self.query_ids = self.query_dataset.column_ids

        # Log candidate dataset information for debugging
        logger.info(
            "Candidate dataset metadata: {}",
            self.candidate_dataset.metadata[self.CANDIDATE_VIEW_IDX],
        )

        # Store file paths for reference
        self.query_path = query_path
        self.gt_path = ground_truth_path
        self.ckpt_path = checkpoint_path
        if checkpoint_path:
            logger.info("Using projection model from checkpoint: {}", checkpoint_path)

        # Store configuration
        self.hnsw_config = hnsw_config
        self.filter_ground_truth = filter_ground_truth

        # Set up computation device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load and construct required components
        self.query_df = self._load_dataframe(query_path)
        self.ground_truth_df = self._load_dataframe(ground_truth_path)
        self.projection_model = (
            load_projection_model(checkpoint_path, self.device) if checkpoint_path else None
        )
        self.hnsw_index = self._construct_hnsw_index(hnsw_config)

    @abc.abstractmethod
    def evaluate(self, k_vals: list[int], view_idx: int | Literal["iterate"]) -> pl.DataFrame:
        """Perform task-specific evaluation and return metrics.

        This abstract method must be implemented by subclasses to define the specific
        evaluation procedure for their augmentation task (e.g., join discovery, union discovery).

        Args:
            k_vals: List of k values for computing top-k retrieval metrics (e.g., [1, 5, 10]).
            view_idx: Index of the embedding view to use from the query dataset for projection.
                Different views correspond to different embedding models/strategies.
                If set to "iterate", the evaluator uses a different view for each query.

        Returns:
            DataFrame containing evaluation metrics for each query, with columns depending
            on the specific evaluation task and requested k values.
        """

    def _construct_hnsw_index(self, config: HNSWConfig) -> hnswlib.Index:
        """Build an HNSW index from the candidate dataset embeddings for fast similarity search.

        The HNSW (Hierarchical Navigable Small World) index enables efficient approximate
        nearest neighbor search over the candidate column embeddings. This is crucial for
        scalable table augmentation as it allows fast retrieval of similar columns without
        computing pairwise similarities with all candidates.

        Args:
            config: HNSW configuration including metric, construction parameters, and
                search parameters.

        Returns:
            Constructed HNSW index loaded with all candidate embeddings.

        Note:
            The candidate embeddings must be L2-normalized beforehand for cosine similarity search.
        """
        logger.info(
            "Constructing HNSW index with {} metric and {} dimensions.",
            config.metric,
            self.candidate_dataset.embedding_dims[self.CANDIDATE_VIEW_IDX],
        )
        t0 = time.time()

        # Initialize the index with the embedding dimensionality
        index = hnswlib.Index(
            space=config.metric, dim=self.candidate_dataset.embedding_dims[self.CANDIDATE_VIEW_IDX]
        )

        # Configure index parameters that affect search quality vs. speed tradeoffs
        index.init_index(
            max_elements=len(self.candidate_dataset),
            ef_construction=config.ef_construction,  # Controls index construction quality
            M=config.M,  # Maximum number of connections per node
            random_seed=config.seed,
        )
        index.set_ef(config.ef)  # Controls search quality at query time

        # Add all candidate embeddings to the index
        # NOTE: The candidate dataset embeddings must be normalized beforehand
        index.add_items(self.candidate_dataset.get_view(self.CANDIDATE_VIEW_IDX))
        logger.info(
            "Constructed HNSW index with {} elements in {}s.",
            index.get_current_count(),
            int(time.time() - t0),
        )

        return index

    def _query_index(self, query_emb: ArrayLike, k: int) -> NDArray[np.uint64]:
        """Query the HNSW index to retrieve top-k nearest neighbor candidate IDs.

        Args:
            query_emb: Numpy array of shape (D,) representing the query embedding.
            k: Number of nearest neighbors to retrieve.

        Returns:
            Numpy array of shape (k,) containing the retrieved candidate IDs.
        """
        if self.hnsw_index.ef < k:
            logger.warning(
                "HNSW EF parameter ({}) is less than required minimum ({}). Increasing ef to {}.",
                self.hnsw_index.ef,
                k,
                k * 2,
            )
            self.hnsw_index.set_ef(k * 2)

        try:
            pred_idx, _ = self.hnsw_index.knn_query(query_emb, k=k)
        except RuntimeError:
            logger.error("Error querying HNSW index (k: {}, ef: {})", k, self.hnsw_index.ef)
            raise
        return pred_idx

    def _load_dataframe(self, file_path: Path) -> pl.DataFrame:
        """Load a data file into a Polars DataFrame with automatic format detection.

        Supports multiple common tabular data formats used in table augmentation benchmarks.

        Args:
            file_path: Path to the data file.

        Returns:
            Loaded DataFrame with all columns as strings by default.

        Raises:
            ValueError: If the file format is not supported.
        """
        if file_path.suffix.lower() == ".csv":
            return pl.read_csv(file_path)
        if file_path.suffix.lower() == ".tsv":
            return pl.read_csv(file_path, separator="\t")
        if file_path.suffix.lower() in {".pq", ".parquet"}:
            return pl.read_parquet(file_path)

        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _filter_gt(
        self, ground_truth: pl.DataFrame, valid_ids: pl.Series, include_col_level: bool = True
    ) -> pl.DataFrame:
        """Filter ground truth to only include candidates present in the dataset collection.

        This is important because benchmark ground truth may reference tables/columns that are not
        included in the current dataset collection, leading to evaluation against impossible
        targets.
        The filtering ensures fair evaluation by only considering candidates that could actually
        be retrieved.

        Args:
            ground_truth: DataFrame containing ground truth annotations.
            valid_ids: Series of valid candidate identifiers present in the dataset collection.
            include_col_level: Whether to include column-level filtering (True for join discovery)
                or only table-level filtering (False for union discovery).

        Returns:
            Filtered ground truth DataFrame with only reachable candidates.
        """
        # Replicate the logic of sanitize_col_name() in Polars expressions
        # This ensures column names match the format used in dataset identifiers
        # sanitize_col_name does:
        # 1. If col == "*", return r"\*"
        # 2. Replace "::" with ":\:"
        # 3. If starts with "^" and ends with "$", wrap with "\\"

        # Construct candidate identifiers in the same format as the dataset
        if include_col_level:
            # Apply column name sanitization logic
            no_double_colon = pl.col("candidate_column").str.replace_all("::", ":\\:")
            sanitized_col = (
                pl.when(pl.col("candidate_column") == "*")
                .then(pl.lit(r"\*"))
                .otherwise(
                    pl.when(
                        no_double_colon.str.starts_with("^") & no_double_colon.str.ends_with("$")
                    )
                    .then(pl.lit("\\") + no_double_colon + pl.lit("\\"))
                    .otherwise(no_double_colon)
                )
            )

            # Format: "table_name::column_name" for column-level tasks (join discovery)
            candidate_id = pl.col("candidate_table") + pl.lit("::") + sanitized_col
        else:
            # Format: "table_name" for table-level tasks (union discovery)
            candidate_id = pl.col("candidate_table")
            # Extract table-level IDs from column-level dataset identifiers
            valid_ids = valid_ids.str.split("::").list.get(0)

        # Filter to only include ground truth entries with valid candidates
        filtered_gt = ground_truth.filter(candidate_id.is_in(valid_ids))
        if len(ground_truth) != len(filtered_gt):
            logger.warning(
                "Filtered ground truth from {} to {} entries based on candidate presence in "
                "dataset collection.",
                len(ground_truth),
                len(filtered_gt),
            )
            logger.debug(
                "The following entries were not found:\n{}",
                ground_truth.filter(~candidate_id.is_in(valid_ids)),
            )

        return filtered_gt
