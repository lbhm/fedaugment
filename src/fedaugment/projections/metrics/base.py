import abc
from typing import Any, override

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from tqdm.auto import tqdm


class ChunkMetricBase(abc.ABC):
    """Base class for metrics that aggregate results from streamed similarity chunks."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def reset_state(self, labels: Tensor, device: torch.device) -> None:
        """Reset (or initialize) internal state before processing chunks."""

    @abc.abstractmethod
    def update_from_chunk(
        self,
        sim_chunk: Tensor,
        row_labels: Tensor,
        col_labels: Tensor,
        row_start: int,
        col_start: int,
        self_sim_mask: Tensor,
    ) -> None:
        """Update internal accumulators from a chunk of the similarity matrix.

        Args:
            sim_chunk (Tensor): Similarity values for this chunk (n_rows, n_cols).
            row_labels (Tensor): Labels for rows in this chunk (n_rows,).
            col_labels (Tensor): Labels for columns in this chunk (n_cols,).
            row_start (int): Global starting index for rows.
            col_start (int): Global starting index for columns.
            self_sim_mask (Tensor): Boolean mask (n_rows, n_cols), False for self-similarities.
        """

    @abc.abstractmethod
    def finalize(self) -> dict[str, Tensor]:
        """Finalize and return computed metric(s) as a dictionary."""

    def drop_buffers(self) -> None:  # noqa: B027
        """Drop any internal buffers to free memory."""
        # Default implementation does nothing; override in subclasses if needed


class CosineSimilarityOrchestrator(Metric):
    projections: list[Tensor]
    labels: list[Tensor]

    def __init__(
        self, chunk_size: int | None = None, use_half_precision: bool = False, **kwargs: Any
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.use_half_precision = use_half_precision

        self.add_state("projections", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")

        # Registered children (BaseChunkMetric instances)
        self._children: list[ChunkMetricBase] = []

        # Cache + lock to avoid duplicated heavy compute
        self._cached_results: dict[str, Tensor] | None = None
        self._is_dirty = True  # True if cache is stale / not present

    def register_metric(self, child_metric: ChunkMetricBase) -> None:
        """Register a child metric to be computed by this orchestrator.

        Args:
            child_metric (ChunkMetricBase): Child metric instance to register.
        """
        if child_metric in self._children:
            logger.warning("Child metric {} is already registered.", child_metric.name)
            return
        self._children.append(child_metric)

    @override
    def update(self, projections: Tensor, labels: Tensor) -> None:
        """Store projections and labels for later aggregation.

        Args:
            projections (Tensor): Projections of shape (B * V, D).
            labels (Tensor): Corresponding labels of shape (B * V,).
        """
        self.projections.append(projections.detach())
        self.labels.append(labels.detach())

        # Invalidate cache
        self._is_dirty = True
        self._cached_results = None

    @override
    def compute(self) -> dict[str, Tensor]:
        # Only compute if cache is dirty
        if (not self._is_dirty) and (self._cached_results is not None):
            return self._cached_results

        projections = dim_zero_cat(self.projections)  # (N*B*V, D)
        labels = dim_zero_cat(self.labels)  # (N*B*V,)

        device = projections.device
        dtype = torch.float16 if self.use_half_precision else projections.dtype
        n_projections = projections.shape[0]
        chunk_size = self.chunk_size or n_projections

        # We log this to detect potential race conditions in multi-threaded scenarios
        logger.debug("Computing CosineSimilarityOrchestrator metrics.")
        if n_projections % chunk_size != 0:
            logger.warning(
                "Number of projections ({}) is not divisible by chunk size ({}). "
                "This may lead to suboptimal performance.",
                n_projections,
                chunk_size,
            )

        # Reset children states and notify children of total size before processing
        for c in self._children:
            c.reset_state(labels, device)

        # Normalize
        projections = F.normalize(projections.to(dtype), dim=1)  # (N*B*V, D)

        # Pre-allocate self-similarity mask buffer to avoid repeated allocations
        # This buffer is reused across all chunk iterations
        self_sim_mask_buffer = torch.zeros(chunk_size, chunk_size, dtype=torch.bool, device=device)

        # Process similarity matrix in chunks of (C x C)
        with tqdm(
            desc="Similarity chunks",
            total=((n_projections + chunk_size - 1) // chunk_size) ** 2,
            leave=False,
            unit="chunk",
            mininterval=1.0,
            dynamic_ncols=True,
        ) as pbar:
            for row_start in range(0, n_projections, chunk_size):
                row_end = min(n_projections, row_start + chunk_size)
                n_rows = row_end - row_start

                # Row-chunk labels
                row_labels = labels[row_start:row_end]  # (C,)

                # Iterate over columns
                for col_start in range(0, n_projections, chunk_size):
                    col_end = min(n_projections, col_start + chunk_size)
                    n_cols = col_end - col_start
                    col_labels = labels[col_start:col_end]  # (C,)

                    # Reuse pre-allocated buffer (get view of appropriate size)
                    self_sim_mask_buffer.fill_(False)  # Reset mask to all False
                    self_sim_mask = self_sim_mask_buffer[:n_rows, :n_cols]

                    # Mark diagonal elements (self-similarities) as True
                    if row_start < col_end and col_start < row_end:
                        overlap_start = max(row_start, col_start)
                        overlap_end = min(row_end, col_end)
                        if overlap_start < overlap_end:
                            # Compute diagonal positions using vectorized operations
                            row_offset = overlap_start - row_start
                            col_offset = overlap_start - col_start
                            diag_length = overlap_end - overlap_start

                            # Use vectorized indexing instead of Python loop
                            diag_indices = torch.arange(diag_length, device=device)
                            self_sim_mask[row_offset + diag_indices, col_offset + diag_indices] = (
                                True
                            )

                    # Compute similarity block (C x C)
                    sim_chunk = projections[row_start:row_end] @ projections[col_start:col_end].T
                    # NOTE: Do not set self-similarities to -inf because -inf * 0 = nan, which
                    # breaks multiplication-based filtering. sim_chunk.dtype.min is below the valid
                    # cosine similarity range [-1, 1] and gets zeroed by mask multiplication.
                    sim_chunk[self_sim_mask] = torch.finfo(sim_chunk.dtype).min

                    # Update each child with the same chunk
                    self_sim_mask = ~self_sim_mask  # Invert mask to indicate valid similarities
                    for c in self._children:
                        c.update_from_chunk(
                            sim_chunk, row_labels, col_labels, row_start, col_start, self_sim_mask
                        )
                    pbar.update(1)

        # Finalize children metrics
        results: dict[str, Tensor] = {}
        for c in self._children:
            results.update(c.finalize())
            c.drop_buffers()

        logger.debug("Finalized CosineSimilarityOrchestrator metrics")
        self._cached_results = results
        self._is_dirty = False
        return self._cached_results


class OrchestratorChild(Metric):
    """Wrapper that exposes a CosineSimilarityOrchestrator child in a Metric-compatible manner."""

    def __init__(
        self,
        orchestrator: CosineSimilarityOrchestrator,
        child_metric: ChunkMetricBase,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.orchestrator = orchestrator
        self.child_metric = child_metric
        self.orchestrator.register_metric(child_metric)

    @property
    def update_called(self) -> bool:
        """The underlying orchestrator handles updates."""
        return self.orchestrator.update_called

    @override
    def update(self, *args: Any, **kwargs: Any) -> None:
        """No-op update since updates must go to the orchestrator."""
        logger.warning(
            "update() called on child metric, but updates should go to the orchestrator."
        )

    @override
    def compute(self) -> Tensor:
        """Request metric value from orchestrator."""
        metrics = self.orchestrator.compute()
        return metrics[self.child_metric.name]

    @override
    def reset(self) -> None:
        super().reset()
        self.orchestrator.reset()
