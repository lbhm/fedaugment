import torch
from torch import Tensor

from fedaugment.projections.metrics.base import ChunkMetricBase


class TopKSimilarityBase(ChunkMetricBase):
    """Base class for top-k similarity metrics from streamed similarity chunks."""

    def __init__(self, name: str, k: int) -> None:
        super().__init__(name=name)
        self.k = k

        # State variables
        self.total_items: int = 0
        self.device: torch.device | None = None
        self.all_labels: Tensor | None = None

        # Per-row top-k tracking (maintained globally across chunks)
        self.topk_values: Tensor | None = None
        self.topk_indices: Tensor | None = None

        # Merge buffers (lazily allocated)
        self.merged_vals_buffer: Tensor | None = None
        self.merged_idx_buffer: Tensor | None = None

    def reset_state(self, labels: Tensor, device: torch.device) -> None:
        self.all_labels = labels
        self.total_items = labels.shape[0]
        self.device = device

        # Allocate global top-k buffers (recreate if size changed)
        # We check if the buffers already exist to avoid reallocation cost
        if (
            self.topk_values is not None
            and self.topk_values.shape == (self.total_items, self.k)
            and self.topk_values.device == device
        ):
            self.topk_values.fill_(-torch.inf)
        else:
            self.topk_values = torch.full(
                (self.total_items, self.k), -torch.inf, dtype=torch.float32, device=device
            )
        min_long = torch.iinfo(torch.long).min
        if (
            self.topk_indices is not None
            and self.topk_indices.shape == (self.total_items, self.k)
            and self.topk_indices.device == device
        ):
            self.topk_indices.fill_(min_long)
        else:
            self.topk_indices = torch.full(
                (self.total_items, self.k), min_long, dtype=torch.long, device=device
            )

    # torch.compile does not work for this function due to too many control flow paths
    @torch.compile(dynamic=True, disable=True)
    def update_from_chunk(
        self,
        sim_chunk: Tensor,
        row_labels: Tensor,
        col_labels: Tensor,
        row_start: int,
        col_start: int,
        self_sim_mask: Tensor,
    ) -> None:
        if self.device is None or self.topk_values is None or self.topk_indices is None:
            raise RuntimeError("reset_state() must be called before update_from_chunk")

        n_rows, n_cols = sim_chunk.shape

        # Extract top-k from this chunk
        k_chunk = min(self.k, n_cols)
        chunk_topk_vals, chunk_topk_local_idx = torch.topk(
            sim_chunk, k=k_chunk, dim=1, sorted=False
        )  # (n_rows, k_chunk)

        # Convert local column indices to global indices
        chunk_topk_global_idx = chunk_topk_local_idx + col_start

        # Retrieve current global top-k for these rows
        current_topk_vals = self.topk_values[row_start : row_start + n_rows]  # (n_rows, k)
        current_topk_idx = self.topk_indices[row_start : row_start + n_rows]  # (n_rows, k)

        # Lazy allocation: create merge buffers on first use, reuse thereafter
        # Buffer size depends on chunk dimensions (n_rows) and total candidates (k + k_chunk)
        buffer_shape = (n_rows, self.k + k_chunk)
        if (
            self.merged_vals_buffer is None
            or self.merged_idx_buffer is None
            or self.merged_vals_buffer.shape != buffer_shape
        ):
            self.merged_vals_buffer = torch.empty(
                buffer_shape, dtype=torch.float32, device=self.device
            )
            self.merged_idx_buffer = torch.empty(
                buffer_shape, dtype=torch.long, device=self.device
            )

        # In-place merge using pre-allocated buffers and slicing
        self.merged_vals_buffer[:, : self.k] = current_topk_vals
        self.merged_vals_buffer[:, self.k : self.k + k_chunk] = chunk_topk_vals
        self.merged_idx_buffer[:, : self.k] = current_topk_idx
        self.merged_idx_buffer[:, self.k : self.k + k_chunk] = chunk_topk_global_idx

        # Select top-k from merged candidates
        new_topk_vals, sort_indices = torch.topk(self.merged_vals_buffer, k=self.k, dim=1)
        new_topk_idx = torch.gather(self.merged_idx_buffer, dim=1, index=sort_indices)

        # Update global state
        self.topk_values[row_start : row_start + n_rows] = new_topk_vals
        self.topk_indices[row_start : row_start + n_rows] = new_topk_idx

    def drop_buffers(self) -> None:
        # Free merge buffers to allow garbage collection when metric is inactive
        # These will be lazily allocated on reset_state() or first update_from_chunk() call
        self.topk_values = None
        self.topk_indices = None
        self.merged_vals_buffer = None
        self.merged_idx_buffer = None


class TopKSimilarityPrecision(TopKSimilarityBase):
    """Computes top-k similarity precision from streamed similarity chunks.

    This metric finds the k most similar items for each row and computes the proportion of those k
    items that have matching labels.
    """

    def __init__(self, k: int) -> None:
        super().__init__(f"top_{k}_precision", k)

    def finalize(self) -> dict[str, Tensor]:
        if self.all_labels is None or self.topk_indices is None:
            raise RuntimeError("Labels and top-k indices must be set before finalize")

        # Get labels and targets for top-k indices
        topk_labels = self.all_labels[self.topk_indices]  # (N, k)
        target_labels = self.all_labels.unsqueeze(1)  # (N, 1)

        # Count matches per row, normalize by k
        topk_precision = (topk_labels == target_labels).sum(dim=1).float().mean() / self.k

        return {self.name: topk_precision}


class TopKSimilarityAccuracy(TopKSimilarityBase):
    """Computes top-k similarity accuracy from streamed similarity chunks.

    This metric finds the k most similar items for each row and checks if at least one of those k
    items has a matching label.
    """

    def __init__(self, k: int) -> None:
        super().__init__(f"top_{k}_accuracy", k)

    def finalize(self) -> dict[str, Tensor]:
        if self.all_labels is None or self.topk_indices is None:
            raise RuntimeError("Labels and top-k indices must be set before finalize")

        # Get labels and targets for top-k indices
        topk_labels = self.all_labels[self.topk_indices]  # (N, k)
        target_labels = self.all_labels.unsqueeze(1)  # (N, 1)

        # Check for at least one match per row
        topk_accuracy = (topk_labels == target_labels).any(dim=1).float().mean()

        return {self.name: topk_accuracy}


class EntityStability(TopKSimilarityBase):
    """Computes entity stability metric from streamed similarity chunks.

    Entity stability measures how consistent the k-nearest neighbors of different views from the
    same entity (i.e., table column) are. It groups by label and computes the average pairwise
    overlap of top-k neighbors within each label group.

    The metric is normalized by k and ranges from 0 to 1, where:
    - 0: No overlap in top-k neighbors between entities of same label
    - 1: Perfect overlap (identical top-k neighbors for all entities with same label)

    Based on the Observatory paper, adapted for n>=2 models.
    """

    def __init__(self, k: int, scaled_normalize: bool = True) -> None:
        """Initialize entity stability metric.

        Args:
            k: Number of top similar items to consider
            scaled_normalize: If True, normalize by dividing by (k-1) instead of k to account for
                the fact that each element is not included in its own top-k set. This allows the
                metric to reach 1.0 for a perfect overlap. If False, the maximum possible overlap
                is k-1, so the metric ranges from 0 to (k-1)/k.
        """
        super().__init__(f"entity_stability_{k}", k)
        self.scaled_normalize = scaled_normalize

    def finalize(self) -> dict[str, Tensor]:
        if self.all_labels is None or self.topk_indices is None:
            raise RuntimeError("Labels and top-k indices must be set before finalize")

        device = self.all_labels.device
        total_overlap = torch.tensor(0.0, dtype=torch.float32, device=device)
        total_pairs = torch.tensor(0, dtype=torch.int32, device=device)
        for label in torch.unique(self.all_labels):
            # Find all rows with the current label
            label_indices = torch.where(self.all_labels == label)[0]  # Only take row indices
            # Each label occurs V (number of views) times
            V = label_indices.shape[0]
            if V <= 1:
                continue  # Skip labels with less than 2 instances (can't compute pairwise overlap)

            # Get top-k indices for all entities with this label
            label_topk = self.topk_indices[label_indices]  # (V, k)

            # Compute pairwise overlaps for all pairs of entities with this label
            # topk_indices[:, None, :, None] -> (V, 1, k, 1)
            # topk_indices[None, :, None, :] -> (1, V, 1, k)
            matches = label_topk[:, None, :, None] == label_topk[None, :, None, :]  # (V, V, k, k)

            # Count overlaps: sum over both k dimensions to get intersection size
            overlap_matrix = matches.any(dim=3).sum(dim=2, dtype=torch.float32)  # (V, V)

            # Extract upper triangular part (excluding diagonal) to get unique pairs
            i, j = torch.triu_indices(V, V, offset=1, device=label_topk.device)
            pairwise_overlaps = overlap_matrix[i, j]

            # Add to running totals
            total_overlap += pairwise_overlaps.sum()
            total_pairs += pairwise_overlaps.shape[0]

        # Average overlap across all pairs and normalize
        if total_pairs == 0:
            stability = torch.tensor(0.0, device=device)
        elif self.scaled_normalize:
            stability = total_overlap / (total_pairs * (self.k - 1))
        else:
            stability = total_overlap / (total_pairs * self.k)

        return {self.name: stability}
