import abc

import torch
from loguru import logger
from torch import Tensor

from fedaugment.projections.metrics.base import ChunkMetricBase


class SimilarityStdDevBase(ChunkMetricBase):
    """Base class for metrics that compute standard deviation of similarity from streamed chunks.

    Uses the computational formula: Var(X) = E[X**2] - E[X]**2 to compute variance in a streaming
    fashion, then takes the square root for standard deviation.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.device: torch.device | None = None
        self.total_similarity: Tensor | None = None
        self.total_similarity_squared: Tensor | None = None
        self.total_count: Tensor | None = None

    def reset_state(self, labels: Tensor, device: torch.device) -> None:
        self.device = device
        self.total_similarity = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.total_similarity_squared = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.total_count = torch.tensor(0, dtype=torch.long, device=device)

    @abc.abstractmethod
    def _compute_mask(
        self, self_sim_mask: Tensor, row_labels: Tensor, col_labels: Tensor
    ) -> Tensor:
        """Compute the mask for selecting which similarity values to include.

        Returns:
            Tensor: Boolean mask of shape (n_rows, n_cols) indicating which values to include.
        """

    @torch.compile(dynamic=True)
    def update_from_chunk(
        self,
        sim_chunk: Tensor,
        row_labels: Tensor,
        col_labels: Tensor,
        row_start: int,
        col_start: int,
        self_sim_mask: Tensor,
    ) -> None:
        if (
            self.total_similarity is None
            or self.total_similarity_squared is None
            or self.total_count is None
        ):
            raise RuntimeError("reset_state() must be called before update_from_chunk")

        # Compute mask for valid values
        valid_mask = self._compute_mask(self_sim_mask, row_labels, col_labels)

        # Accumulate sum, sum of squares, and count
        # NOTE: Since the mask is dense, multiplication is more efficient than indexing
        masked_sim_chunk = sim_chunk * valid_mask
        self.total_similarity += masked_sim_chunk.sum().float()
        self.total_similarity_squared += (masked_sim_chunk**2).sum().float()
        self.total_count += valid_mask.sum()

    def finalize(self) -> dict[str, Tensor]:
        if (
            self.total_similarity is None
            or self.total_similarity_squared is None
            or self.total_count is None
        ):
            raise RuntimeError("Accumulators must be initialized before finalize")

        if self.total_count == 0:
            logger.warning("No similarity values found for {}; returning 0.0.", self.name)
            return {self.name: torch.tensor(0.0, device=self.device)}

        mean = self.total_similarity / self.total_count.float()
        mean_squared = self.total_similarity_squared / self.total_count.float()
        variance = mean_squared - mean**2
        variance = torch.clamp(variance, min=0.0)  # In case of numerical instability
        std_dev = torch.sqrt(variance)

        return {self.name: std_dev}


class CosineSimilarityStdDev(SimilarityStdDevBase):
    """Computes standard deviation of cosine similarity (excluding self-similarities).

    This metric accumulates all non-self similarity values and their squares across chunks,
    then computes the standard deviation using the computational formula.
    """

    def __init__(self) -> None:
        super().__init__(name="stddev_cosine_similarity")

    def _compute_mask(
        self, self_sim_mask: Tensor, row_labels: Tensor, col_labels: Tensor
    ) -> Tensor:
        return self_sim_mask


class SameLabelCosineSimilarityStdDev(SimilarityStdDevBase):
    """Computes standard deviation of cosine similarity for same-label pairs.

    This metric accumulates similarity values only for pairs with matching labels (excluding
    self-similarities) and computes their standard deviation.
    """

    def __init__(self) -> None:
        super().__init__(name="stddev_same_label_cosine_similarity")

    def _compute_mask(
        self, self_sim_mask: Tensor, row_labels: Tensor, col_labels: Tensor
    ) -> Tensor:
        same_label_mask = row_labels.unsqueeze(1) == col_labels.unsqueeze(0)
        return same_label_mask & self_sim_mask
