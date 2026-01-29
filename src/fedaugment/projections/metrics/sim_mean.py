import abc

import torch
from loguru import logger
from torch import Tensor

from fedaugment.projections.metrics.base import ChunkMetricBase


class SimilarityMeanBase(ChunkMetricBase):
    """Base class for metrics that compute mean similarity from streamed chunks."""

    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.device: torch.device | None = None
        self.total_similarity: Tensor | None = None
        self.total_count: Tensor | None = None

    def reset_state(self, labels: Tensor, device: torch.device) -> None:
        self.device = device
        self.total_similarity = torch.tensor(0.0, dtype=torch.float32, device=device)
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
        if self.total_similarity is None or self.total_count is None:
            raise RuntimeError("reset_state() must be called before update_from_chunk")

        # Compute mask for valid values
        valid_mask = self._compute_mask(self_sim_mask, row_labels, col_labels)

        # Accumulate sum and count
        # NOTE: Since the mask is dense, multiplication is more efficient than indexing
        self.total_similarity += (sim_chunk * valid_mask).sum().float()
        self.total_count += valid_mask.sum()

    def finalize(self) -> dict[str, Tensor]:
        if self.total_similarity is None or self.total_count is None:
            raise RuntimeError("Accumulators must be initialized before finalize")

        if self.total_count == 0:
            logger.warning("No similarity values found for {}; returning 0.0.", self.name)
            return {self.name: torch.tensor(0.0, device=self.device)}

        mean_similarity = self.total_similarity / self.total_count.float()
        return {self.name: mean_similarity}


class CosineSimilarityMean(SimilarityMeanBase):
    """Computes mean cosine similarity (excluding self-similarities) from streamed chunks.

    This metric accumulates all non-self similarity values across chunks and computes their mean.
    """

    def __init__(self) -> None:
        super().__init__(name="mean_cosine_similarity")

    def _compute_mask(
        self, self_sim_mask: Tensor, row_labels: Tensor, col_labels: Tensor
    ) -> Tensor:
        return self_sim_mask


class SameLabelCosineSimilarityMean(SimilarityMeanBase):
    """Computes mean cosine similarity for same-label pairs from streamed chunks.

    This metric accumulates similarity values only for pairs with matching labels (excluding
    self-similarities) and computes their mean.
    """

    def __init__(self) -> None:
        super().__init__(name="mean_same_label_cosine_similarity")

    def _compute_mask(
        self, self_sim_mask: Tensor, row_labels: Tensor, col_labels: Tensor
    ) -> Tensor:
        same_label_mask = row_labels.unsqueeze(1) == col_labels.unsqueeze(0)
        return same_label_mask & self_sim_mask
