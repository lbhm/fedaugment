import torch
from loguru import logger
from torch import Tensor

from fedaugment.projections.metrics.base import ChunkMetricBase


class MeanRelativeChunkRank(ChunkMetricBase):
    """Computes mean relative rank per chunk from streamed similarity chunks.

    For each chunk, finds the best same-label match and computes its rank within that chunk
    (i.e., how many items in the chunk have higher similarity). The final metric is the average
    of these chunk-local relative ranks.
    """

    def __init__(self) -> None:
        super().__init__(name="mean_relative_chunk_rank")
        self.device: torch.device | None = None
        self.total_rank: Tensor | None = None
        self.total_count: Tensor | None = None

    def reset_state(self, labels: Tensor, device: torch.device) -> None:
        self.device = device
        self.total_rank = torch.tensor(0.0, dtype=torch.float32, device=device)
        self.total_count = torch.tensor(0, dtype=torch.long, device=device)

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
        if self.total_rank is None or self.total_count is None:
            raise RuntimeError("reset_state() must be called before update_from_chunk")

        # Identify valid same-label pairs (exclude self-similarities via mask)
        same_label_mask = (col_labels.unsqueeze(0) == row_labels.unsqueeze(1)) & self_sim_mask
        # Identify rows with at least one same-label match
        has_candidate = same_label_mask.any(dim=1)  # (n_rows,)
        if not has_candidate.any():
            return

        # Filter out invalid similarities so downstream comparisons ignore them
        same_label_scores = sim_chunk.masked_fill(~same_label_mask, float("-inf"))
        best_same_label_sim = same_label_scores.max(dim=1).values[has_candidate]

        rank = (sim_chunk[has_candidate] > best_same_label_sim.unsqueeze(1)).sum(dim=1).float()
        valid_counts = self_sim_mask[has_candidate].sum(dim=1).float()
        denom = torch.clamp(valid_counts - 1, min=1.0)  # Account for zero-based indexing
        relative_rank = 1.0 - (rank / denom)

        # Accumulate
        self.total_rank += relative_rank.sum()
        self.total_count += has_candidate.sum()

    def finalize(self) -> dict[str, Tensor]:
        if self.total_rank is None or self.total_count is None:
            raise RuntimeError("Accumulators must be initialized before finalize")

        if self.total_count == 0:
            logger.warning("No values accumulated for {}; returning 0.0.", self.name)
            mean_rank = torch.tensor(0.0, device=self.device)
        else:
            mean_rank = self.total_rank / self.total_count.float()

        return {self.name: mean_rank}
