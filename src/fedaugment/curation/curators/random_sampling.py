import torch
from torch import Tensor

from .curator import Curator


class RandomSampling(Curator):
    """Randomly samples candidate indices."""

    def __init__(self, alias: str) -> None:
        super().__init__(alias)

    @torch.no_grad()
    def curate(self, candidates: Tensor, k: int, seed: int) -> Tensor:
        device = candidates.device
        n = candidates.shape[0]
        self._validate_inputs(candidates, k)

        generator = torch.Generator(device=device).manual_seed(seed)
        return torch.randperm(n, generator=generator, device=device)[:k]
