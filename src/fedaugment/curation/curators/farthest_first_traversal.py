from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm

from .curator import Curator


def _distance_fn(metric: Literal["euclidean", "cosine"]) -> Callable[[Tensor, Tensor], Tensor]:
    """Factory function to create a distance computation function based on the metric."""
    if metric not in {"euclidean", "cosine"}:
        raise ValueError(f"Unsupported metric: {metric}")

    if metric == "euclidean":

        def distance_fn(candidates: Tensor, center_idx: Tensor) -> Tensor:
            center = candidates[center_idx]
            diff = candidates - center
            return torch.linalg.norm(diff, ord=2, dim=1)  # type: ignore[no-any-return]
    else:  # metric == "cosine"

        def distance_fn(candidates: Tensor, center_idx: Tensor) -> Tensor:
            center = candidates[center_idx]
            sims = (candidates * center).sum(dim=1)
            return 1 - sims

    return distance_fn


class FarthestFirstTraversal(Curator):
    """K-center greedy curator that selects k farthest points under a chosen metric.

    Selects k points from the candidates such that each newly selected point is the farthest from
    the set of already selected points. This is done iteratively starting from a given initial
    point.
    """

    def __init__(self, alias: str, metric: Literal["euclidean", "cosine"] = "euclidean") -> None:
        super().__init__(alias=alias)
        self._distance_fn = _distance_fn(metric)
        self.metric: Literal["euclidean", "cosine"] = metric

    @torch.no_grad()
    def curate(self, candidates: Tensor, k: int, seed: int) -> Tensor:
        n = candidates.shape[0]
        device = candidates.device
        self._validate_inputs(candidates, k)

        if self.metric == "cosine":
            candidates = F.normalize(candidates, p=2, dim=1, eps=1e-12)

        # Pick random start
        generator = torch.Generator(device=device).manual_seed(seed)
        next_index = torch.randint(0, n, (1,), device=device, generator=generator).squeeze(0)

        selected = torch.empty((k,), dtype=torch.int64, device=device)
        selected[0] = next_index
        min_dists = torch.full((n,), torch.inf, device=device, dtype=candidates.dtype)

        for i in tqdm(
            range(1, k),
            desc="Selecting points",
            total=k,
            leave=False,
            unit="point",
            mininterval=1.0,
            dynamic_ncols=True,
            initial=1,
        ):
            # Compute distances from the last selected point
            new_dists = self._distance_fn(candidates, next_index)

            # Update minimum distances to the selected set
            min_dists = torch.minimum(min_dists, new_dists)
            argmax = torch.argmax(min_dists)
            selected[i] = argmax
            next_index = argmax

        return selected
