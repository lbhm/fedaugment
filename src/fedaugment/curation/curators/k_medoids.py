from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from tqdm.auto import tqdm

from .curator import Curator


def _pairwise_distance_fn(
    metric: Literal["euclidean", "cosine"],
) -> Callable[[Tensor, Tensor], Tensor]:
    """Make a function that computes distances between two sets of points.

    Args:
        metric: Distance metric to use.

    Returns:
        Function that takes (points1, points2) and returns distance matrix of shape (n1, n2).
    """
    if metric not in {"euclidean", "cosine"}:
        raise ValueError(f"Unsupported metric: {metric}")

    if metric == "euclidean":

        def distance_fn(points1: Tensor, points2: Tensor) -> Tensor:
            return torch.cdist(points1, points2, p=2)
    else:  # metric == "cosine"

        def distance_fn(points1: Tensor, points2: Tensor) -> Tensor:
            normalized1 = F.normalize(points1, p=2, dim=1, eps=1e-12)
            normalized2 = F.normalize(points2, p=2, dim=1, eps=1e-12)
            sims = normalized1 @ normalized2.T
            return 1 - sims

    return distance_fn


class KMedoids(Curator):
    """Curator that applies the k-medoids algorithm to pick representative indices."""

    def __init__(
        self,
        alias: str,
        metric: Literal["euclidean", "cosine"] = "euclidean",
        max_iters: int = 100,
    ) -> None:
        super().__init__(alias)
        if max_iters <= 0:
            raise ValueError("max_iters must be positive")
        self.metric: Literal["euclidean", "cosine"] = metric
        self.max_iters = max_iters
        self._distance_fn = _pairwise_distance_fn(metric)

    @torch.no_grad()
    def curate(self, candidates: Tensor, k: int, seed: int) -> Tensor:
        if candidates.dtype == torch.bfloat16:
            logger.warning("KMedoids does not support bfloat16; casting to float32")
            candidates = candidates.to(torch.float32)

        n = candidates.shape[0]
        device = candidates.device
        self._validate_inputs(candidates, k)
        if k == n:
            return torch.arange(n, dtype=torch.int64, device=device)

        generator = torch.Generator(device=device).manual_seed(seed)
        medoids = torch.randperm(n, generator=generator, device=device)[:k]

        for _ in tqdm(
            range(self.max_iters),
            desc="Iterating k-medoids",
            total=self.max_iters,
            leave=False,
            unit="it",
            mininterval=1.0,
            dynamic_ncols=True,
        ):
            distances_to_medoids = self._distance_fn(candidates, candidates[medoids])
            assignments = torch.argmin(distances_to_medoids, dim=1)
            new_medoids = medoids.clone()

            for cluster_idx in range(k):
                members = torch.where(assignments == cluster_idx)[0]
                if members.numel() == 0:
                    new_medoids[cluster_idx] = self._resample_medoid(generator, new_medoids, n)
                    continue

                # Compute intra-cluster distances for cluster members
                intra = self._distance_fn(candidates[members], candidates[members])
                costs = intra.sum(dim=1)
                best_member = members[torch.argmin(costs)]
                new_medoids[cluster_idx] = best_member

            if torch.equal(new_medoids, medoids):
                break
            medoids = new_medoids

        return medoids

    def _resample_medoid(self, generator: torch.Generator, medoids: Tensor, n: int) -> Tensor:
        while True:
            candidate = torch.randint(0, n, (1,), generator=generator, device=medoids.device)
            if not (medoids == candidate).any().item():
                return candidate
