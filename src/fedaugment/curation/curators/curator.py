from abc import ABC, abstractmethod

from torch import Tensor


class Curator(ABC):
    """Abstract base class for training data curators."""

    def __init__(self, alias: str) -> None:
        super().__init__()
        self.alias = alias

    @abstractmethod
    def curate(self, candidates: Tensor, k: int, seed: int) -> Tensor:
        """Curate training data from a collection of candidates and return selected indices.

        Args:
            candidates: Candidate data points of shape (N, d).
            k: Number of data points to select.
            seed: Random seed for the curation process.

        Returns:
            int64 tensor with shape (k,) containing the selected indices.
        """

    def _validate_inputs(self, candidates: Tensor, k: int) -> None:
        expected_dims = 2
        n = candidates.shape[0]

        # Input validation
        if candidates.dim() != expected_dims:
            raise ValueError(f"candidates must be 2D, got shape {candidates.shape}")
        if n == 0:
            raise ValueError("candidates cannot be empty")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if k > n:
            raise ValueError(f"k ({k}) cannot exceed number of candidates ({n})")
