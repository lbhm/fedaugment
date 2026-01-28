"""Metric implementations from the paper `Integrating Vector Databases across Embedding Models`."""

from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from torch.linalg import matrix_norm, svd
from torchmetrics import Metric


class NormalizedResidualNorm(Metric):
    """Measures the quality of embedding alignment as defined in the LA2M paper.

    This metric computes the Frobenius norm of the difference between aligned embeddings and their
    targets, normalized by the number of samples. Corresponds to paper's alpha.
    """

    total_samples: Tensor
    sum_squared_error: Tensor

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, aligned_embeddings: Tensor, target_embeddings: Tensor) -> None:
        """Update metric state with aligned and target embeddings.

        Args:
            aligned_embeddings: Aligned embeddings (N, D)
            target_embeddings: Target embeddings (N, D)
        """
        # Compute squared Frobenius norm
        diff = aligned_embeddings - target_embeddings
        squared_error = torch.sum(diff**2)

        self.sum_squared_error += squared_error
        self.total_samples += aligned_embeddings.shape[0]

    def compute(self) -> Tensor:
        # Normalize by the number of samples before completing Frobenius norm computation
        if self.total_samples == 0:
            logger.warning(
                "NormalizedResidualNorm.compute() called before any samples were observed; "
                "returning 0.0."
            )
            return torch.tensor(0.0, device=self.sum_squared_error.device)
        return torch.sqrt(self.sum_squared_error / self.total_samples.float())


class LocalDistanceCorrelation(Metric):
    """Measures how well local isometry (i.e., distances) is preserved in the mapped space."""

    isometry_preservation: Tensor
    total_comparisons: Tensor

    MIN_STD_THRESHOLD: ClassVar[float] = 1e-8

    def __init__(self, k_neighbors: int = 5, batch_size: int | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.k_neighbors = k_neighbors
        self.batch_size = batch_size
        self.add_state("isometry_preservation", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_comparisons", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, source_embeddings: Tensor, aligned_embeddings: Tensor) -> None:
        """Update metric with source and aligned embeddings.

        Args:
            source_embeddings: Original embeddings (N, D_s)
            aligned_embeddings: Aligned embeddings (N, D_r)
        """
        N = source_embeddings.shape[0]
        if self.k_neighbors + 1 > N:
            return  # Not enough samples

        # Process embeddings in batches to reduce memory usage
        batch_size = self.batch_size or N
        total_preservation = torch.tensor(0.0, device=source_embeddings.device)
        n_comparisons = torch.tensor(0, device=source_embeddings.device)

        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)
            batch_indices = torch.arange(batch_start, batch_end, device=source_embeddings.device)
            B = batch_end - batch_start

            # Get batch embeddings
            source_batch = source_embeddings[batch_indices]  # (B, D_s)
            aligned_batch = aligned_embeddings[batch_indices]  # (B, D_r)

            # Compute distances only for this batch against all embeddings
            source_dist_batch = torch.cdist(source_batch, source_embeddings)  # (B, N)
            aligned_dist_batch = torch.cdist(aligned_batch, aligned_embeddings)  # (B, N)

            # Find k+1 nearest neighbors in source space for all batch samples
            _, source_neighbors = torch.topk(
                source_dist_batch, self.k_neighbors + 1, largest=False, dim=1
            )  # (B, k+1)
            source_neighbors = source_neighbors[:, 1:]  # (B, k) - exclude self

            # Gather distances for k-nearest neighbors
            batch_idx = (
                torch.arange(B, device=source_embeddings.device)
                .unsqueeze(1)
                .expand(-1, self.k_neighbors)
            )  # (B, k)

            # Extract neighbor distances in both spaces: (B, k)
            source_neighbor_dists = source_dist_batch[batch_idx, source_neighbors]
            aligned_neighbor_dists = aligned_dist_batch[batch_idx, source_neighbors]

            # Compute unbiased standard deviations: (B,)
            source_std = source_neighbor_dists.std(dim=1, unbiased=True)
            aligned_std = aligned_neighbor_dists.std(dim=1, unbiased=True)

            # Compute Pearson correlation: r = cov(X,Y) / (std(X) * std(Y))
            # Center the data: (B, k)
            source_centered = source_neighbor_dists - source_neighbor_dists.mean(
                dim=1, keepdim=True
            )
            aligned_centered = aligned_neighbor_dists - aligned_neighbor_dists.mean(
                dim=1, keepdim=True
            )

            # Covariance and correlation: (B,)
            cov = (source_centered * aligned_centered).sum(dim=1) / (self.k_neighbors - 1)
            correlations = cov / (source_std * aligned_std + 1e-10)

            # Validity mask: (B,)
            valid_mask = (
                (source_std > self.MIN_STD_THRESHOLD)
                & (aligned_std > self.MIN_STD_THRESHOLD)
                & ~torch.isnan(correlations)
            )

            # Accumulate valid correlations
            if valid_mask.any():
                total_preservation += correlations[valid_mask].abs().sum()
                n_comparisons += valid_mask.sum()

        if n_comparisons > 0:
            self.isometry_preservation += total_preservation
            self.total_comparisons += n_comparisons

    def compute(self) -> Tensor:
        if self.total_comparisons == 0:
            logger.warning(
                "LocalDistanceCorrelation.compute() called before any samples were observed; "
                "returning 0.0."
            )
            return torch.tensor(0.0, device=self.isometry_preservation.device)
        return self.isometry_preservation / self.total_comparisons.float()


class AlignmentConsistency(Metric):
    """Measures consistency of aligned embeddings across different views.

    The metric computes the average absolute cosine similarity between aligned projections of the
    same sample from multiple views. Corresponds to paper's beta.
    """

    consistency_sum: Tensor
    total_pairs: Tensor

    MIN_ALIGNED_VIEWS: ClassVar[int] = 2

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("consistency_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_pairs", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, aligned_projections: Tensor) -> None:
        """Update metric with aligned projections from multiple views.

        Args:
            aligned_projections: Aligned projections (B, V, D) where V is number of views
        """
        _, V, _ = aligned_projections.shape

        # We need at least two views to compute alignment consistency
        if V < self.MIN_ALIGNED_VIEWS:
            return

        # Generate all unique view pair indices
        pairs = torch.combinations(torch.arange(V), r=2)  # (num_pairs, 2)

        # For each pair index, get the corresponding view for all batch samples
        view_pairs_i = aligned_projections[:, pairs[:, 0], :]  # (B, num_pairs, D)
        view_pairs_j = aligned_projections[:, pairs[:, 1], :]  # (B, num_pairs, D)

        # Compute cosine similarity for all pairs: (B, num_pairs)
        cosine_sims = F.cosine_similarity(view_pairs_i, view_pairs_j, dim=2)

        self.consistency_sum += cosine_sims.abs().sum()
        self.total_pairs += cosine_sims.numel()

    def compute(self) -> Tensor:
        if self.total_pairs == 0:
            logger.warning(
                "AlignmentConsistency.compute() called before any samples were observed; "
                "returning 0.0."
            )
            return torch.tensor(0.0, device=self.consistency_sum.device)
        return self.consistency_sum / self.total_pairs.float()


class ProcrustesError(Metric):
    """Measures the residual error after Procrustes alignment.

    This metric specifically evaluates the quality of Procrustes-based alignment by measuring the
    remaining error after optimal orthogonal transformation.
    """

    total_error: Tensor
    total_samples: Tensor

    def __init__(self, batch_size: int | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.add_state("total_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, source_embeddings: Tensor, target_embeddings: Tensor) -> None:
        """Update metric with source and target embeddings.

        Args:
            source_embeddings: Source embeddings before alignment (N, D)
            target_embeddings: Target embeddings (N, D)
        """
        N = source_embeddings.shape[0]
        batch_size = self.batch_size or N
        error = torch.tensor(0.0, device=source_embeddings.device)
        n_samples = 0

        # Process in batches to reduce memory usage
        for batch_start in range(0, N, batch_size):
            batch_end = min(batch_start + batch_size, N)

            # Get batch embeddings
            source_batch = source_embeddings[batch_start:batch_end]
            target_batch = target_embeddings[batch_start:batch_end]

            # Compute optimal Procrustes transformation for this batch
            source_centered = source_batch - torch.mean(source_batch, dim=0)
            target_centered = target_batch - torch.mean(target_batch, dim=0)

            # SVD for optimal rotation (cast to float32 for SVD compatibility)
            h_matrix = torch.mm(source_centered.T, target_centered).float()  # (D_s, D_t)
            u_matrix, _, vt_matrix = svd(h_matrix, full_matrices=False)
            # u_matrix: (D_s, min(D_s, D_t)), vt_matrix: (min(D_s, D_t), D_t)
            # r_matrix should be (D_s, D_t) for source_centered @ r_matrix to work
            r_matrix = torch.mm(u_matrix, vt_matrix)

            # Ensure proper rotation (cast to float32 for det compatibility)
            if torch.det(r_matrix.float()) < 0:
                vt_matrix[-1, :] *= -1
                r_matrix = torch.mm(vt_matrix.T, u_matrix.T)

            # Apply transformation and compute error for this batch
            aligned_source = torch.mm(source_centered, r_matrix)
            batch_error: Tensor = matrix_norm(aligned_source - target_centered, ord="fro") ** 2

            error += batch_error
            n_samples += source_batch.shape[0]

        self.total_error += error
        self.total_samples += n_samples

    def compute(self) -> Tensor:
        if self.total_samples == 0:
            logger.warning(
                "ProcrustesError.compute() called before any samples were observed; returning 0.0."
            )
            return torch.tensor(0.0, device=self.total_error.device)
        return torch.sqrt(self.total_error / self.total_samples.float())
