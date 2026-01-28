"""This module contains the Procrustes projection model.

It is based on the A2M method from the paper "Integrating Vector Databases across Embedding
Models".
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torchmetrics import Metric, MetricCollection

from fedaugment.config import (
    CriterionConfig,
    LRSchedulerConfig,
    OptimizerConfig,
    ProcrustesModuleConfig,
)
from fedaugment.projections.metrics import (
    AlignmentConsistency,
    CosineSimilarityMean,
    CosineSimilarityOrchestrator,
    EntityStability,
    NormalizedResidualNorm,
    OrchestratorChild,
    ProcrustesError,
    SameLabelCosineSimilarityMean,
)

from .projection_model import ProjectionModel


@dataclass
class ProcrustesParameters:
    """Parameters for Procrustes transformation."""

    rotation_matrix: Tensor  # (D_s, D_t)
    source_mean: Tensor  # (D_s,)
    target_mean: Tensor  # (D_t,)
    source_norm: float  # Frobenius norm of source embeddings
    target_norm: float  # Frobenius norm of target embeddings

    def to(self, device: torch.device) -> "ProcrustesParameters":
        return ProcrustesParameters(
            rotation_matrix=self.rotation_matrix.to(device),
            source_mean=self.source_mean.to(device),
            target_mean=self.target_mean.to(device),
            source_norm=self.source_norm,
            target_norm=self.target_norm,
        )


def compute_procrustes_parameters(
    source_view_emb: Tensor,
    target_view_emb: Tensor,
    approximate: bool = True,
    q: int = 1500,
    with_rotation: bool = True,
) -> ProcrustesParameters:
    """Compute Procrustes transformation parameters from source to target view.

    Args:
        source_view_emb: Source view embeddings of shape (B, D_s)
        target_view_emb: Target view embeddings of shape (B, D_t)
        approximate: Whether to use low-rank SVD approximation
        q: Number of singular values to keep for low-rank approximation
        with_rotation: Whether to compute the rotation matrix (if False, uses identity)

    Returns:
        ProcrustesParameters containing transformation parameters
    """
    # Compute centroids
    source_mean = source_view_emb.mean(dim=0)  # (D_s,)
    target_mean = target_view_emb.mean(dim=0)  # (D_t,)

    # Compute norms for normalization
    source_norm = float(torch.norm(source_view_emb).item())
    target_norm = float(torch.norm(target_view_emb).item())

    # Center and normalize the embeddings
    source_centered = (source_view_emb - source_mean) / source_norm  # (B, D_s)
    target_centered = (target_view_emb - target_mean) / target_norm  # (B, D_t)

    # Compute rotation matrix
    if with_rotation:
        # Compute covariance matrix
        covariance_matrix = torch.mm(source_centered.T, target_centered)  # (D_s, D_t)

        # Perform SVD
        if approximate:
            # torch.svd_lowrank returns V (not V^T)
            # i = min(q, rank(covariance_matrix)) ~ min(q, D_s, D_t)
            U, _, V = torch.svd_lowrank(covariance_matrix, q=q)  # (D_s, i), (i,), (D_t, i)
            rotation_matrix = torch.mm(U, V.T)  # (D_s, D_t)
        else:
            if covariance_matrix.size(0) != covariance_matrix.size(1):
                raise ValueError(
                    "Exact Procrustes analysis requires square covariance matrix. "
                    f"Got shape: {covariance_matrix.shape}"
                )
            U, _, Vt = torch.linalg.svd(
                covariance_matrix
            )  # (D_s, D_s), (min(D_s, D_t),), (D_t, D_t)
            rotation_matrix = torch.mm(U, Vt)

    else:
        # Use identity matrix (no rotation)
        rotation_matrix = torch.eye(
            source_centered.shape[1], target_centered.shape[1], device=source_view_emb.device
        )

    return ProcrustesParameters(
        rotation_matrix=rotation_matrix,
        source_mean=source_mean,
        target_mean=target_mean,
        source_norm=source_norm,
        target_norm=target_norm,
    )


def apply_procrustes_transformation(
    source_view_emb: Tensor, params: ProcrustesParameters
) -> Tensor:
    """Apply a precomputed Procrustes transformation to source view embeddings.

    Args:
        source_view_emb: Source view embeddings of shape (B, D_s) or (D_s,)
        params: Precomputed transformation parameters

    Returns:
        Transformed embeddings of shape (B, D_t) or (D_t,)
    """
    was_1d = source_view_emb.dim() == 1
    if was_1d:
        source_view_emb = source_view_emb.unsqueeze(0)

    # Center and normalize the source embeddings
    source_centered = source_view_emb - params.source_mean  # (B, D_s)
    source_normalized = source_centered / params.source_norm  # (B, D_s)

    # Apply rotation: (B, D_s) @ (D_s, D_t) = (B, D_t)
    transformed = torch.mm(source_normalized, params.rotation_matrix)

    # Denormalize and recenter
    result = transformed * params.target_norm + params.target_mean
    return result.squeeze(0) if was_1d else result


class ProcrustesModel(ProjectionModel):
    """Procrustes-based projection model for embedding alignment.

    This model implements the A2M method for aligning embeddings from different views. It learns an
    orthogonal transformation that minimizes the Frobenius norm between aligned embedding pairs.
    """

    def __init__(
        self,
        module_kwargs: ProcrustesModuleConfig,
        criterion: CriterionConfig,
        optim_configs: list[OptimizerConfig],
        sched_configs: list[LRSchedulerConfig],
        pipeline_names: list[str],
        embedding_dims: list[int],
        data_batch_size: int,
        metric_batch_size: int | None = None,
    ) -> None:
        orchestrator = CosineSimilarityOrchestrator(chunk_size=metric_batch_size)
        metrics_dict: dict[str, Metric | MetricCollection] = {
            "cossim_mean": OrchestratorChild(orchestrator, CosineSimilarityMean()),
            "same_lbl_cossim_mean": OrchestratorChild(
                orchestrator, SameLabelCosineSimilarityMean()
            ),
            "entity_stability_5": OrchestratorChild(orchestrator, EntityStability(k=5)),
            "entity_stability_15": OrchestratorChild(orchestrator, EntityStability(k=15)),
            "entity_stability_50": OrchestratorChild(orchestrator, EntityStability(k=50)),
            "mapping_quality": NormalizedResidualNorm(),
            "procrustes_error": ProcrustesError(),
            "alignment_consistency": AlignmentConsistency(),
        }
        metrics = MetricCollection(metrics_dict)

        super().__init__(
            module_kwargs=module_kwargs,
            criterion=criterion,
            optim_configs=optim_configs,
            sched_configs=sched_configs,
            pipeline_names=pipeline_names,
            embedding_dims=embedding_dims,
            data_batch_size=data_batch_size,
            metric_batch_size=metric_batch_size,
            metrics=metrics,
        )
        self.config = module_kwargs

        # Procrustes transformation parameters per view
        self.mapping_params: dict[int, ProcrustesParameters] = {}

        # Disable lightning's automatic optimization for non-parametric model
        self.automatic_optimization = False

    @property
    def output_dim(self) -> int:
        return self.embedding_dims[0]

    def training_step(
        self, batch: Sequence[Tensor], batch_idx: int, _dataloader_idx: int = 0
    ) -> None:
        """Perform a training step on the full dataset (no mini-batching).

        View 0 is used as the reference view.

        Since there is no optimizer step, this method does not return anything.
        The input is a list of V tensors, each of shape (B, D_i), where B is the batch/dataset size
        and D_i is the embedding dimension for view i.
        """
        reference_view_idx = 0
        if batch_idx > 0:
            raise ValueError(
                "ProcrustesProjectionModel does not support batching across samples. "
                "All samples must be processed in one batch."
            )
        if batch[reference_view_idx].numel() == 0:
            raise ValueError("Reference view embeddings are empty.")

        # The first view is the reference view so we don't need to compute a mapping for it
        reference_emb = batch[reference_view_idx]
        for view_idx, view_emb in enumerate(batch[1:], start=1):
            # Compute and store Procrustes transformation parameters
            self.mapping_params[view_idx] = compute_procrustes_parameters(
                view_emb,
                reference_emb,
                approximate=self.config.approximate,
                q=self.config.q,
                with_rotation=self.config.with_rotation,
            )

        # We do not compute a loss here since computing a contrastive loss over the entire dataset
        # is not feasible. Instead, we only compute evaluation metrics during val/test.

    def validation_step(
        self, batch: Sequence[Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._shared_eval_step(batch, batch_idx, self.val_metrics, self.val_orchestrator)

    def test_step(self, batch: Sequence[Tensor], batch_idx: int, dataloader_idx: int = 0) -> None:
        self._shared_eval_step(batch, batch_idx, self.test_metrics, self.test_orchestrator)

    def predict_step(self, batch: Sequence[Tensor]) -> Tensor:
        """Project multiple views to the reference view space.

        View 0 must be the reference view.

        Args:
            batch: A list of V tensors, each of shape (B, D_i), where B is the batch size and
                D_i is the embedding dimension for view i.

        Returns:
            A tensor of shape (B, V, D_r), where D_r is the dimension of the reference view.
        """
        if len(batch) - 1 != len(self.mapping_params):
            raise ValueError(
                f"This model was trained with {len(self.mapping_params) + 1} views "
                f"but received {len(batch)} views."
            )

        reference_view_idx = 0

        # Initialize output with reference view embeddings
        aligned_views = [batch[reference_view_idx]]
        for view_idx, view_emb in enumerate(batch[1:], start=1):
            # Get mapping parameters and move to correct device
            params = self.mapping_params[view_idx].to(view_emb.device)

            # Apply Procrustes transformation
            aligned_emb = apply_procrustes_transformation(view_emb, params)
            aligned_views.append(aligned_emb)

        return torch.stack(aligned_views, dim=1)  # (B, V, D)

    def project(self, batch: Tensor, view_idx: int) -> Tensor:
        """Project a single batch of embeddings to the reference view space.

        Args:
            batch: A tensor of shape (B, D_i), where B is the batch size and
                D_i is the embedding dimension for the input view.
            view_idx: Index of the embedding space/view that the batch comes from.
                View 0 is the reference view and is returned unchanged.

        Returns:
            A tensor of shape (B, D_out), where D_out is the output dimension of the
            projection (reference view dimension).
        """
        if view_idx == 0:
            # Reference view - no transformation needed
            return batch
        if view_idx not in self.mapping_params:
            raise ValueError(
                f"No mapping parameters found for view {view_idx}. "
                f"Available views: {list(self.mapping_params.keys())}"
            )

        # Get mapping parameters and move to correct device
        params = self.mapping_params[view_idx].to(batch.device)

        # Apply Procrustes transformation
        return apply_procrustes_transformation(batch, params)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """No optimizers needed for non-parametric model."""
        return None

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["config"] = self.config.model_dump()
        checkpoint["mapping_params"] = self.mapping_params

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.config = ProcrustesModuleConfig.model_validate(checkpoint["config"])
        self.mapping_params = checkpoint["mapping_params"]

    def _shared_eval_step(
        self,
        batch: Sequence[Tensor],
        batch_idx: int,
        metrics: MetricCollection,
        orchestrator: CosineSimilarityOrchestrator,
    ) -> None:
        B, _ = batch[0].shape  # (B, D_i)
        V = len(batch)  # number of views

        projections = self.predict_step(batch)  # (B, V, D)

        reference_view = projections[:, 0, :]  # (B, D)
        for view_idx, view_embeddings in enumerate(batch[1:], start=1):
            aligned_view = projections[:, view_idx, :]  # (B, D)

            metrics["mapping_quality"].update(aligned_view, reference_view)
            metrics["procrustes_error"].update(view_embeddings, aligned_view)
        metrics["alignment_consistency"].update(projections)

        # The last batch may be smaller than the batch size
        start_idx = self.data_batch_size * batch_idx
        end_idx = start_idx + min(self.data_batch_size, B)
        sample_labels = torch.arange(start_idx, end_idx, device=self.device).repeat_interleave(V)

        orchestrator.update(
            projections=projections.view(B * V, -1),  # (B * V, D)
            labels=sample_labels,  # (B * V,)
        )
        self.log_dict(metrics, on_epoch=True, batch_size=B)
