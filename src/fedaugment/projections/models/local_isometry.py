"""This module contains the LocalIsometry projection model.

It is based on the LA2M method from the paper "Integrating Vector Databases across Embedding
Models".
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from loguru import logger
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from torch import Tensor
from torchmetrics import Metric, MetricCollection

from fedaugment.config import (
    CriterionConfig,
    LocalIsometryModuleConfig,
    LRSchedulerConfig,
    OptimizerConfig,
)
from fedaugment.projections.metrics import (
    AlignmentConsistency,
    CosineSimilarityMean,
    CosineSimilarityOrchestrator,
    EntityStability,
    LocalDistanceCorrelation,
    NormalizedResidualNorm,
    OrchestratorChild,
    SameLabelCosineSimilarityMean,
)

from .projection_model import ProjectionModel


@dataclass
class ProcrustesMapping:
    rotation_matrix: Tensor  # (D_s, D_t)
    scaling_factor: float
    source_mean: Tensor  # (D_s,)
    target_mean: Tensor  # (D_t,)

    def to(self, device: torch.device) -> "ProcrustesMapping":
        return ProcrustesMapping(
            rotation_matrix=self.rotation_matrix.to(device),
            scaling_factor=self.scaling_factor,
            source_mean=self.source_mean.to(device),
            target_mean=self.target_mean.to(device),
        )


def compute_procrustes_mapping(
    source_view_emb: Tensor, target_view_emb: Tensor, approximate: bool = True, q: int = 1500
) -> ProcrustesMapping:
    """Compute an isometric mapping from source view embeddings to a target view space.

    Args:
        source_view_emb: Source view embeddings of shape (B, D_s)
        target_view_emb: Target view embeddings of shape (B, D_t)
        approximate: Whether to use low-rank approximation
        q: Number of singular values to keep for low-rank approximation

    Returns:
        Mapping parameters
    """
    source_mean = source_view_emb.mean(dim=0)  # (D_s,)
    target_mean = target_view_emb.mean(dim=0)  # (D_t,)
    source_centered = source_view_emb - source_mean
    target_centered = target_view_emb - target_mean

    covariance_matrix = torch.mm(source_centered.T, target_centered)  # (D_s, D_t)
    if approximate:
        # i = min(q, rank(covariance_matrix)) ~ min(q, D_s, D_t)
        U, S, V = torch.svd_lowrank(covariance_matrix, q=q)  # (D_s, i), (i,), (D_t, i)
        rotation_matrix = torch.mm(U, V.T)
    else:
        if covariance_matrix.size(0) != covariance_matrix.size(1):
            raise ValueError(
                "Exact Procrustes analysis requires square covariance matrix. "
                f"Got shape: {covariance_matrix.shape}"
            )
        U, S, Vt = torch.linalg.svd(covariance_matrix)  # (D_s, D_s), (min(D_s, D_t),), (D_t, D_t)
        rotation_matrix = torch.mm(U, Vt)

    k = torch.linalg.norm(S, ord=2) / torch.trace(torch.mm(source_centered, source_centered.T))
    return ProcrustesMapping(
        rotation_matrix=rotation_matrix,
        scaling_factor=1.0 if torch.isnan(k) else float(k.item()),
        source_mean=source_mean,
        target_mean=target_mean,
    )


def apply_procrustes_mapping(source_view_emb: Tensor, params: ProcrustesMapping) -> Tensor:
    """Apply a precomputed isometric mapping to source view embeddings.

    Args:
        source_view_emb: Source view embeddings
        params: Mapping parameters

    Returns:
        Transformed embeddings
    """
    if source_view_emb.dim() == 1:
        source_view_emb = source_view_emb.unsqueeze(0)

    # NOTE: We use the source mean from the training phase here since centering a single sample
    # using its own mean would always yield a zero vector.
    source_centered = source_view_emb - params.source_mean  # (B, D_s)
    source_emb_scaled = params.scaling_factor * torch.mm(
        source_centered, params.rotation_matrix
    )  # (B, D_t)
    source_emb_scaled += params.target_mean  # (B, D_t)

    return source_emb_scaled


class LocalIsometryModel(ProjectionModel):
    """Local Isometry projection model based on LA2M.

    This model cannot be trained with mini-batches due to the global clustering step before
    mapping computation.
    """

    def __init__(
        self,
        module_kwargs: LocalIsometryModuleConfig,
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
            "local_isometry": LocalDistanceCorrelation(k_neighbors=5),
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

        # Alignment parameters per view
        self.pca_models: dict[int, PCA] = {}
        # Cluster centers per view to select the correct mapping params during inference
        self.cluster_centers: dict[int, Tensor] = {}
        # Procrustes mapping params per view_idx and cluster_idx
        self.mapping_params: dict[int, dict[int, ProcrustesMapping]] = defaultdict(dict)

        # Disable lightning's automatic optimization for non-parametric model
        self.automatic_optimization = False

    @property
    def output_dim(self) -> int:
        # If PCA reduction is used, output dimension is the reduced dimension
        # Otherwise, it's the dimension of the reference view (view 0)
        if self.config.reduced_dim > 0:
            return self.config.reduced_dim
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
                "LocalIsometryModel does not support batching across samples. "
                "All samples must be processed in one batch."
            )
        if batch[reference_view_idx].numel() == 0:
            raise ValueError("Reference view embeddings are empty.")

        # Apply PCA reduction if needed
        if self.config.reduced_dim > 0:
            reduced_batch = []
            for view_idx, view_emb in enumerate(batch):
                n_samples, n_features = view_emb.shape  # (B, D_i)
                max_components = min(self.config.reduced_dim, n_features, max(1, n_samples))
                logger.info(
                    "PCA reduction for view {}: {} -> {}", view_idx, n_features, max_components
                )
                if max_components <= 0:
                    raise ValueError(f"Invalid number of PCA components: {max_components}")

                pca = PCA(n_components=max_components, svd_solver="randomized")
                reduced_emb = pca.fit_transform(view_emb.cpu().numpy())
                self.pca_models[view_idx] = pca
                reduced_batch.append(torch.from_numpy(reduced_emb).to(view_emb.device))
            batch = reduced_batch

        # The first view is the reference view so we don't need to compute a mapping for it
        reference_emb = batch[reference_view_idx]
        for view_idx, view_emb in enumerate(batch[1:], start=1):
            logger.info("Computing clustering for view {}", view_idx)
            clustering = self._compute_clustering(view_emb.cpu().numpy())

            logger.info("Computing cluster centers and mapping parameters for view {}", view_idx)
            self.cluster_centers[view_idx] = torch.empty(
                (len(clustering), view_emb.shape[1]), dtype=torch.float32, device=view_emb.device
            )
            for cluster_idx, cluster_indices in enumerate(clustering):
                # Compute cluster centers and store them for inference
                center = view_emb[cluster_indices].mean(dim=0)
                self.cluster_centers[view_idx][cluster_idx] = center

                # Compute and store Procrustes mapping for each cluster
                self.mapping_params[view_idx][cluster_idx] = compute_procrustes_mapping(
                    view_emb[cluster_indices],
                    reference_emb[cluster_indices],
                    approximate=self.config.approximate,
                    q=self.config.q,
                )
        logger.info("Completed computing mapping parameters for all views.")

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
        if len(batch) - 1 != len(self.cluster_centers):
            raise ValueError(
                f"This model was trained with {len(self.cluster_centers) + 1} views "
                f"but received {len(batch)} views."
            )

        reference_view_idx = 0

        # Apply PCA reduction if needed
        if self.config.reduced_dim > 0:
            batch = [
                torch.from_numpy(self.pca_models[view_idx].transform(view_emb.cpu().numpy())).to(
                    view_emb.device
                )
                for view_idx, view_emb in enumerate(batch)
            ]

        # Initialize output with reference view embeddings
        aligned_views = [batch[reference_view_idx]]
        aligned_dim = aligned_views[0].shape[1]
        for view_idx, view_emb in enumerate(batch[1:], start=1):
            aligned_emb = torch.empty((len(view_emb), aligned_dim), device=view_emb.device)
            closest_cluster_idx = torch.argmin(
                torch.cdist(view_emb, self.cluster_centers[view_idx]), dim=1
            )

            # Group sample indices by assigned cluster to process each cluster in a single batch
            for cluster in torch.unique(closest_cluster_idx):
                idxs = torch.nonzero(closest_cluster_idx == cluster, as_tuple=True)[0]

                # Gather embeddings for this cluster and move mapping params to the correct device
                params = self.mapping_params[view_idx][int(cluster.item())].to(view_emb.device)

                # Apply mapping in a single call for the whole cluster and assign results
                transformed = apply_procrustes_mapping(view_emb[idxs], params)  # (n_cluster, D_r)
                aligned_emb[idxs] = transformed.to(aligned_emb.device)

            aligned_views.append(aligned_emb)

        return torch.stack(aligned_views, dim=1)  # (B, V, D)

    def project(self, batch: Tensor, view_idx: int) -> Tensor:
        # Apply PCA reduction if needed
        if self.config.reduced_dim > 0:
            batch = torch.from_numpy(self.pca_models[view_idx].transform(batch.cpu().numpy())).to(
                batch.device
            )

        if view_idx == 0:
            # Reference view - no further transformation needed
            return batch
        if view_idx not in self.cluster_centers:
            raise ValueError(
                f"No mapping parameters found for view {view_idx}. "
                f"Available views: {list(self.cluster_centers.keys())}"
            )

        aligned_emb = torch.empty((len(batch), self.output_dim), device=batch.device)
        # Move cluster centers to the same device as the batch
        cluster_centers = self.cluster_centers[view_idx].to(batch.device)
        closest_cluster_idx = torch.argmin(torch.cdist(batch, cluster_centers), dim=1)
        # Group sample indices by assigned cluster to process each cluster in a single batch
        for cluster in torch.unique(closest_cluster_idx):
            idxs = torch.nonzero(closest_cluster_idx == cluster, as_tuple=True)[0]

            # Gather embeddings for this cluster and move mapping params to the correct device
            params = self.mapping_params[view_idx][int(cluster.item())].to(batch.device)

            # Apply mapping in a single call for the whole cluster and assign results
            transformed = apply_procrustes_mapping(batch[idxs], params)  # (n_cluster, D_r)
            aligned_emb[idxs] = transformed.to(aligned_emb.device)

        return aligned_emb

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """No optimizers needed for non-parametric model."""
        return None

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["config"] = self.config.model_dump()
        checkpoint["pca_models"] = self.pca_models
        checkpoint["cluster_centers"] = self.cluster_centers
        checkpoint["mapping_params"] = self.mapping_params

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.config = LocalIsometryModuleConfig.model_validate(checkpoint["config"])
        self.pca_models = checkpoint["pca_models"]
        self.cluster_centers = checkpoint["cluster_centers"]
        self.mapping_params = checkpoint["mapping_params"]

    def _compute_clustering(self, embeddings: NDArray[np.float32]) -> list[NDArray[np.int32]]:
        """Compute a clustering of the given embeddings.

        Returns:
            List of arrays of embedding indices for each cluster
        """
        if self.config.num_clusters > embeddings.shape[0]:
            logger.warning(
                "Number of clusters ({}) is greater than number of samples ({}). "
                "Setting number of clusters to {}.",
                self.config.num_clusters,
                embeddings.shape[0],
                embeddings.shape[0],
            )
        num_clusters = min(self.config.num_clusters, embeddings.shape[0])

        if self.config.clustering_method == "kmeans":
            clusterer = KMeans(n_clusters=num_clusters, n_init="auto")
        elif self.config.clustering_method == "avg_linkage":
            clusterer = AgglomerativeClustering(n_clusters=num_clusters, linkage="average")
        else:
            raise ValueError(f"Unknown clustering method: {self.config.clustering_method}")

        labels = clusterer.fit_predict(embeddings)
        indices = np.arange(embeddings.shape[0], dtype=np.int32)
        return [indices[np.where(labels == i)] for i in range(num_clusters)]

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
            metrics["local_isometry"].update(view_embeddings, aligned_view)
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
