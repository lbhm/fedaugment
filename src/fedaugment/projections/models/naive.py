from collections.abc import Sequence
from typing import Any

import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor
from torchmetrics import MetricCollection

from fedaugment.config import (
    CriterionConfig,
    LRSchedulerConfig,
    NaiveModuleConfig,
    OptimizerConfig,
)
from fedaugment.projections.metrics import CosineSimilarityOrchestrator

from .projection_model import ProjectionModel


class NaiveModel(ProjectionModel):
    """Naive projection model that truncatees or pads embeddings to the same dimension."""

    def __init__(
        self,
        module_kwargs: NaiveModuleConfig,
        criterion: CriterionConfig,
        optim_configs: list[OptimizerConfig],
        sched_configs: list[LRSchedulerConfig],
        pipeline_names: list[str],
        embedding_dims: list[int],
        data_batch_size: int,
        metric_batch_size: int | None = None,
    ) -> None:
        super().__init__(
            module_kwargs=module_kwargs,
            criterion=criterion,
            optim_configs=optim_configs,
            sched_configs=sched_configs,
            pipeline_names=pipeline_names,
            embedding_dims=embedding_dims,
            data_batch_size=data_batch_size,
            metric_batch_size=metric_batch_size,
        )

        self.mode = module_kwargs.mode
        self._output_dim = (
            max(self.embedding_dims) if self.mode == "pad" else min(self.embedding_dims)
        )

        # Disable lightning's automatic optimization for non-parametric model
        self.automatic_optimization = False

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def training_step(
        self, batch: Sequence[Tensor], batch_idx: int, _dataloader_idx: int = 0
    ) -> None:
        """The NaiveModel has no trainable parameters."""

    def validation_step(
        self, batch: Sequence[Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._shared_eval_step(batch, batch_idx, self.val_metrics, self.val_orchestrator)

    def test_step(self, batch: Sequence[Tensor], batch_idx: int, dataloader_idx: int = 0) -> None:
        self._shared_eval_step(batch, batch_idx, self.test_metrics, self.test_orchestrator)

    def project(self, batch: Tensor, view_idx: int) -> Tensor:
        """Truncate or pad a single batch of embeddings to the output dimension.

        Args:
            batch: A tensor of shape (B, D_i), where B is the batch size and
                D_i is the embedding dimension for the input view.
            view_idx: Index of the embedding space/view that the batch comes from.
                View 0 is the reference view and is returned unchanged.

        Returns:
            A tensor of shape (B, D_out), where D_out is the output dimension of the
            projection (min dimension for truncate and max dimension for pad).
        """
        B, in_dim = batch.shape

        if self.mode == "truncate":
            return batch[:, : self.output_dim]
        if self.mode == "pad":
            pad_size = self.output_dim - in_dim
            padding = torch.zeros((B, pad_size), device=batch.device, dtype=batch.dtype)
            return torch.cat([batch, padding], dim=1)
        raise ValueError(f"Unknown mode: {self.mode}")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """No optimizers needed for non-parametric model."""
        return None

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["mode"] = self.mode

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.mode = checkpoint["mode"]

    def _shared_eval_step(
        self,
        batch: Sequence[Tensor],
        batch_idx: int,
        metrics: MetricCollection,
        orchestrator: CosineSimilarityOrchestrator,
    ) -> None:
        B, _ = batch[0].shape  # (B, D_i)
        V = len(batch)  # number of views

        projections = torch.stack(
            [self.project(batch[v], v) for v in range(V)], dim=1
        )  # (B, V, D)

        # The last batch may be smaller than the batch size
        start_idx = self.data_batch_size * batch_idx
        end_idx = start_idx + min(self.data_batch_size, B)
        sample_labels = torch.arange(start_idx, end_idx, device=self.device).repeat_interleave(V)

        orchestrator.update(
            projections=projections.view(B * V, -1),  # (B * V, D)
            labels=sample_labels,  # (B * V,)
        )
        self.log_dict(metrics, on_epoch=True, batch_size=B)
