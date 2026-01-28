import abc
from typing import cast

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from pydantic import BaseModel
from torch import Tensor
from torchmetrics import Metric, MetricCollection

from fedaugment.config import CriterionConfig, LRSchedulerConfig, OptimizerConfig
from fedaugment.projections.losses import get_criterion
from fedaugment.projections.metrics import (
    CosineSimilarityMean,
    CosineSimilarityOrchestrator,
    EntityStability,
    OrchestratorChild,
    SameLabelCosineSimilarityMean,
)


class ProjectionModel(L.LightningModule, abc.ABC):
    """Base class for all projection models."""

    pipeline_names: list[str]
    """Names of the embedding pipelines this model can project."""
    embedding_dims: list[int]
    """Dimensions of the embeddings for each view (i.e., embedding pipeline)."""

    def __init__(
        self,
        module_kwargs: BaseModel,
        criterion: CriterionConfig,
        optim_configs: list[OptimizerConfig],
        sched_configs: list[LRSchedulerConfig],
        pipeline_names: list[str],
        embedding_dims: list[int],
        data_batch_size: int,
        metric_batch_size: int | None = None,
        metrics: MetricCollection | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["metrics"])

        self.criterion: torch.nn.Module = get_criterion(criterion)
        self.optim_configs = optim_configs
        self.sched_configs = sched_configs

        self.pipeline_names = pipeline_names
        self.embedding_dims = embedding_dims

        self.num_views = len(self.embedding_dims)
        self.data_batch_size = data_batch_size
        self.metric_batch_size = metric_batch_size

        if metrics is None:
            # Create shared metrics orchestrator and child metrics
            orchestrator = CosineSimilarityOrchestrator(chunk_size=self.metric_batch_size)
            metrics_dict: dict[str, Metric | MetricCollection] = {
                "cossim_mean": OrchestratorChild(orchestrator, CosineSimilarityMean()),
                "same_lbl_cossim_mean": OrchestratorChild(
                    orchestrator, SameLabelCosineSimilarityMean()
                ),
                "entity_stability_5": OrchestratorChild(orchestrator, EntityStability(k=5)),
                "entity_stability_15": OrchestratorChild(orchestrator, EntityStability(k=15)),
                "entity_stability_50": OrchestratorChild(orchestrator, EntityStability(k=50)),
            }
            metrics = MetricCollection(metrics_dict)

        # Clone for validation and test
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        # NOTE: We assume that 'cossim_mean' metric is always present
        self.val_orchestrator = cast(
            "OrchestratorChild", self.val_metrics["cossim_mean"]
        ).orchestrator
        self.test_orchestrator = cast(
            "OrchestratorChild", self.test_metrics["cossim_mean"]
        ).orchestrator

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if len(self.optim_configs) != 1 or len(self.sched_configs) != 1:
            raise NotImplementedError(
                "The default implementation only supports a single optimizer and scheduler "
                "configuration."
            )

        optimizer_class = getattr(torch.optim, self.optim_configs[0].class_)
        optimizer = optimizer_class(params=self.parameters(), **self.optim_configs[0].kwargs)
        scheduler_class = getattr(torch.optim.lr_scheduler, self.sched_configs[0].class_)
        scheduler = scheduler_class(optimizer=optimizer, **self.sched_configs[0].kwargs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.sched_configs[0].interval,
                "frequency": self.sched_configs[0].frequency,
                "monitor": self.sched_configs[0].monitor,
                "strict": True,
            },
        }

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        """Returns the output dimension of the aligned embedding space."""

    @abc.abstractmethod
    def project(self, batch: Tensor, view_idx: int) -> Tensor:
        """Project a single batch of embeddings to a common space.

        Args:
            batch: A tensor of shape (B, D_i), where B is the batch size and
                D_i is the embedding dimension for the input view.
            view_idx: Index of the embedding space/view that the batch comes from.

        Returns:
            A tensor of shape (B, D_out), where D_out is the output dimension of the
            projection (common embedding space).
        """
