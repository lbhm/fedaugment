from collections.abc import Sequence
from typing import Any, ClassVar, Literal

import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn
from torchmetrics import MetricCollection

from fedaugment.config import CLModuleConfig, CriterionConfig, LRSchedulerConfig, OptimizerConfig
from fedaugment.projections.metrics import CosineSimilarityOrchestrator

from .projection_model import ProjectionModel


class ContrastiveLearningModel(ProjectionModel):
    """Training procedure for a list of projection heads using contrastive learning.

    We do not implement the forward() method because the trained projection heads have to be split
    into seperate modules and loaded into a different object for inference.
    """

    def __init__(
        self,
        module_kwargs: CLModuleConfig,
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

        self.projection_heads = nn.ModuleList(
            [
                ProjectionHead(
                    in_dim=embedding_dim,
                    out_dim=module_kwargs.out_dim,
                    hidden_dims=module_kwargs.hidden_dims,
                    activation=module_kwargs.activation,
                    normalization=module_kwargs.normalization,
                    dropout=module_kwargs.dropout,
                )
                for embedding_dim in embedding_dims
            ]
        )
        self._output_dim = module_kwargs.out_dim

        if module_kwargs.no_val_orchestrator:
            self.val_metrics = MetricCollection({})
        if module_kwargs.no_test_orchestrator:
            self.test_metrics = MetricCollection({})

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def training_step(
        self, batch: Sequence[Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        # batch: A list of V tensors, each of shape (B, D_i), where B is the batch size and D_i is
        # the embedding dimension for view i.
        projections = torch.stack(
            [self.project(batch[v], v) for v in range(len(batch))], dim=1
        )  # (B, V, D), where D is the output dimension of the projection heads
        loss: Tensor = self.criterion(projections)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=batch[0].shape[0])
        return loss

    def validation_step(
        self, batch: Sequence[Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        self._shared_eval_step(batch, batch_idx, self.val_metrics, self.val_orchestrator, "val")

    def test_step(self, batch: Sequence[Tensor], batch_idx: int, dataloader_idx: int = 0) -> None:
        self._shared_eval_step(batch, batch_idx, self.test_metrics, self.test_orchestrator, "test")

    def project(self, batch: Tensor, view_idx: int) -> Tensor:
        out: Tensor = self.projection_heads[view_idx](batch)
        return out

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if len(self.optim_configs) != 1 or len(self.sched_configs) != 1:
            raise NotImplementedError(
                "ContrastiveLearningModel only supports a single optimizer and scheduler "
                "configuration."
            )

        decay: list[torch.nn.Parameter] = []
        no_decay: list[torch.nn.Parameter] = []
        for name, param in self.projection_heads.named_parameters():
            if not param.requires_grad:
                continue  # skip frozen parameters
            if name.endswith("bias") or "norm" in name.lower() or "bn" in name.lower():
                no_decay.append(param)
            else:
                decay.append(param)

        param_groups: list[dict[str, Any]] = []
        if decay:
            param_groups.append({"params": decay, **self.optim_configs[0].kwargs})
        if no_decay:
            # We don't apply weight decay to bias or normalization layers
            optim_kwargs_no_wd = self.optim_configs[0].kwargs.copy()
            optim_kwargs_no_wd.pop("weight_decay", None)
            param_groups.append({"params": no_decay, **optim_kwargs_no_wd})

        optimizer_class = getattr(torch.optim, self.optim_configs[0].class_)
        optimizer = optimizer_class(param_groups)

        scheduler_class = getattr(torch.optim.lr_scheduler, self.sched_configs[0].class_)
        scheduler_kwargs = self._resolve_scheduler_kwargs(self.sched_configs[0])
        scheduler = scheduler_class(optimizer=optimizer, **scheduler_kwargs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self._resolve_scheduler_interval(self.sched_configs[0]),
                "frequency": self.sched_configs[0].frequency,
                "monitor": self.sched_configs[0].monitor,
                "strict": True,
            },
        }

    def _shared_eval_step(
        self,
        batch: Sequence[Tensor],
        batch_idx: int,
        metrics: MetricCollection,
        orchestrator: CosineSimilarityOrchestrator,
        prefix: str,
    ) -> None:
        B, _ = batch[0].shape  # (B, D_i)
        V = len(batch)  # number of views

        projections = torch.stack(
            [self.project(batch[v], v) for v in range(V)], dim=1
        )  # (B, V, D)
        loss = self.criterion(projections)

        self.log(f"{prefix}_loss", loss, on_epoch=True, batch_size=B)
        if len(metrics) == 0:
            return  # No other metrics to compute

        # The last batch may be smaller than the batch size
        start_idx = self.data_batch_size * batch_idx
        end_idx = start_idx + min(self.data_batch_size, B)
        sample_labels = torch.arange(start_idx, end_idx, device=self.device).repeat_interleave(V)

        orchestrator.update(
            projections=projections.view(B * V, -1),  # (B * V, D)
            labels=sample_labels,  # (B * V,)
        )
        self.log_dict(metrics, on_epoch=True, batch_size=B)


class ProjectionHead(nn.Module):
    activation_map: ClassVar[dict[str, type[nn.Module]]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Sequence[int] | None = None,
        activation: Literal["relu", "gelu", "silu"] = "relu",
        normalization: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [out_dim * 2]
        act_fn = self.activation_map[activation]

        norm_layer: type[nn.Module]
        if normalization == "batch":
            norm_layer = nn.BatchNorm1d
        elif normalization == "layer":
            norm_layer = nn.LayerNorm
        else:
            norm_layer = nn.Identity
        use_batch_norm = normalization == "batch"

        layers: list[nn.Module] = []
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            # Linear layer, normalization, and activation function
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim, bias=not use_batch_norm),
                    norm_layer(hidden_dim),
                    act_fn(),
                ]
            )

            # Dropout (optional)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Final linear layer
        layers.append(nn.Linear(prev_dim, out_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.net(x)
        # NOTE: We do not apply L2-normalization here because our losses and metrics already handle
        # that internally.
        return out
