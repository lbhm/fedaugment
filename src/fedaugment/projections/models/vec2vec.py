from collections.abc import Iterable, Sequence
from typing import Any, Literal, cast

import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import MetricCollection

from fedaugment.config import (
    CriterionConfig,
    LRSchedulerConfig,
    OptimizerConfig,
    Vec2VecModuleConfig,
)
from fedaugment.projections.metrics import CosineSimilarityOrchestrator

from .projection_model import ProjectionModel


class Vec2VecModel(ProjectionModel):
    """Training procedure for unsupervised GAN-based embedding alignment inspired by Vec2Vec.

    Implements the training loop with manual optimization for GAN training.

    See http://arxiv.org/abs/2505.12540 for more details.
    """

    def __init__(
        self,
        module_kwargs: Vec2VecModuleConfig,
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
        self.module_kwargs = module_kwargs

        # model to be trained
        self.trans = Translators(module_kwargs, embedding_dims)
        # discriminators for gan loss
        self._init_discs(embedding_dims)
        # necessary for gan setups...
        self.automatic_optimization = False

        # GAN training optimizations
        self.disc_update_counter = 0
        self.gen_ema: dict[str, Tensor] | None = None
        if module_kwargs.use_ema:
            self.gen_ema = {}
            for name, param in self.trans.named_parameters():
                self.gen_ema[name] = param.data.clone()

    @property
    def output_dim(self) -> int:
        return self.embedding_dims[0]

    def training_step(
        self, batch: Sequence[Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Manual optimization training step implementing GAN loop with Lightning.

        Outline:
         1) Forward pass through translators to obtain recons/translations/latents
         2) Update discriminators (manual steps via _disc_step)
         3) Compute GAN generator loss using _gan_loss and additional projection losses
         4) Backpropagate and step the Lightning-managed generator optimizer

        Manual optimization (self.automatic_optimization = False) is required so we can
        interleave discriminator updates (stepped via their optimizers) and generator updates.
        """
        # forward
        inputs, recons, trans_to, trans_from, reps = self(batch)

        # Update discriminators (potentially multiple times per generator update)
        disc_metrics = {}
        for _ in range(self.module_kwargs.disc_updates_per_gen):
            disc_metrics = self._disc_step(inputs, trans_from, trans_to, reps)
            self.disc_update_counter += 1

        # gan loss afterwards
        # Retrieve generator optimizer and scheduler
        trans_opt = cast("list[Optimizer]", self.optimizers())[0]
        trans_sched = cast("list[LRScheduler]", self.lr_schedulers())[0]
        self.toggle_optimizer(trans_opt)
        gan_loss, gen_metrics = self._gan_loss(inputs, trans_from, trans_to, reps)
        # invert translations
        trans_from_inverted = self.trans([trans.detach() for trans in trans_from], mode="from")
        trans_to_inverted = self.trans([trans.detach() for trans in trans_to], mode="to")
        # compute other loss constraints
        loss_dict: dict[str, Tensor] = self.criterion(
            inputs, recons, trans_from_inverted, trans_to_inverted
        )
        loss_dict["gan_loss"] = gan_loss
        loss = torch.stack(list(loss_dict.values())).sum()

        # Backpropagate generator loss and step Lightning-managed optimizer
        self.manual_backward(loss)

        # Compute generator gradient norm for stability monitoring
        gen_grad_norm = torch.nn.utils.clip_grad_norm_(self.trans.parameters(), max_norm=1000.0)

        trans_opt.step()
        trans_opt.zero_grad(set_to_none=True)
        trans_sched.step()
        self.untoggle_optimizer(trans_opt)

        # Update exponential moving average of generator weights
        if self.module_kwargs.use_ema and self.gen_ema is not None:
            self._update_ema()

        # Combine all metrics for logging
        all_metrics = {**loss_dict, **disc_metrics, **gen_metrics}
        all_metrics["gen_grad_norm"] = gen_grad_norm

        # Compute stability indicators
        avg_disc_loss = (
            disc_metrics["disc_loss_from"]
            + disc_metrics["disc_loss_to"]
            + disc_metrics["disc_loss_latent"]
        ) / 3.0
        all_metrics["gan_loss_ratio"] = gan_loss / (avg_disc_loss + 1e-8)
        all_metrics["disc_grad_norm_ratio"] = disc_metrics["disc_grad_norm"] / (
            gen_grad_norm + 1e-8
        )

        self.log_dict(all_metrics, on_step=False, on_epoch=True, batch_size=batch[0].shape[0])

    def validation_step(
        self, batch: Sequence[Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if self.module_kwargs.use_ema:
            self._apply_ema_weights()
        try:
            self._shared_eval_step(
                batch, batch_idx, self.val_metrics, self.val_orchestrator, "val"
            )
        finally:
            if self.module_kwargs.use_ema:
                self._restore_original_weights()

    def test_step(self, batch: Sequence[Tensor], batch_idx: int, dataloader_idx: int = 0) -> None:
        if self.module_kwargs.use_ema:
            self._apply_ema_weights()
        try:
            self._shared_eval_step(
                batch, batch_idx, self.test_metrics, self.test_orchestrator, "test"
            )
        finally:
            if self.module_kwargs.use_ema:
                self._restore_original_weights()

    def forward(
        self, batch: Sequence[Tensor]
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
        """Run translator forward and return normalized inputs and predicted tensors.

        Returns a (inputs_normalized, recons, trans_to, trans_from, reps) tuple, where:
          - recons: reconstructions (i->i)
          - trans_to: translations from others to view 1
          - trans_from: translations from view 1 to others
          - reps: latent representations for each view
        """
        inputs = [F.normalize(b, dim=1) for b in batch]
        # get all the predictions
        recons, trans_to, trans_from, reps = self.trans(inputs, mode="all")
        return inputs, recons, trans_to, trans_from, reps

    def project(self, batch: Tensor, view_idx: int) -> Tensor:
        if self.module_kwargs.use_ema and self.training:
            self._apply_ema_weights()
        try:
            out: Tensor = self.trans([batch], mode="eval", idx=view_idx)[0]
            return out
        finally:
            if self.module_kwargs.use_ema and self.training:
                self._restore_original_weights()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Register generator and discriminator optimizers and schedulers with Lightning.

        Convention used here (important for optimizer ordering when retrieving via
        self.optimizers()):
            - index 0: generator optimizer
            - indices 1..: discriminator optimizers in same order as disc_latent, disc_trans_from,
                disc_trans_to

        This ordering is relied upon by _get_disc.
        """
        if len(self.optim_configs) not in {1, 2} or len(self.sched_configs) not in {1, 2}:
            raise NotImplementedError(
                "vec2vec requires one or two optimizer and scheduler configs."
            )

        optimizers: list[Optimizer] = []
        schedulers: list[LRScheduler] = []

        gen_optim_class: type[Optimizer] = getattr(torch.optim, self.optim_configs[0].class_)
        gen_sched_class: type[LRScheduler] = getattr(
            torch.optim.lr_scheduler, self.sched_configs[0].class_
        )

        # If only one optimizer is provided, use it for both generator and discriminators
        disc_optim_class: type[Optimizer]
        disc_optim_kwargs: dict[str, Any]
        if len(self.optim_configs) == 1:
            disc_optim_class = gen_optim_class
            disc_optim_kwargs = self.optim_configs[0].kwargs
        else:
            disc_optim_class = getattr(torch.optim, self.optim_configs[1].class_)
            disc_optim_kwargs = self.optim_configs[1].kwargs

        # If only one scheduler is provided, use it for both generator and discriminators
        disc_sched_class: type[LRScheduler]
        disc_sched_kwargs: dict[str, Any]
        if len(self.sched_configs) == 1:
            disc_sched_class = gen_sched_class
            disc_sched_kwargs = self.sched_configs[0].kwargs
        else:
            disc_sched_class = getattr(torch.optim.lr_scheduler, self.sched_configs[1].class_)
            disc_sched_kwargs = self.sched_configs[1].kwargs

        # optimizer and scheduler for generator
        gen_optim = gen_optim_class(params=self.trans.parameters(), **self.optim_configs[0].kwargs)
        gen_sched = gen_sched_class(optimizer=gen_optim, **self.sched_configs[0].kwargs)
        optimizers.append(gen_optim)
        schedulers.append(gen_sched)

        # optimizer and schedulers for discriminators
        for disc in self.disc_latent + self.disc_trans_from + self.disc_trans_to:
            disc_optim = disc_optim_class(params=disc.parameters(), **disc_optim_kwargs)
            disc_sched = disc_sched_class(optimizer=disc_optim, **disc_sched_kwargs)
            optimizers.append(disc_optim)
            schedulers.append(disc_sched)

        return optimizers, schedulers

    def _init_discs(self, embedding_dims: Sequence[int]) -> None:
        """Create discriminator modules.

        We create three groups of discriminators:
        - disc_latent: operates in the shared latent space (one per view)
        - disc_trans_from: discriminators that judge translations coming FROM view 1 to other views
        - disc_trans_to: discriminators that judge translations produced from other views TO view 1

        These discriminators are simple feed-forward networks that output a scalar logit.
        """
        self.disc_latent = nn.ModuleList(
            [
                Discriminator(
                    latent_dim=self.module_kwargs.latent_dim,
                    discriminator_dim=self.module_kwargs.disc_dim,
                    depth=self.module_kwargs.disc_depth,
                    weight_init=self.module_kwargs.weight_init,
                    use_spectral_norm=self.module_kwargs.use_spectral_norm,
                    dropout_p=self.module_kwargs.disc_dropout,
                )
                for _ in range(len(embedding_dims))
            ]
        )

        # discriminators for F_(1->i), i != 1
        self.disc_trans_from = nn.ModuleList(
            [
                Discriminator(
                    latent_dim=dim,
                    discriminator_dim=self.module_kwargs.disc_dim,
                    depth=self.module_kwargs.disc_depth,
                    weight_init=self.module_kwargs.weight_init,
                    use_spectral_norm=self.module_kwargs.use_spectral_norm,
                    dropout_p=self.module_kwargs.disc_dropout,
                )
                for dim in embedding_dims[1:]
            ]
        )

        # discriminators for F_(i->1), i != 1
        self.disc_trans_to = nn.ModuleList(
            [
                Discriminator(
                    latent_dim=embedding_dims[0],
                    discriminator_dim=self.module_kwargs.disc_dim,
                    depth=self.module_kwargs.disc_depth,
                    weight_init=self.module_kwargs.weight_init,
                    use_spectral_norm=self.module_kwargs.use_spectral_norm,
                    dropout_p=self.module_kwargs.disc_dropout,
                )
                for _ in range(len(embedding_dims) - 1)
            ]
        )

    def _shared_eval_step(
        self,
        batch: Sequence[Tensor],
        batch_idx: int,
        metrics: MetricCollection,
        orchestrator: CosineSimilarityOrchestrator,
        prefix: str,
    ) -> None:
        """Evaluate retrieval-like metrics for the current batch.

        - Normalizes each view's embeddings and projects them into the shared latent/target spaces
        - Constructs (B, V, D) tensor of projections and flattens it for metric update
        - Uses deterministic sample indices so epoch-level metric aggregation is stable
        """
        B, _ = batch[0].shape  # (B, D_i)
        V = len(batch)  # number of views

        projections_list: list[Tensor] = []
        for model_idx in range(V):
            x = batch[model_idx]
            x = F.normalize(x, dim=1)
            proj = self.project(x, view_idx=model_idx)
            projections_list.append(proj)
        projections = torch.stack(projections_list, dim=1)  # (B, V, D)

        # The last batch may be smaller than the batch size
        start_idx = self.data_batch_size * batch_idx
        end_idx = start_idx + min(self.data_batch_size, B)
        sample_labels = torch.arange(start_idx, end_idx, device=self.device).repeat_interleave(V)

        orchestrator.update(
            projections=projections.view(B * V, -1),  # (B * V, D)
            labels=sample_labels,  # (B * V,)
        )
        self.log_dict(metrics, on_epoch=True, batch_size=B)

    def _disc_step(
        self,
        inputs: list[Tensor],
        trans_from: list[Tensor],
        trans_to: list[Tensor],
        reps: list[Tensor],
    ) -> dict[str, Tensor]:
        """Run discriminator updates for all discriminator types.

        All inputs are detached to avoid gradient flow back into the generator/translators during
        discriminator updates.

        Returns:
            Dictionary with discriminator stability metrics including losses, accuracies, logit
            statistics, and gradient norms.
        """
        disc_losses_from: list[Tensor] = []
        disc_accs_from: list[float] = []
        real_logits_from: list[Tensor] = []
        fake_logits_from: list[Tensor] = []
        grad_norms: list[Tensor] = []

        for i in range(len(trans_from)):
            disc, optim, sched = self._get_disc("from", i)
            loss, acc, real_logit, fake_logit, grad_norm = self._disc_single_step(
                disc, optim, sched, inputs[i + 1], trans_from[i]
            )
            disc_losses_from.append(loss)
            disc_accs_from.append(acc)
            real_logits_from.append(real_logit)
            fake_logits_from.append(fake_logit)
            grad_norms.append(grad_norm)

        disc_losses_to: list[Tensor] = []
        disc_accs_to: list[float] = []
        real_logits_to: list[Tensor] = []
        fake_logits_to: list[Tensor] = []

        for i in range(len(trans_to)):
            disc, optim, sched = self._get_disc("to", i)
            loss, acc, real_logit, fake_logit, grad_norm = self._disc_single_step(
                disc, optim, sched, inputs[0], trans_to[i]
            )
            disc_losses_to.append(loss)
            disc_accs_to.append(acc)
            real_logits_to.append(real_logit)
            fake_logits_to.append(fake_logit)
            grad_norms.append(grad_norm)

        disc_losses_latent: list[Tensor] = []
        disc_accs_latent: list[float] = []
        real_logits_latent: list[Tensor] = []
        fake_logits_latent: list[Tensor] = []

        for i in range(len(reps)):
            disc, optim, sched = self._get_disc("latent", i)
            for j in range(len(reps)):
                if i != j:
                    loss, acc, real_logit, fake_logit, grad_norm = self._disc_single_step(
                        disc, optim, sched, reps[i], reps[j]
                    )
                    disc_losses_latent.append(loss)
                    disc_accs_latent.append(acc)
                    real_logits_latent.append(real_logit)
                    fake_logits_latent.append(fake_logit)
                    grad_norms.append(grad_norm)

        # Aggregate metrics
        metrics: dict[str, Tensor] = {}
        if disc_losses_from:
            metrics["disc_loss_from"] = torch.stack(disc_losses_from).mean()
            metrics["disc_acc_from"] = torch.tensor(disc_accs_from).mean()
            metrics["disc_real_logit_from"] = torch.stack(real_logits_from).mean()
            metrics["disc_fake_logit_from"] = torch.stack(fake_logits_from).mean()
        if disc_losses_to:
            metrics["disc_loss_to"] = torch.stack(disc_losses_to).mean()
            metrics["disc_acc_to"] = torch.tensor(disc_accs_to).mean()
            metrics["disc_real_logit_to"] = torch.stack(real_logits_to).mean()
            metrics["disc_fake_logit_to"] = torch.stack(fake_logits_to).mean()
        if disc_losses_latent:
            metrics["disc_loss_latent"] = torch.stack(disc_losses_latent).mean()
            metrics["disc_acc_latent"] = torch.tensor(disc_accs_latent).mean()
            metrics["disc_real_logit_latent"] = torch.stack(real_logits_latent).mean()
            metrics["disc_fake_logit_latent"] = torch.stack(fake_logits_latent).mean()
        if grad_norms:
            metrics["disc_grad_norm"] = torch.stack(grad_norms).mean()

        return metrics

    def _disc_single_step(
        self,
        discriminator: nn.Module,
        optim: Optimizer,
        sched: LRScheduler,
        real: Tensor,
        fake: Tensor,
    ) -> tuple[Tensor, float, Tensor, Tensor, Tensor]:
        """Perform a single discriminator update using LS-GAN loss.

        Steps:
        1. Toggle the optimizer so Lightning's bookkeeping is correct for manual optimization.
        2. Compute logits for real and fake, build least-squares loss, and backpropagate.
        3. Step optimizer and scheduler and untoggle optimizer.

        Returns:
            Tuple of (disc_loss, disc_accuracy, mean_real_logit, mean_fake_logit, grad_norm).
            - disc_loss: Discriminator loss value
            - disc_accuracy: Fraction of samples correctly classified (real as real, fake as fake)
            - mean_real_logit: Average discriminator output for real samples
            - mean_fake_logit: Average discriminator output for fake samples
            - grad_norm: L2 norm of discriminator gradients
        """
        self.toggle_optimizer(optim)

        real = real.detach().requires_grad_(True)
        fake = fake.detach().requires_grad_(True)

        d_real_logits: Tensor = discriminator(real)
        d_fake_logits: Tensor = discriminator(fake)

        # Apply label smoothing
        real_target = 1.0 - self.module_kwargs.label_smoothing
        fake_target = self.module_kwargs.label_smoothing

        disc_loss_real = ((d_real_logits - real_target) ** 2).mean()
        disc_loss_fake = ((d_fake_logits - fake_target) ** 2).mean()
        disc_loss = 0.5 * (disc_loss_real + disc_loss_fake)
        # pytorch lightning way of doing backward pass with manual optimization
        self.manual_backward(disc_loss)

        # Compute gradient norm BEFORE clipping and optimizer step
        grad_norm_sq = torch.tensor(0.0, device=disc_loss.device)
        for p in discriminator.parameters():
            if p.grad is not None:
                grad_norm_sq += (p.grad.detach() ** 2).sum()
        grad_norm = grad_norm_sq.sqrt()

        self.clip_gradients(optim, gradient_clip_val=1000.0)
        optim.step()
        optim.zero_grad(set_to_none=True)

        sched.step()

        self.untoggle_optimizer(optim)

        # Compute discriminator accuracy: real samples should have logits close to their
        # respective (possibly smoothed) targets.
        decision_boundary = 0.5
        real_correct = ((d_real_logits - real_target).abs() < decision_boundary).float().mean()
        fake_correct = ((d_fake_logits - fake_target).abs() < decision_boundary).float().mean()
        disc_accuracy = 0.5 * (real_correct + fake_correct).item()

        return (
            disc_loss.detach(),
            disc_accuracy,
            d_real_logits.mean().detach(),
            d_fake_logits.mean().detach(),
            grad_norm,
        )

    def _gan_loss(
        self,
        inputs: list[Tensor],
        trans_from: list[Tensor],
        trans_to: list[Tensor],
        reps: list[Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Aggregate generator losses computed against each discriminator.

        The returned scalar should be added to other projection/consistency losses and used to
        update the generator (translator) parameters.

        Returns:
            Tuple of (total_gan_loss, gen_metrics) where gen_metrics contains per-group
            generator losses and accuracies for stability monitoring.
        """
        gen_losses_from: list[Tensor] = []
        gen_accs_from: list[float] = []
        gen_fake_logits_from: list[Tensor] = []

        for i in range(len(trans_from)):
            disc, _, _ = self._get_disc("from", i)
            loss, acc, fake_logit = self._gan_single_loss(disc, inputs[i + 1], trans_from[i])
            gen_losses_from.append(loss)
            gen_accs_from.append(acc)
            gen_fake_logits_from.append(fake_logit)

        gen_losses_to: list[Tensor] = []
        gen_accs_to: list[float] = []
        gen_fake_logits_to: list[Tensor] = []

        for i in range(len(trans_to)):
            disc, _, _ = self._get_disc("to", i)
            loss, acc, fake_logit = self._gan_single_loss(disc, inputs[0], trans_to[i])
            gen_losses_to.append(loss)
            gen_accs_to.append(acc)
            gen_fake_logits_to.append(fake_logit)

        gen_losses_latent: list[Tensor] = []
        gen_accs_latent: list[float] = []
        gen_fake_logits_latent: list[Tensor] = []

        for i in range(len(reps)):
            disc, _, _ = self._get_disc("latent", i)
            for j in range(len(reps)):
                if i != j:
                    loss, acc, fake_logit = self._gan_single_loss(disc, reps[i], reps[j])
                    gen_losses_latent.append(loss)
                    gen_accs_latent.append(acc)
                    gen_fake_logits_latent.append(fake_logit)

        # Aggregate all generator losses
        all_gen_losses = gen_losses_from + gen_losses_to + gen_losses_latent
        total_loss = torch.stack(all_gen_losses).sum()

        # Build metrics dictionary
        gen_metrics: dict[str, Tensor] = {}
        if gen_losses_from:
            gen_metrics["gen_loss_from"] = torch.stack(gen_losses_from).mean()
            gen_metrics["gen_acc_from"] = torch.tensor(gen_accs_from).mean()
            gen_metrics["gen_fake_logit_from"] = torch.stack(gen_fake_logits_from).mean()
        if gen_losses_to:
            gen_metrics["gen_loss_to"] = torch.stack(gen_losses_to).mean()
            gen_metrics["gen_acc_to"] = torch.tensor(gen_accs_to).mean()
            gen_metrics["gen_fake_logit_to"] = torch.stack(gen_fake_logits_to).mean()
        if gen_losses_latent:
            gen_metrics["gen_loss_latent"] = torch.stack(gen_losses_latent).mean()
            gen_metrics["gen_acc_latent"] = torch.tensor(gen_accs_latent).mean()
            gen_metrics["gen_fake_logit_latent"] = torch.stack(gen_fake_logits_latent).mean()

        return total_loss, gen_metrics

    def _update_ema(self) -> None:
        """Update exponential moving average of generator weights.

        EMA helps stabilize training and can produce better final models.
        """
        if self.gen_ema is None:
            return

        with torch.no_grad():
            for name, param in self.trans.named_parameters():
                if name in self.gen_ema:
                    self.gen_ema[name].mul_(self.module_kwargs.ema_decay).add_(
                        param.data, alpha=1 - self.module_kwargs.ema_decay
                    )

    def _apply_ema_weights(self) -> None:
        """Temporarily apply EMA weights to the generator for evaluation."""
        if self.gen_ema is None:
            return

        # Store current weights
        self._stored_params = {}
        with torch.no_grad():
            for name, param in self.trans.named_parameters():
                if name in self.gen_ema:
                    self._stored_params[name] = param.data.clone()
                    param.data.copy_(self.gen_ema[name])

    def _restore_original_weights(self) -> None:
        """Restore original weights after evaluation with EMA."""
        if not hasattr(self, "_stored_params"):
            return

        with torch.no_grad():
            for name, param in self.trans.named_parameters():
                if name in self._stored_params:
                    param.data.copy_(self._stored_params[name])
        del self._stored_params

    def on_train_end(self) -> None:
        """Copy EMA weights to model parameters at the end of training.

        This ensures the saved checkpoint contains the EMA weights, which typically
        produce better results than the final training weights.
        """
        if self.gen_ema is not None:
            with torch.no_grad():
                for name, param in self.trans.named_parameters():
                    if name in self.gen_ema:
                        param.data.copy_(self.gen_ema[name])
            # Clear EMA storage as it's now in the main parameters
            self.gen_ema = None

    def _get_disc(
        self, kind: Literal["latent", "from", "to"], idx: int
    ) -> tuple[nn.Module, Optimizer, LRScheduler]:
        """Return a discriminator, its optimizer, and scheduler.

        Note: We fetch optimizers/schedulers from Lightning's configure_optimizers() ordering.
        The +1 offsets skip the generator's optimizer, which is placed first in the list.
        """
        # NOTE: We pretend that self.optimizers() returns list[Optimizer] and instead of
        # LightningOptimizer to not confuse type checkers
        optims = cast("list[Optimizer]", self.optimizers())
        scheds = cast("list[LRScheduler]", self.lr_schedulers())
        if kind == "latent":
            tool_idx = idx + 1
            return (self.disc_latent[idx], optims[tool_idx], scheds[tool_idx])
        if kind == "from":
            tool_idx = idx + 1 + len(self.disc_latent)
            return (self.disc_trans_from[idx], optims[tool_idx], scheds[tool_idx])
        if kind == "to":
            tool_idx = idx + 1 + len(self.disc_latent) + len(self.disc_trans_from)
            return (self.disc_trans_to[idx], optims[tool_idx], scheds[tool_idx])
        raise ValueError(f"Unknown discriminator kind: {kind}")

    def _gan_single_loss(
        self, discriminator: nn.Module, real: Tensor, fake: Tensor
    ) -> tuple[Tensor, float, Tensor]:
        """Compute generator-side LS-GAN loss for a discriminator.

        This does NOT step any optimizer; it only evaluates discriminator(fake) and
        returns the scalar loss that the generator should minimize plus diagnostics.

        In LS-GAN, the generator tries to fool the discriminator by making fake samples
        have logits close to 1 (i.e., making them indistinguishable from real samples).

        Returns:
            Tuple of (gen_loss, gen_accuracy, mean_fake_logit) where:
            - gen_loss: Generator loss for fooling this discriminator
            - gen_accuracy: Fraction of fake samples successfully fooling discriminator
            - mean_fake_logit: Average discriminator output for generated samples
        """
        threshold = 0.5
        d_fake_logits: Tensor = discriminator(fake)
        # Generator wants discriminator to output 1 for fake samples (fooling the discriminator)
        gen_loss = ((d_fake_logits - 1.0) ** 2).mean() * 0.5

        # Generator success: fake samples with logits close to 1
        gen_acc = ((d_fake_logits - 1.0).abs() < threshold).float().mean().item()
        mean_fake_logit = d_fake_logits.mean().detach()

        return gen_loss, gen_acc, mean_fake_logit


class Translators(nn.Module):
    """Collection of encoder/decoder adapters and a shared transform.

    - Each view has an encoder and decoder (adapter MLPs) to map into/from a common latent space.
    - The shared `transform` maps encoder outputs into a fixed latent dimensionality.
    - forward() supports different modes: 'all' (full set), 'to', 'from' and 'eval'.
    """

    def __init__(
        self,
        module_kwargs: Vec2VecModuleConfig,
        embedding_dims: Sequence[int],
        normalize_embs: bool = True,
    ) -> None:
        """Translators container for encoders/decoders.

        Initializes the Translators class with a list of embedding models and parameters for the
        encoders and decoders.

        Args:
            module_kwargs: Configuration parameters for the translators.
            embedding_dims: The embedding dimension of each view.
            normalize_embs: Whether to normalize the output embeddings.
        """
        super().__init__()
        self.embedding_dims = embedding_dims
        self.module_kwargs = module_kwargs
        self.normalize_embs = normalize_embs
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.latent_transform = MLPWithResidual(
            depth=self.module_kwargs.latent_transform_depth,
            in_dim=self.module_kwargs.latent_dim,
            hidden_dim=self.module_kwargs.latent_transform_dim,
            out_dim=self.module_kwargs.latent_dim,
            norm_style=self.module_kwargs.norm_style,
            weight_init=self.module_kwargs.weight_init,
        )
        # encoders and decoders
        for dim in embedding_dims:
            enc, dec = self.init_adapters(dim)
            self.encoder.append(enc)
            self.decoder.append(dec)
        self.n = len(embedding_dims)

    def init_adapters(self, emb_dim: int) -> tuple[nn.Module, nn.Module]:
        return (
            MLPWithResidual(
                self.module_kwargs.translator_depth,
                emb_dim,
                self.module_kwargs.translator_dim,
                self.latent_transform.in_dim,
                norm_style=self.module_kwargs.norm_style,
                output_norm=False,
                weight_init=self.module_kwargs.weight_init,
            ),
            MLPWithResidual(
                self.module_kwargs.translator_depth,
                self.latent_transform.out_dim,
                self.module_kwargs.translator_dim,
                emb_dim,
                norm_style=self.module_kwargs.norm_style,
                output_norm=False,
                weight_init=self.module_kwargs.weight_init,
            ),
        )

    def forward(  # noqa: C901
        self,
        inputs: list[Tensor],
        mode: Literal["all", "to", "from", "eval"],
        idx: int | None = None,
    ) -> tuple[list[Tensor], ...] | list[Tensor]:
        """Compute embeddings using one of the supported modes.

        Modes:
          - 'all': returns (recons, trans_to, trans_from, reps)
          - 'to': translate inputs to view-0 space
          - 'from': translate inputs from view-0 into others
          - 'eval': produce projections for evaluation (requires idx)
        """
        # TODO: Explain supported modes and returned shapes for easier debugging.
        if mode == "all":
            # inputs are raw embeddings
            reps = [self.get_latents(inputs[i], i) for i in range(len(inputs))]
            recons = []
            trans_to = []
            trans_from = []
            for i in range(self.n):
                rep = reps[i]
                for j in range(self.n):
                    if i == j:
                        recons.append(self.out_project(rep, j))
                    elif i == 0:
                        trans_from.append(self.out_project(rep, j))
                    elif j == 0:
                        trans_to.append(self.out_project(rep, j))

            return recons, trans_to, trans_from, reps
        if mode == "to":
            # inputs are translations to the first embedding space
            reps = [self.get_latents(inputs[i], 0) for i in range(len(inputs))]
            return [self.out_project(reps[i], i + 1) for i in range(len(inputs))]
        if mode == "from":
            # inputs are translations from the first embedding space
            reps = [self.get_latents(inputs[i], i + 1) for i in range(len(inputs))]
            return [self.out_project(reps[i], 0) for i in range(len(inputs))]
        if mode == "eval":
            if idx is None:
                raise ValueError("idx must be provided in eval mode")
            reps = [self.get_latents(inputs[i], idx) for i in range(len(inputs))]
            return [self.out_project(reps[i], 0) for i in range(len(inputs))]
        raise ValueError(f"Unsupported mode: {mode}")

    def get_latents(self, x: Tensor, idx: int) -> Tensor:
        """Encode input and apply shared transform to obtain latent representation."""
        z: Tensor = self.encoder[idx](x)
        out: Tensor = self.latent_transform(z)
        return out

    def out_project(self, x: Tensor, idx: int) -> Tensor:
        """Decode latent to target embedding space and optionally normalize output."""
        out: Tensor = self.decoder[idx](x)
        if self.normalize_embs:
            out = F.normalize(out, dim=1)
        return out


def init_weights(
    modules: Iterable[nn.Module], weight_init: Literal["kaiming", "xavier", "orthogonal"]
) -> None:
    for module in modules:
        if isinstance(module, nn.Linear):
            if weight_init == "kaiming":
                torch.nn.init.kaiming_normal_(
                    module.weight, a=0, mode="fan_in", nonlinearity="relu"
                )
            elif weight_init == "xavier":
                torch.nn.init.xavier_normal_(module.weight)
            elif weight_init == "orthogonal":
                torch.nn.init.orthogonal_(module.weight)
            module.bias.data.fill_(0)
        elif isinstance(module, nn.BatchNorm1d):
            torch.nn.init.normal_(module.weight, mean=1.0, std=0.02)
            torch.nn.init.normal_(module.bias, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)


class MLPWithResidual(nn.Module):
    """MLP stack with additive residual connections between layers.

    Residuals are size-adjusted when in/out dimensions differ. Optionally applies
    normalization to the final output.
    """

    def __init__(
        self,
        depth: int,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        norm_style: Literal["batch", "layer"] = "layer",
        output_norm: bool = False,
        weight_init: Literal["kaiming", "xavier", "orthogonal"] = "kaiming",
    ) -> None:
        super().__init__()
        self.depth = depth
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.layers = nn.ModuleList()

        norm_layer: type[nn.Module]
        if norm_style == "batch":
            norm_layer = nn.BatchNorm1d
        elif norm_style == "layer":
            norm_layer = nn.LayerNorm

        for layer_idx in range(self.depth):
            if layer_idx == 0:
                hidden_dim = out_dim if self.depth == 1 else hidden_dim
                self.layers.append(nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU()))
            elif layer_idx < self.depth - 1:
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.SiLU(),
                        norm_layer(hidden_dim),
                        nn.Dropout(p=0.1),
                    )
                )
            else:
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.Dropout(p=0.1),
                        nn.SiLU(),
                        nn.Linear(hidden_dim, out_dim),
                    )
                )
        init_weights(self.modules(), weight_init)

        self.output_norm: nn.LayerNorm | None
        if output_norm:
            self.output_norm = nn.LayerNorm(out_dim, elementwise_affine=False)
        else:
            self.output_norm = None

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            input_x = x
            x = layer(x)
            x = self.add_residual(input_x, x)

        if self.output_norm is not None:
            x = self.output_norm(x)

        return x

    def add_residual(self, input_x: Tensor, x: Tensor) -> Tensor:
        if input_x.shape[1] < x.shape[1]:
            padding = torch.zeros(x.shape[0], x.shape[1] - input_x.shape[1], device=x.device)
            input_x = torch.cat([input_x, padding], dim=1)
        elif input_x.shape[1] > x.shape[1]:
            input_x = input_x[:, : x.shape[1]]
        return x + input_x


class Discriminator(nn.Module):
    """Simple feed-forward discriminator that maps latent vectors to scalar logits.

    Configurable depth allows using either a single linear layer (depth=1) or
    a deeper MLP with SiLU activations and normalization layers.

    Supports spectral normalization for improved training stability.
    """

    def __init__(
        self,
        latent_dim: int,
        discriminator_dim: int = 1024,
        depth: int = 3,
        weight_init: Literal["kaiming", "orthogonal", "xavier"] = "kaiming",
        use_spectral_norm: bool = False,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.use_spectral_norm = use_spectral_norm
        self.dropout_p = dropout_p
        self.layers = nn.ModuleList()

        if depth < 1:
            raise ValueError("Depth must be at least 1")
        if depth == 1:
            linear = nn.Linear(latent_dim, 1)
            if use_spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            self.layers.append(linear)
        else:
            layers: list[nn.Module] = []
            first_linear = nn.Linear(latent_dim, discriminator_dim)
            if use_spectral_norm:
                first_linear = nn.utils.spectral_norm(first_linear)
            layers.extend([first_linear, nn.Dropout(self.dropout_p)])

            for _ in range(depth - 2):
                hidden_linear = nn.Linear(discriminator_dim, discriminator_dim)
                if use_spectral_norm:
                    hidden_linear = nn.utils.spectral_norm(hidden_linear)
                layers.extend(
                    [
                        nn.SiLU(),
                        hidden_linear,
                        nn.LayerNorm(discriminator_dim),
                        nn.Dropout(self.dropout_p),
                    ]
                )

            final_linear = nn.Linear(discriminator_dim, 1)
            if use_spectral_norm:
                final_linear = nn.utils.spectral_norm(final_linear)
            layers.extend([nn.SiLU(), final_linear])
            self.layers.append(nn.Sequential(*layers))

        init_weights(self.modules(), weight_init)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
