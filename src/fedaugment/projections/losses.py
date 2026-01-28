import abc
import sys
from itertools import starmap
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fedaugment.config import CriterionConfig


def get_criterion(config: CriterionConfig) -> nn.Module:
    """Get the criterion for a model.

    Args:
        config: The criterion configuration.

    Returns:
        The criterion for the model.
    """
    criterion_class: type[nn.Module] | None
    try:
        # Check if the class exists in our custom loss module
        criterion_class = getattr(sys.modules[__name__], config.class_)
    except AttributeError:
        # If not, check if exists in torch's loss module
        criterion_class = getattr(torch.nn.modules.loss, config.class_, None)  # ty: ignore[unresolved-attribute]

    if criterion_class is None:
        raise ValueError(f"Criterion class {config.class_} not found.")
    return criterion_class(**config.kwargs)


class MultiViewLoss(nn.Module, abc.ABC):
    """Abstract base class for multi-view loss functions."""

    MIN_VIEWS: ClassVar[int] = 2

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    @abc.abstractmethod
    def forward(self, z: Tensor) -> Tensor:
        """Compute the loss for a batch of multi-view embeddings.

        Args:
            z: Tensor of shape (B, V, D), where
                B = batch size,
                V = number of views,
                D = embedding dimension.

        Returns:
            Scalar loss.
        """


class MultiViewInfoNCELoss(MultiViewLoss):
    """Multi-view loss for a batch of positive pairs.

    This is a contrastive loss using all other samples in the batch as negatives.
    """

    def forward(self, z: Tensor) -> Tensor:
        B, V, D = z.shape
        if V < self.MIN_VIEWS:
            raise ValueError(f"Multi-view loss requires at least 2 views, got {V}")

        # Step 1: Normalize along embedding dimension and reshape to (B*V, D)
        z = F.normalize(z.view(B * V, D), dim=1)

        # Step 2: Compute pairwise similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # (B*V, B*V)
        sim_matrix.fill_diagonal_(-1e4)  # avoid self-similarity

        # Step 3: Build positive pair mask (B*V, B*V)
        labels = torch.arange(B, device=z.device).repeat_interleave(V)  # (B*V,)
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B*V, B*V)
        pos_mask.fill_diagonal_(False)  # remove self-pairs

        # Step 4: Log-softmax over rows
        log_probs = F.log_softmax(sim_matrix, dim=1)  # (B*V, B*V)

        # Step 5: Average negative log-prob of positives
        return -log_probs[pos_mask].mean()


class MultiViewNTXentLoss(MultiViewLoss):
    def forward(self, z: Tensor) -> Tensor:
        B, V, D = z.shape
        if V < self.MIN_VIEWS:
            raise ValueError(f"Multi-view loss requires at least 2 views, got {V}")

        # Normalize along embedding dimension and reshape to (B*V, D)
        z = F.normalize(z.view(B * V, D), dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # (B*V, B*V)

        # Build positive mask: views of same sample (excluding self-similarity)
        labels = torch.arange(B, device=z.device).repeat_interleave(V)  # (B*V,)
        pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B*V, B*V)
        pos_mask.fill_diagonal_(False)  # Exclude self-pairs

        # Compute logsumexp with diagonal excluded
        # Use index_fill which seems to be slightly more efficient than masked_fill for this case
        diag_idx = torch.arange(B * V, device=z.device)
        sim_matrix_masked = sim_matrix.clone()  # Still need to clone to avoid autograd issues
        sim_matrix_masked[diag_idx, diag_idx] = -torch.inf

        # Compute log probabilities
        log_probs = sim_matrix - torch.logsumexp(sim_matrix_masked, dim=1, keepdim=True)

        # Average log-probabilities of positive pairs
        mean_log_prob_pos = (log_probs * pos_mask).sum(dim=1) / pos_mask.sum(dim=1)

        # NT-Xent = -log probability mean
        return -mean_log_prob_pos.mean()


class Vec2VecLosses(nn.Module):
    """Loss function defined by the vec2vec paper."""

    def __init__(self, weight_rec: float, weight_vsp: float, weight_cc: float) -> None:
        super().__init__()
        self.weight_rec = weight_rec
        self.weight_vsp = weight_vsp
        self.weight_cc = weight_cc

    def forward(
        self,
        inputs: list[Tensor],
        recons: list[Tensor],
        trans_from_inverted: list[Tensor],
        trans_to_inverted: list[Tensor],
    ) -> dict[str, Tensor]:
        rec = self.rec_loss(inputs, recons)
        vsp = self.vsp_loss_recon(inputs, recons)
        cc_trans = self.cc_loss(inputs, trans_from_inverted, trans_to_inverted)
        cc_vsp = self.vsp_loss(inputs, trans_to_inverted, trans_from_inverted)
        return {
            "rec_loss": rec * self.weight_rec,
            "vsp_loss": vsp * self.weight_vsp,
            "cc_trans_loss": cc_trans * self.weight_cc,
            "cc_vsp_loss": cc_vsp * self.weight_cc,
        }

    def similarity_loss(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute the cosine similarity loss between two tensors."""
        # NOTE: I think we could replace this with nn.CosineEmbeddingLoss
        return 1 - (F.cosine_similarity(x, y, dim=1).mean() + 1e-8)

    def rec_loss(self, inputs: list[Tensor], recons: list[Tensor]) -> Tensor:
        """Compute the reconstruction loss."""
        assert len(inputs) == len(recons), "Inputs and reconstructions must have the same length."
        losses = list(starmap(self.similarity_loss, zip(inputs, recons, strict=False)))
        return torch.stack(losses).mean()

    def cc_loss(
        self,
        inputs: list[Tensor],
        trans_from_inverted: list[Tensor],
        trans_to_inverted: list[Tensor],
    ) -> Tensor:
        """Compute the cycle-consistency loss."""
        assert len(trans_from_inverted) == len(trans_to_inverted), (
            "Inverted translations must have the same length."
        )
        # first Fi1(F1i(x)), then F1i(Fi1(x))
        losses = [self.similarity_loss(inputs[0], tfi) for tfi in trans_from_inverted] + [
            self.similarity_loss(inputs[i + 1], tti) for i, tti in enumerate(trans_to_inverted)
        ]
        return torch.stack(losses).mean()

    def vsp_loss(
        self, inputs: list[Tensor], trans_from: list[Tensor], trans_to: list[Tensor]
    ) -> Tensor:
        """Compute the vector space preservation loss.

        Args:
            inputs (list[Tensor]): List of original input tensors.
            trans_from (list[Tensor]): Transformations of the first input into other vector spaces.
            trans_to (list[Tensor]): Transformations of other inputs into the first vector space.

        Returns:
            Tensor: The combined vector space preservation loss.
        """
        assert len(trans_from) == len(trans_to), "Translations must have the same length."
        assert len(inputs) == len(trans_to) + 1, "Inputs must match the number of translations."

        losses: list[Tensor] = []

        # Compute VSP loss for transformations from the first input to other vector spaces
        input_norm = F.normalize(inputs[0], dim=1)
        input_sims = input_norm @ input_norm.T
        for trans in trans_from:
            trans_norm = F.normalize(trans, dim=1)
            min_dim = min(input_norm.shape[1], trans_norm.shape[1])

            # Compute similarity matrices
            trans_sims = trans_norm @ trans_norm.T
            trans_sims_reflected = trans_norm[:, :min_dim] @ input_norm[:, :min_dim].T
            # Compute the loss
            vsp_loss = (input_sims - trans_sims).abs().mean()
            vsp_loss_reflected = (input_sims - trans_sims_reflected).abs().mean()
            losses.append(vsp_loss + vsp_loss_reflected)

        # Compute VSP loss for transformations from other inputs to the first vector space
        for i, trans in enumerate(trans_to):
            # Normalize inputs[i + 1] and trans_to[i]
            input_norm = F.normalize(inputs[i + 1], dim=1)
            trans_norm = F.normalize(trans, dim=1)
            min_dim = min(input_norm.shape[1], trans_norm.shape[1])
            # Compute similarity matrices
            input_sims = input_norm @ input_norm.T
            trans_sims = trans_norm @ trans_norm.T
            trans_sims_reflected = trans_norm[:, :min_dim] @ input_norm[:, :min_dim].T
            # Compute the loss
            vsp_loss = (input_sims - trans_sims).abs().mean()
            vsp_loss_reflected = (input_sims - trans_sims_reflected).abs().mean()
            losses.append(vsp_loss + vsp_loss_reflected)

        # Combine all losses
        return torch.stack(losses).mean()

    def vsp_loss_recon(
        self, inputs: list[Tensor], recons: list[Tensor], eps: float = 1e-10
    ) -> Tensor:
        assert len(inputs) == len(recons), "Inputs and reconstructions must have the same length."
        losses = []
        for i in range(len(inputs)):
            inp = F.normalize(inputs[i].detach(), dim=1)
            in_sims = inp @ inp.T
            out = F.normalize(recons[i], dim=1)
            out_sims = out @ out.T
            out_sims_reflected = out @ inp.T
            vsp_loss = (in_sims - out_sims).abs().mean()
            vsp_loss_reflected = (in_sims - out_sims_reflected).abs().mean()
            losses.append(vsp_loss + vsp_loss_reflected)
        return torch.stack(losses).mean()
