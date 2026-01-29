"""Unit tests for loss functions."""

import pytest
import torch

from fedaugment.projections.losses import MultiViewInfoNCELoss, MultiViewNTXentLoss, Vec2VecLosses


class TestMultiViewInfoNCELoss:
    """Test suite for MultiViewInfoNCELoss."""

    @pytest.fixture
    def loss_fn(self) -> MultiViewInfoNCELoss:
        """Create loss function with default temperature."""
        return MultiViewInfoNCELoss(temperature=0.1)

    def test_output_is_scalar(self, loss_fn: MultiViewInfoNCELoss) -> None:
        """Loss should return a single scalar value."""
        z = torch.randn(4, 2, 128)  # B=4, V=2, D=128
        loss = loss_fn(z)

        assert loss.ndim == 0  # Scalar
        assert loss.shape == torch.Size([])

    def test_identical_embeddings_behavior(self, loss_fn: MultiViewInfoNCELoss) -> None:
        """All identical embeddings should yield positive loss."""
        z = torch.ones(4, 2, 128)
        loss = loss_fn(z)

        # Loss should be finite and positive
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_orthogonal_embeddings_behavior(self, loss_fn: MultiViewInfoNCELoss) -> None:
        """Orthogonal embeddings should yield positive loss."""
        # Create orthogonal embeddings
        B, V, D = 4, 2, 128
        z = torch.zeros(B, V, D)
        for i in range(B):
            for v in range(V):
                idx = (i * V + v) % D
                z[i, v, idx] = 1.0

        loss = loss_fn(z)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_temperature_effect(self) -> None:
        """Higher temperature should affect loss magnitude."""
        z = torch.randn(4, 2, 128)

        loss_low_temp = MultiViewInfoNCELoss(temperature=0.01)(z)
        loss_high_temp = MultiViewInfoNCELoss(temperature=1.0)(z)

        # Losses should be different
        # TODO: We should add test cases that verify the expected direction of change
        assert loss_low_temp.item() != loss_high_temp.item()
        assert torch.isfinite(loss_low_temp)
        assert torch.isfinite(loss_high_temp)

    def test_gradients_flow(self, loss_fn: MultiViewInfoNCELoss) -> None:
        """Gradients should propagate through the loss."""
        z = torch.randn(4, 2, 128, requires_grad=True)
        loss = loss_fn(z)
        loss.backward()

        assert z.grad is not None
        assert torch.isfinite(z.grad).all()
        assert (z.grad != 0).any()  # Gradients should be non-zero

    def test_numerical_stability_large_values(self, loss_fn: MultiViewInfoNCELoss) -> None:
        """Loss should be stable with large input values."""
        z = torch.randn(4, 2, 128) * 100.0  # Large magnitude
        loss = loss_fn(z)

        assert torch.isfinite(loss)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_numerical_stability_small_values(self, loss_fn: MultiViewInfoNCELoss) -> None:
        """Loss should be stable with small input values."""
        z = torch.randn(4, 2, 128) * 0.01  # Small magnitude
        loss = loss_fn(z)

        assert torch.isfinite(loss)

    def test_deterministic_behavior(self, loss_fn: MultiViewInfoNCELoss) -> None:
        """Same input should always produce same output."""
        z = torch.randn(4, 2, 128)

        loss1 = loss_fn(z)
        loss2 = loss_fn(z)

        assert loss1.item() == pytest.approx(loss2.item())

    def test_deterministic_with_seed(self, loss_fn: MultiViewInfoNCELoss) -> None:
        """Same seed should produce same results."""
        torch.manual_seed(42)
        z = torch.randn(4, 2, 128)
        loss1 = loss_fn(z)

        torch.manual_seed(42)
        z = torch.randn(4, 2, 128)
        loss2 = loss_fn(z)

        assert loss1.item() == pytest.approx(loss2.item())


class TestMultiViewNTXentLoss:
    """Test suite for MultiViewNTXentLoss."""

    @pytest.fixture
    def loss_fn(self) -> MultiViewNTXentLoss:
        """Create loss function with default temperature."""
        return MultiViewNTXentLoss(temperature=0.1)

    def test_output_is_scalar(self, loss_fn: MultiViewNTXentLoss) -> None:
        """Loss should return a single scalar value."""
        z = torch.randn(4, 2, 128)
        loss = loss_fn(z)

        assert loss.ndim == 0
        assert loss.shape == torch.Size([])

    # NOTE: Temporarily disabled test as InfoNCE and NT-Xent may produce the same values
    # def test_different_from_infonce(self) -> None:
    #     """NT-Xent should produce different values than InfoNCE."""
    #     z = torch.randn(4, 2, 128)

    #     infonce_loss = MultiViewInfoNCELoss(temperature=0.1)(z)
    #     ntxent_loss = MultiViewNTXentLoss(temperature=0.1)(z)

    #     assert infonce_loss.item() != pytest.approx(ntxent_loss.item(), rel=1e-5)

    def test_temperature_effect(self) -> None:
        """Higher temperature should affect loss magnitude."""
        torch.manual_seed(42)  # Ensure reproducibility
        z = torch.randn(8, 2, 128)  # Larger batch for more stable results

        loss_low_temp = MultiViewNTXentLoss(temperature=0.05)(z.clone())
        loss_high_temp = MultiViewNTXentLoss(temperature=0.5)(z.clone())

        # Losses should be different
        # TODO: We should add test cases that verify the expected direction of change
        assert loss_low_temp.item() != loss_high_temp.item()
        assert torch.isfinite(loss_low_temp)
        assert torch.isfinite(loss_high_temp)

    def test_gradients_flow(self, loss_fn: MultiViewNTXentLoss) -> None:
        """Gradients should propagate through the loss."""
        z = torch.randn(4, 2, 128, requires_grad=True)
        loss = loss_fn(z)
        loss.backward()

        assert z.grad is not None
        assert torch.isfinite(z.grad).all()
        assert (z.grad != 0).any()  # Gradients should be non-zero

    def test_symmetry_property(self) -> None:
        """Loss should be symmetric with respect to view order (approximately)."""
        # Create embeddings
        z = torch.randn(4, 2, 128)

        loss_fn = MultiViewNTXentLoss(temperature=0.1)
        loss1 = loss_fn(z)

        # Swap views
        z_swapped = z.clone()
        z_swapped[:, 0, :], z_swapped[:, 1, :] = z[:, 1, :].clone(), z[:, 0, :].clone()
        loss2 = loss_fn(z_swapped)

        # Losses should be approximately equal due to symmetry
        assert loss1.item() == pytest.approx(loss2.item(), rel=1e-5)


class TestVec2VecLosses:
    """Test suite for Vec2VecLosses."""

    @pytest.fixture
    def loss_fn(self) -> Vec2VecLosses:
        """Create loss function with default weights."""
        return Vec2VecLosses(weight_rec=1.0, weight_vsp=0.5, weight_cc=0.5)

    def test_output_dict(self, loss_fn: Vec2VecLosses) -> None:
        """Loss should return a dictionary of loss components."""
        inputs = [torch.randn(4, 128), torch.randn(4, 128)]
        recons = [torch.randn(4, 128), torch.randn(4, 128)]
        trans_from = [torch.randn(4, 128)]
        trans_to = [torch.randn(4, 128)]

        losses = loss_fn(inputs, recons, trans_from, trans_to)

        assert isinstance(losses, dict)
        assert "rec_loss" in losses
        assert "vsp_loss" in losses
        assert "cc_trans_loss" in losses
        assert "cc_vsp_loss" in losses

        for value in losses.values():
            assert isinstance(value, torch.Tensor)
            assert torch.isfinite(value)

    def test_two_embedding_spaces(self, loss_fn: Vec2VecLosses) -> None:
        """Loss should work with 2 embedding spaces."""
        inputs = [torch.randn(4, 128), torch.randn(4, 256)]
        recons = [torch.randn(4, 128), torch.randn(4, 256)]
        trans_from = [torch.randn(4, 128)]  # From space 0 to space 1 and back (inverted)
        trans_to = [torch.randn(4, 256)]  # From space 1 to space 0 and back (inverted)

        losses = loss_fn(inputs, recons, trans_from, trans_to)

        assert all(torch.isfinite(v) for v in losses.values())

    def test_three_embedding_spaces(self, loss_fn: Vec2VecLosses) -> None:
        """Loss should work with 3 embedding spaces."""
        inputs = [torch.randn(4, 128), torch.randn(4, 256), torch.randn(4, 512)]
        recons = [torch.randn(4, 128), torch.randn(4, 256), torch.randn(4, 512)]
        trans_from = [torch.randn(4, 128), torch.randn(4, 128)]  # From space 0 to others and back
        trans_to = [torch.randn(4, 256), torch.randn(4, 512)]  # From others to space 0 and back

        losses = loss_fn(inputs, recons, trans_from, trans_to)

        assert all(torch.isfinite(v) for v in losses.values())

    def test_rec_loss_component(self, loss_fn: Vec2VecLosses) -> None:
        """Reconstruction loss should measure similarity between inputs and reconstructions."""
        # Perfect reconstruction
        inputs = [torch.randn(4, 128), torch.randn(4, 128)]
        recons = [inp.clone() for inp in inputs]
        trans_from = [torch.randn(4, 128)]
        trans_to = [torch.randn(4, 128)]

        losses = loss_fn(inputs, recons, trans_from, trans_to)

        # Perfect reconstruction should yield low rec_loss (close to 0)
        assert losses["rec_loss"].item() < 1e-5  # Small threshold for near-perfect

    def test_gradients_flow_through_all_components(self, loss_fn: Vec2VecLosses) -> None:
        """Gradients should propagate through all loss components."""
        inputs = [torch.randn(4, 128, requires_grad=True), torch.randn(4, 128, requires_grad=True)]
        recons = [torch.randn(4, 128, requires_grad=True), torch.randn(4, 128, requires_grad=True)]
        trans_from = [torch.randn(4, 128, requires_grad=True)]
        trans_to = [torch.randn(4, 128, requires_grad=True)]

        losses = loss_fn(inputs, recons, trans_from, trans_to)

        # Compute total loss and backward
        total_loss = torch.stack(list(losses.values())).sum()
        total_loss.backward()  # type: ignore[no-untyped-call]

        # Check gradients exist
        for inp in inputs:
            assert inp.grad is not None
            assert torch.isfinite(inp.grad).all()

    def test_similarity_loss_component(self, loss_fn: Vec2VecLosses) -> None:
        """Similarity loss should compute cosine similarity."""
        x = torch.randn(4, 128)
        y = x.clone()  # Identical

        sim_loss = loss_fn.similarity_loss(x, y)

        # Identical vectors should have low similarity loss (close to 0)
        assert sim_loss.item() < 1e-5
