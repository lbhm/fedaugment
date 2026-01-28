"""Unit tests for projection metrics."""

import pytest
import torch

from fedaugment.projections.metrics import (
    CosineSimilarityMean,
    CosineSimilarityOrchestrator,
    MeanRelativeChunkRank,
    OrchestratorChild,
    SameLabelCosineSimilarityMean,
    TopKSimilarityAccuracy,
    TopKSimilarityPrecision,
)


@pytest.fixture
def orchestrator() -> CosineSimilarityOrchestrator:
    """Create a fresh orchestrator for each test."""
    return CosineSimilarityOrchestrator()


class TestTopKSimilarityPrecision:
    def test_perfect_retrieval(self, orchestrator: CosineSimilarityOrchestrator) -> None:
        """All top-k neighbors with same label should give precision=1.0."""
        # Create embeddings where label groups are identical
        projections = torch.tensor(
            [
                [1.0, 0.0],  # label 0
                [1.0, 0.0],  # label 0
                [1.0, 0.0],  # label 0
                [0.0, 1.0],  # label 1
                [0.0, 1.0],  # label 1
                [0.0, 1.0],  # label 1
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 0, 1, 1, 1])

        child_metric = TopKSimilarityPrecision(k=2)
        metric = OrchestratorChild(orchestrator, child_metric)

        orchestrator.update(projections, labels)
        precision = metric.compute()

        assert precision.item() == pytest.approx(1.0)

    def test_no_retrieval(self, orchestrator: CosineSimilarityOrchestrator) -> None:
        """Orthogonal embeddings with different labels -> precision=0.0."""
        # Create orthogonal embeddings with all different labels
        projections = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        )
        labels = torch.tensor([0, 1, 2])

        child_metric = TopKSimilarityPrecision(k=2)
        metric = OrchestratorChild(orchestrator, child_metric)

        orchestrator.update(projections, labels)
        precision = metric.compute()

        assert precision.item() == pytest.approx(0.0)

    def test_half_precision(self, orchestrator: CosineSimilarityOrchestrator) -> None:
        """Test that precision matches in half precision."""
        projections = torch.randn(20, 64)
        labels = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long)])

        # Full precision
        orch_full = CosineSimilarityOrchestrator(use_half_precision=False)
        child_full = TopKSimilarityPrecision(k=3)
        metric_full = OrchestratorChild(orch_full, child_full)
        orch_full.update(projections, labels)
        result_full = metric_full.compute()

        # Half precision
        orch_half = CosineSimilarityOrchestrator(use_half_precision=True)
        child_half = TopKSimilarityPrecision(k=3)
        metric_half = OrchestratorChild(orch_half, child_half)
        orch_half.update(projections, labels)
        result_half = metric_half.compute()

        # Results should be close (allowing for half precision error)
        assert result_full.item() == pytest.approx(result_half.item(), abs=1e-2)

    @pytest.mark.parametrize("k", [1, 3, 5, 10])
    def test_various_k_values(self, orchestrator: CosineSimilarityOrchestrator, k: int) -> None:
        """Metric should work with various k values."""
        projections = torch.randn(20, 128)
        labels = torch.randint(0, 5, (20,))

        child_metric = TopKSimilarityPrecision(k=k)
        metric = OrchestratorChild(orchestrator, child_metric)

        orchestrator.update(projections, labels)
        precision = metric.compute()

        assert 0.0 <= precision.item() <= 1.0
        assert torch.isfinite(precision)

    def test_state_accumulation(self, orchestrator: CosineSimilarityOrchestrator) -> None:
        """Multiple updates should accumulate state correctly."""
        child_metric = TopKSimilarityPrecision(k=2)
        metric = OrchestratorChild(orchestrator, child_metric)

        # First batch
        orchestrator.update(torch.randn(4, 64), torch.tensor([0, 0, 1, 1]))
        # Second batch
        orchestrator.update(torch.randn(4, 64), torch.tensor([2, 2, 3, 3]))

        # Should have accumulated 8 samples total
        result = metric.compute()
        assert torch.isfinite(result)

    def test_reset_behavior(self, orchestrator: CosineSimilarityOrchestrator) -> None:
        """Metric should reset properly between epochs."""
        child_metric = TopKSimilarityPrecision(k=2)
        metric = OrchestratorChild(orchestrator, child_metric)

        orchestrator.update(torch.randn(4, 64), torch.tensor([0, 0, 1, 1]))
        _ = metric.compute()

        # Reset
        metric.reset()

        # Orchestrator state should be cleared
        assert len(orchestrator.projections) == 0
        assert len(orchestrator.labels) == 0


class TestTopKSimilarityAccuracy:
    def test_perfect_retrieval(self, orchestrator: CosineSimilarityOrchestrator) -> None:
        """All neighbors with same label should give accuracy=1.0."""
        projections = torch.tensor(
            [
                [1.0, 0.0],  # label 0
                [1.0, 0.0],  # label 0
                [0.0, 1.0],  # label 1
                [0.0, 1.0],  # label 1
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 1, 1])

        child_metric = TopKSimilarityAccuracy(k=1)
        metric = OrchestratorChild(orchestrator, child_metric)

        orchestrator.update(projections, labels)
        accuracy = metric.compute()

        assert accuracy.item() == pytest.approx(1.0)

    def test_no_retrieval(self, orchestrator: CosineSimilarityOrchestrator) -> None:
        """No matches in top-k -> accuracy=0.0."""
        projections = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        )
        labels = torch.tensor([0, 1, 2])

        child_metric = TopKSimilarityAccuracy(k=1)
        metric = OrchestratorChild(orchestrator, child_metric)

        orchestrator.update(projections, labels)
        accuracy = metric.compute()

        assert accuracy.item() == pytest.approx(0.0)


class TestMeanCosineSimilarity:
    def test_orthogonal_vectors(self, orchestrator: CosineSimilarityOrchestrator) -> None:
        """Orthogonal vectors should have mean similarity ≈ 0."""
        projections = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        )
        labels = torch.tensor([0, 1, 2])

        child_metric = CosineSimilarityMean()
        metric = OrchestratorChild(orchestrator, child_metric)

        orchestrator.update(projections, labels)
        mean_sim = metric.compute()

        assert mean_sim.item() == pytest.approx(0.0, abs=1e-5)

    def test_identical_vectors(self, orchestrator: CosineSimilarityOrchestrator) -> None:
        """Identical vectors should have mean similarity = 1.0."""
        projections = torch.ones(5, 3)
        labels = torch.arange(5)

        child_metric = CosineSimilarityMean()
        metric = OrchestratorChild(orchestrator, child_metric)

        orchestrator.update(projections, labels)
        mean_sim = metric.compute()

        assert mean_sim.item() == pytest.approx(1.0, abs=1e-5)

    def test_opposite_vectors(self, orchestrator: CosineSimilarityOrchestrator) -> None:
        """Opposite vectors should have mean similarity = -1.0."""
        projections = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float32)
        labels = torch.tensor([0, 1])

        child_metric = CosineSimilarityMean()
        metric = OrchestratorChild(orchestrator, child_metric)

        orchestrator.update(projections, labels)
        mean_sim = metric.compute()

        assert mean_sim.item() == pytest.approx(-1.0, abs=1e-5)

    def test_range(self, orchestrator: CosineSimilarityOrchestrator) -> None:
        """Mean cosine similarity should be in [-1, 1]."""
        projections = torch.randn(20, 64)
        labels = torch.arange(20)

        child_metric = CosineSimilarityMean()
        metric = OrchestratorChild(orchestrator, child_metric)

        orchestrator.update(projections, labels)
        mean_sim = metric.compute()

        assert -1.0 <= mean_sim.item() <= 1.0


class TestMeanSameLabelCosineSimilarity:
    def test_perfect_clustering(self, orchestrator: CosineSimilarityOrchestrator) -> None:
        """Identical embeddings within label groups -> similarity = 1.0."""
        projections = torch.tensor(
            [
                [1.0, 0.0],  # label 0
                [1.0, 0.0],  # label 0
                [0.0, 1.0],  # label 1
                [0.0, 1.0],  # label 1
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 1, 1])

        child_metric = SameLabelCosineSimilarityMean()
        metric = OrchestratorChild(orchestrator, child_metric)

        orchestrator.update(projections, labels)
        mean_sim = metric.compute()

        assert mean_sim.item() == pytest.approx(1.0, abs=1e-5)

    def test_no_same_label_pairs(self, orchestrator: CosineSimilarityOrchestrator) -> None:
        """When all labels are unique, metric should return 0 (no pairs to average)."""
        projections = torch.randn(5, 64)
        labels = torch.arange(5)  # All unique

        child_metric = SameLabelCosineSimilarityMean()
        metric = OrchestratorChild(orchestrator, child_metric)

        orchestrator.update(projections, labels)
        mean_sim = metric.compute()

        # With no same-label pairs (excluding self), should return 0
        assert mean_sim.item() == pytest.approx(0.0, abs=1e-5)

    def test_range(self, orchestrator: CosineSimilarityOrchestrator) -> None:
        """Mean same-label cosine similarity should be in [-1, 1]."""
        projections = torch.randn(20, 64)
        # Create some repeated labels
        labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6])

        child_metric = SameLabelCosineSimilarityMean()
        metric = OrchestratorChild(orchestrator, child_metric)

        orchestrator.update(projections, labels)
        mean_sim = metric.compute()

        assert -1.0 <= mean_sim.item() <= 1.0


class TestMeanRelativeChunkRank:
    def test_perfect_ranking(self, orchestrator: CosineSimilarityOrchestrator) -> None:
        """When same-label items are most similar, rank should be 1.0."""
        projections = torch.tensor(
            [
                [1.0, 0.0],  # label 0
                [1.0, 0.0],  # label 0
                [0.0, 1.0],  # label 1
                [0.0, 1.0],  # label 1
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 1, 1])

        child_metric = MeanRelativeChunkRank()
        metric = OrchestratorChild(orchestrator, child_metric)

        orchestrator.update(projections, labels)
        mean_rank = metric.compute()

        assert mean_rank.item() == pytest.approx(1.0, abs=1e-5)

    def test_worst_ranking(self, orchestrator: CosineSimilarityOrchestrator) -> None:
        """When same-label items are least similar, rank should be low."""
        projections = torch.tensor(
            [
                [1.0, 0.0],  # label 0
                [-1.0, 0.0],  # label 0 (opposite to first)
                [0.0, 1.0],  # label 1
                [0.0, -1.0],  # label 1 (opposite to third)
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 0, 1, 1])

        child_metric = MeanRelativeChunkRank()
        metric = OrchestratorChild(orchestrator, child_metric)

        orchestrator.update(projections, labels)
        mean_rank = metric.compute()

        # Same-label matches are worst within each chunk but not necessarily rank 0
        # Just verify it's a valid rank value and lower than perfect case
        assert 0.0 <= mean_rank.item() <= 1.0
        assert mean_rank.item() < 0.9  # Should be worse than nearly perfect

    def test_range(self, orchestrator: CosineSimilarityOrchestrator) -> None:
        """Mean relative chunk rank should be in [0, 1]."""
        projections = torch.randn(20, 64)
        labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6])

        child_metric = MeanRelativeChunkRank()
        metric = OrchestratorChild(orchestrator, child_metric)

        orchestrator.update(projections, labels)
        mean_rank = metric.compute()

        assert 0.0 <= mean_rank.item() <= 1.0


class TestCosineSimilarityOrchestrator:
    def test_multiple_children(self) -> None:
        """Orchestrator should handle multiple child metrics."""
        orchestrator = CosineSimilarityOrchestrator()

        # Register multiple children
        topk_prec = TopKSimilarityPrecision(k=3)
        topk_acc = TopKSimilarityAccuracy(k=3)
        mean_sim = CosineSimilarityMean()

        metric1 = OrchestratorChild(orchestrator, topk_prec)
        metric2 = OrchestratorChild(orchestrator, topk_acc)
        metric3 = OrchestratorChild(orchestrator, mean_sim)

        # All should share the same orchestrator
        assert metric1.orchestrator is metric2.orchestrator
        assert metric2.orchestrator is metric3.orchestrator

        # Update once
        projections = torch.randn(10, 64)
        labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3])

        # Only need to update orchestrator once
        orchestrator.update(projections, labels)

        # All metrics should compute successfully
        result1 = metric1.compute()
        result2 = metric2.compute()
        result3 = metric3.compute()

        assert torch.isfinite(result1)
        assert torch.isfinite(result2)
        assert torch.isfinite(result3)

    def test_cache_invalidation(self) -> None:
        """Cache should be invalidated after update."""
        orchestrator = CosineSimilarityOrchestrator()
        child_metric = CosineSimilarityMean()
        metric = OrchestratorChild(orchestrator, child_metric)

        # First update and compute
        orchestrator.update(torch.randn(5, 64), torch.arange(5))
        result1 = metric.compute()

        # Verify cache is set
        assert orchestrator._cached_results is not None
        assert not orchestrator._is_dirty

        # Second update should invalidate cache
        orchestrator.update(torch.randn(5, 64), torch.arange(5))
        assert orchestrator._is_dirty
        assert orchestrator._cached_results is None

        # Compute should work with new data
        result2 = metric.compute()
        assert torch.isfinite(result1)
        assert torch.isfinite(result2)

    def test_chunked_processing(self) -> None:
        """Chunked processing should give same results as full matrix."""
        projections = torch.randn(20, 64)
        labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6])

        # Full matrix (no chunking)
        orch_full = CosineSimilarityOrchestrator(chunk_size=None)
        child_full = TopKSimilarityPrecision(k=3)
        metric_full = OrchestratorChild(orch_full, child_full)
        orch_full.update(projections, labels)
        result_full = metric_full.compute()

        # Chunked processing
        orch_chunk = CosineSimilarityOrchestrator(chunk_size=5)
        child_chunk = TopKSimilarityPrecision(k=3)
        metric_chunk = OrchestratorChild(orch_chunk, child_chunk)
        orch_chunk.update(projections, labels)
        result_chunk = metric_chunk.compute()

        # Results should be identical
        assert result_full.item() == pytest.approx(result_chunk.item(), abs=1e-6)

    def test_state_reset(self) -> None:
        """Orchestrator state should reset properly."""
        orchestrator = CosineSimilarityOrchestrator()
        child_metric = CosineSimilarityMean()
        metric = OrchestratorChild(orchestrator, child_metric)

        # Add data
        orchestrator.update(torch.randn(5, 64), torch.arange(5))
        _ = metric.compute()

        # Reset
        metric.reset()

        # State should be cleared
        assert len(orchestrator.projections) == 0
        assert len(orchestrator.labels) == 0

    @pytest.mark.parametrize("chunk_size", [None, 5, 10, 20])
    def test_various_chunk_sizes(self, chunk_size: int | None) -> None:
        """Different chunk sizes should all work correctly."""
        orchestrator = CosineSimilarityOrchestrator(chunk_size=chunk_size)
        child_metric = TopKSimilarityPrecision(k=3)
        metric = OrchestratorChild(orchestrator, child_metric)

        projections = torch.randn(20, 64)
        labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6])

        orchestrator.update(projections, labels)
        result = metric.compute()

        assert torch.isfinite(result)
        assert 0.0 <= result.item() <= 1.0
