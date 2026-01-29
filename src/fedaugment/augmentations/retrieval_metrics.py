"""Shared retrieval metrics for table augmentation evaluators."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal


class MetricType(Enum):
    """Enumeration of available retrieval metrics."""

    # Basic metrics
    PRECISION = "precision"
    """The fraction of relevant retrieved items over the total amount of retrieved items."""

    RECALL = "recall"
    """The fraction of relevant retrieved items over the total amount of relevant items."""

    MAX_PRECISION = "max_precision"
    """The best possible precision with k predictions and the given ground truth size."""

    MAX_RECALL = "max_recall"
    """The best possible recall with k predictions and the given ground truth size."""

    # Other metrics
    HITS = "hits"
    """1.0 if there is at least one relevant item in the top-k predictions, otherwise 0.0."""

    MRR = "mrr"
    """Mean Reciprocal Rank (MRR) is the reciprocal of the rank of the first correct prediction."""

    AP = "ap"
    """Average Precision (AP) is the average of precision values at each position where a
    relevant item is retrieved."""

    def format_with_k(self, k: int) -> str:
        """Format metric name with k value (e.g., 'recall@10')."""
        return f"{self.value}@{k}"


class RetrievalMetrics:
    """Container for retrieval metric configurations and results."""

    metrics: tuple[MetricType, ...] = (
        MetricType.PRECISION,
        MetricType.RECALL,
        MetricType.MAX_PRECISION,
        MetricType.MAX_RECALL,
        MetricType.HITS,
        MetricType.MRR,
        MetricType.AP,
    )

    @classmethod
    def get_column_names(cls, k_vals: list[int], mode: Literal["join", "union"]) -> list[str]:
        """Generate column names for all metrics and k values."""
        col_names = ["query_table"] if mode == "union" else ["query_id"]
        for k in k_vals:
            col_names.extend([metric.format_with_k(k) for metric in cls.metrics])
        return col_names

    def __init__(
        self,
        precision: float,
        recall: float,
        max_precision: float,
        max_recall: float,
        hits: float,
        mrr: float,
        ap: float,
    ) -> None:
        self.precision = precision
        self.recall = recall
        self.max_precision = max_precision
        self.max_recall = max_recall
        self.hits = hits
        self.mrr = mrr
        self.ap = ap

    def to_list(self) -> list[float]:
        """Convert to list in the order defined by `RetrievalMetrics.metrics`."""
        return [
            self.precision,
            self.recall,
            self.max_precision,
            self.max_recall,
            self.hits,
            self.mrr,
            self.ap,
        ]


@dataclass(frozen=True)
class MatchStatistics:
    """Statistics about prediction matches for a single query.

    This intermediate representation makes metric calculations self-documenting and easier to test.
    """

    match_positions: list[int]
    """1-based positions where predictions matched ground truth."""
    n_predictions: int
    """Total number of predictions (k)."""
    n_ground_truth: int
    """Total number of ground truth items."""


def compute_match_statistics(
    predictions: Sequence[str], ground_truth: set[str], k: int
) -> MatchStatistics:
    """Compute match statistics between predictions and ground truth.

    Args:
        predictions: Ordered sequence of predicted items (e.g., column IDs or table names).
        ground_truth: Set of ground truth items.
        k: Number of top predictions to consider.

    Returns:
        MatchStatistics containing match positions and counts.
    """
    match_positions = []
    for position, pred in enumerate(predictions[:k], start=1):
        if pred in ground_truth:
            match_positions.append(position)

    return MatchStatistics(
        match_positions=match_positions, n_predictions=k, n_ground_truth=len(ground_truth)
    )


def compute_retrieval_metrics(stats: MatchStatistics) -> RetrievalMetrics:
    """Compute all retrieval metrics from match statistics.

    Args:
        stats: Match statistics containing positions and counts.

    Returns:
        RetrievalMetrics containing all computed metrics.
    """
    n_matches = len(stats.match_positions)
    k = stats.n_predictions
    n_ground_truth = stats.n_ground_truth

    # Basic metrics
    recall = n_matches / n_ground_truth if n_ground_truth > 0 else 0.0
    precision = n_matches / k if k > 0 else 0.0
    hits = 1.0 if n_matches > 0 else 0.0

    # Ranking metrics
    mrr = 1.0 / stats.match_positions[0] if stats.match_positions else 0.0
    ap = _compute_average_precision(stats.match_positions, n_ground_truth)

    # Normalized metrics
    max_recall = min(1.0, k / n_ground_truth) if n_ground_truth > 0 else 0.0
    max_precision = min(k, n_ground_truth) / k if k > 0 else 0.0

    return RetrievalMetrics(
        precision=precision,
        recall=recall,
        max_precision=max_precision,
        max_recall=max_recall,
        hits=hits,
        mrr=mrr,
        ap=ap,
    )


def _compute_average_precision(match_positions: list[int], n_ground_truth: int) -> float:
    if not match_positions or n_ground_truth == 0:
        return 0.0

    precision_sum = sum(i / pos for i, pos in enumerate(match_positions, start=1))
    return precision_sum / n_ground_truth
