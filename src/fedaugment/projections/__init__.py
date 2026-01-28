"""This module contains approaches for training and evaluating the FedAugment projection models."""

from fedaugment.projections.main import (
    PerfMetrics,
    load_projection_model,
    project_embedding_collection,
    train_projection_model,
)

__all__ = [
    "PerfMetrics",
    "load_projection_model",
    "project_embedding_collection",
    "train_projection_model",
]
