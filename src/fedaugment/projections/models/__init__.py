"""This module contains different strategies to train projection models for embeddings."""

from .contrastive_learning import ContrastiveLearningModel
from .local_isometry import LocalIsometryModel
from .naive import NaiveModel
from .procrustes import ProcrustesModel
from .projection_model import ProjectionModel
from .vec2vec import Vec2VecModel

__all__ = [
    "ContrastiveLearningModel",
    "LocalIsometryModel",
    "NaiveModel",
    "ProcrustesModel",
    "ProjectionModel",
    "Vec2VecModel",
]
