from .base import BaseTask
from .vertex_classification import VertexClassificationTask
from .graphs import GraphVertexClassificationTask
from .hypergraphs import HypergraphVertexClassificationTask
from .recommender import UserItemRecommenderTask

__all__ = [
    "BaseTask",
    "VertexClassificationTask",
    "GraphVertexClassificationTask",
    "HypergraphVertexClassificationTask",
    "UserItemRecommenderTask",
]
