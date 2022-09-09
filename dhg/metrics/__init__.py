from .base import BaseEvaluator
from .classification import (
    available_classification_metrics,
    VertexClassificationEvaluator,
)
from .retrieval import available_retrieval_metrics, RetrievalEvaluator
from .recommender import available_recommender_metrics, UserItemRecommenderEvaluator
from .graphs import GraphVertexClassificationEvaluator
from .hypergraphs import HypergraphVertexClassificationEvaluator

from typing import List, Union, Dict
from dhg._global import AUTHOR_EMAIL


def build_evaluator(
    task: str,
    metric_configs: List[Union[str, Dict[str, dict]]],
    validate_index: int = 0,
):
    r"""Return the metric evaluator for the given task.
    
    Args:
        ``task`` (``str``): The type of the task. The supported types include: ``graph_vertex_classification``, ``hypergraph_vertex_classification``, and ``user_item_recommender``.
        ``metric_configs`` (``List[Union[str, Dict[str, dict]]]``): The list of metric names.
        ``validate_index`` (``int``): The specified metric index used for validation. Defaults to ``0``.
    """
    if task == "graph_vertex_classification":
        return GraphVertexClassificationEvaluator(metric_configs, validate_index)
    elif task == "hypergraph_vertex_classification":
        return HypergraphVertexClassificationEvaluator(metric_configs, validate_index)
    elif task == "user_item_recommender":
        return UserItemRecommenderEvaluator(metric_configs, validate_index)
    else:
        raise ValueError(
            f"{task} is not supported yet. Please email '{AUTHOR_EMAIL}' to add it."
        )


__all__ = [
    "BaseEvaluator",
    "build_evaluator",
    "available_classification_metrics",
    "available_retrieval_metrics",
    "available_recommender_metrics",
    "VertexClassificationEvaluator",
    "GraphVertexClassificationEvaluator",
    "HypergraphVertexClassificationEvaluator",
    "UserItemRecommenderEvaluator",
    "RetrievalEvaluator",
]
