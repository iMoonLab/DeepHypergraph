from typing import Union, Dict, List

import torch
from ..classification import VertexClassificationEvaluator


class GraphVertexClassificationEvaluator(VertexClassificationEvaluator):
    r"""Return the metric evaluator for vertex classification task on the graph structure. The supported metrics includes: ``accuracy``, ``f1_score``, ``confusion_matrix``.
    
    Args:
        ``metric_configs`` (``List[Union[str, Dict[str, dict]]]``): The metric configurations. The key is the metric name and the value is the metric parameters.
        ``validate_index`` (``int``): The specified metric index used for validation. Defaults to ``0``.
    
    Examples:
        >>> import torch
        >>> import dhg.metrics as dm
        >>> evaluator = dm.GraphVertexClassificationEvaluator(
                [
                    "accuracy", 
                    {"f1_score": {"average": "macro"}},
                ], 
                0
            )
        >>> y_true = torch.tensor([0, 0, 1, 1, 2, 2])
        >>> y_pred = torch.tensor([0, 2, 1, 2, 1, 2])
        >>> evaluator.validate(y_true, y_pred)
        0.5
        >>> evaluator.test(y_true, y_pred)
        {
            'accuracy': 0.5, 
            'f1_score -> average@macro': 0.5222222222222221
        }
    """

    def __init__(
        self, metric_configs: List[Union[str, Dict[str, dict]]], validate_index: int = 0
    ):
        super().__init__(metric_configs, validate_index)

    def validate(self, y_true: torch.LongTensor, y_pred: torch.Tensor):
        return super().validate(y_true, y_pred)

    def test(self, y_true: torch.LongTensor, y_pred: torch.Tensor):
        return super().test(y_true, y_pred)
