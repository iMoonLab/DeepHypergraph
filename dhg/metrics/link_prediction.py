from typing import Union, List, Dict

import torch
import sklearn.metrics as sm

from .base import BaseEvaluator

__all__ = [
    "available_link_prediction_metrics",
    "accuracy",
    "f1_score",
    "confusion_matrix",
    "auc"
    "LinkPredictionEvaluator",
]


def available_link_prediction_metrics():
    r"""Return available metrics for the link prediction task. 
    
    The available metrics are: ``accuracy``, ``f1_score``, ``confusion_matrix``, ``auc``.
    """
    return ("accuracy", "f1_score", "confusion_matrix", "auc")


def _format_inputs(y_true: torch.LongTensor, y_pred: torch.Tensor):
    r"""Format the inputs.

    Args:
        ``y_true`` (``torch.LongTensor``): The ground truth labels. Size :math:`(N_{samples}, )`.
        ``y_pred`` (``torch.Tensor``): The predicted labels. Size :math:`(N_{samples}, N_{class})` or :math:`(N_{samples}, )`.
    """
    assert y_true.dim() == 1, "y_true must be 1D torch.LongTensor."
    assert y_pred.dim() in (1, 2), "y_pred must be 1D or 2D torch.Tensor."
    y_true = y_true.cpu().detach()
    if y_pred.dim() == 2:
        y_pred = y_pred.argmax(dim=1)
    y_pred = y_pred.cpu().detach()
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same length."
    return (y_true, y_pred)


def accuracy(y_true: torch.LongTensor, y_pred: torch.Tensor):
    r"""Calculate the accuracy score for the link prediction task.

    .. math::
        \text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathcal{I}(y_i, \hat{y}_i),
    
    where :math:`\mathcal{I}(\cdot, \cdot)` is the indicator function, which is 1 if the two inputs are equal, and 0 otherwise.
    :math:`y_i` and :math:`\hat{y}_i` are the ground truth and predicted labels for the i-th sample.

    Args:
        ``y_true`` (``torch.LongTensor``): The ground truth labels. Size :math:`(N_{samples}, )`.
        ``y_pred`` (``torch.Tensor``): The predicted labels. Size :math:`(N_{samples}, N_{class})` or :math:`(N_{samples}, )`.

    Examples:
        >>> import torch
        >>> import dhg.metrics as dm
        >>> y_true = torch.tensor([3, 2, 4])
        >>> y_pred = torch.tensor([
                [0.2, 0.3, 0.5, 0.4, 0.3],
                [0.8, 0.2, 0.3, 0.5, 0.4],
                [0.2, 0.4, 0.5, 0.2, 0.8],
            ])
        >>> dm.link_prediction.accuracy(y_true, y_pred)
        0.3333333432674408
    """
    y_pred = torch.where(y_pred >= y_pred.mean(), 1, 0)
    y_true, y_pred = _format_inputs(y_true, y_pred)
    return sm.accuracy_score(y_true, y_pred)


def f1_score(y_true: torch.LongTensor, y_pred: torch.Tensor, average: str = "macro"):
    r"""Calculate the F1 score for the link prediction task.

    Args:
        ``y_true`` (``torch.LongTensor``): The ground truth labels. Size :math:`(N_{samples}, )`.
        ``y_pred`` (``torch.Tensor``): The predicted labels. Size :math:`(N_{samples}, N_{class})` or :math:`(N_{samples}, )`.
        ``average`` (``str``): The average method. Must be one of "macro", "micro", "weighted".

    Examples:
        >>> import torch
        >>> import dhg.metrics as dm
        >>> y_true = torch.tensor([3, 2, 4, 0])
        >>> y_pred = torch.tensor([
                [0.2, 0.3, 0.5, 0.4, 0.3],
                [0.8, 0.2, 0.3, 0.5, 0.4],
                [0.2, 0.4, 0.5, 0.2, 0.8],
                [0.8, 0.4, 0.5, 0.2, 0.8]
            ])
        >>> dm.link_prediction.f1_score(y_true, y_pred, "macro")
        0.41666666666666663
        >>> dm.link_prediction.f1_score(y_true, y_pred, "micro")
        0.5
        >>> dm.link_prediction.f1_score(y_true, y_pred, "weighted")
        0.41666666666666663
    """
    y_pred = torch.where(y_pred >= y_pred.mean(), 1, 0)
    y_true, y_pred = _format_inputs(y_true, y_pred)
    return sm.f1_score(y_true, y_pred, average=average)


def confusion_matrix(y_true: torch.LongTensor, y_pred: torch.Tensor):
    r"""Calculate the confusion matrix for the link prediction task.

    Args:
        ``y_true`` (``torch.LongTensor``): The ground truth labels. Size :math:`(N_{samples}, )`.
        ``y_pred`` (``torch.Tensor``): The predicted labels. Size :math:`(N_{samples}, )`.

    Examples:
        >>> import torch
        >>> import dhg.metrics as dm
        >>> y_true = torch.tensor([3, 2, 4, 0])
        >>> y_pred = torch.tensor([
                [0.2, 0.3, 0.5, 0.4, 0.3],
                [0.8, 0.2, 0.3, 0.5, 0.4],
                [0.2, 0.4, 0.5, 0.2, 0.8],
                [0.8, 0.4, 0.5, 0.2, 0.8]
            ])
        >>> dm.link_prediction.confusion_matrix(y_true, y_pred)
        array([[1, 0, 0, 0],
               [1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 1]])
    """
    y_true, y_pred = _format_inputs(y_true, y_pred)
    return sm.confusion_matrix(y_true, y_pred)

def auc(y_true: torch.LongTensor, y_pred: torch.Tensor):
    r"""Calculate the auc for the link prediction task.

    Args:
        ``y_true`` (``torch.LongTensor``): The ground truth labels. Size :math:`(N_{samples}, )`.
        ``y_pred`` (``torch.Tensor``): The predicted labels. Size :math:`(N_{samples}, N_{class})` or :math:`(N_{samples}, )`.

    Examples:
        >>> import torch
        >>> import dhg.metrics as dm
        >>> y_pred = torch.tensor([0.2, 0.3, 0.5, 0.4, 0.3])
        >>> y_true = torch.tensor([1, 0, 1, 1, 0])
        >>> y_pred = torch.tensor([0.2, 0.1, 0.5, 0.8, 0.3])
        >>> dm.link_prediction.auc(y_true, y_pred)
        0.8333333333333333
    """
    y_true, y_pred = _format_inputs(y_true, y_pred)
    return sm.roc_auc_score(y_true, y_pred)


# Link Prediction Evaluator
class LinkPredictionEvaluator(BaseEvaluator):
    r"""Return the metric evaluator for link prediction task. The supported metrics includes: ``accuracy``, ``f1_score``, ``confusion_matrix``, ``auc``.
    
    Args:
        ``metric_configs`` (``List[Union[str, Dict[str, dict]]]``): The metric configurations. The key is the metric name and the value is the metric parameters.
        ``validate_index`` (``int``): The specified metric index used for validation. Defaults to ``0``.
    """

    def __init__(
        self,
        metric_configs: List[Union[str, Dict[str, dict]]],
        validate_index: int = 0,
    ):
        super().__init__("link_prediction", metric_configs, validate_index)

    def validate(self, y_true: torch.LongTensor, y_pred: torch.Tensor):
        r"""Return the result of the evaluation on the specified ``validate_index``-th metric.

        Args:
            ``y_true`` (``torch.LongTensor``): The ground truth labels. Size :math:`(N_{samples}, )`.
            ``y_pred`` (``torch.Tensor``): The predicted labels. Size :math:`(N_{samples}, N_{class})` or :math:`(N_{samples}, )`.
        """
        return super().validate(y_true, y_pred)

    def test(self, y_true: torch.LongTensor, y_pred: torch.Tensor):
        r"""Return results of the evaluation on all the metrics in ``metric_configs``.

        Args:
            ``y_true`` (``torch.LongTensor``): The ground truth labels. Size :math:`(N_{samples}, )`.
            ``y_pred`` (``torch.Tensor``): The predicted labels. Size :math:`(N_{samples}, N_{class})` or :math:`(N_{samples}, )`.
        """
        return super().test(y_true, y_pred)
