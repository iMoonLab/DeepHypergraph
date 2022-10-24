from typing import Optional, Union, Tuple, List, Dict

import torch


from .base import BaseEvaluator

__all__ = [
    "available_recommender_metrics",
    "precision",
    "recall",
    "ndcg",
]


def available_recommender_metrics():
    r"""Return available metrics for the recommender task.

    The available metrics are: ``precision``, ``recall``, and ``ndcg``.
    """
    return ("precision", "recall", "ndcg")


def _format_inputs(
    y_true: torch.Tensor, y_pred: torch.Tensor, k: Optional[int] = None, ratio: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    r"""Format the inputs
    
    Args:
        ``y_true`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``y_pred`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``k`` (``int``, optional): The specified top-k value. Default to :math:`N_{target}`.
        ``ratio`` (``float``, optional): The specified ratio of top-k value. If ``ratio`` is not ``None``, ``k`` will be ignored. Defaults to ``None``.
    """
    assert (
        y_true.shape == y_pred.shape
    ), "The shape of y_true and y_pred must be the same."
    assert y_true.dim() in (1, 2), "The input y_true must be 1-D or 2-D."
    assert y_pred.dim() in (1, 2), "The input y_pred must be 1-D or 2-D."
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(0)
    if y_pred.dim() == 1:
        y_pred = y_pred.unsqueeze(0)
    y_true, y_pred = y_true.detach().float(), y_pred.detach().float()
    max_k = y_true.shape[1]
    if ratio is not None:
        k = int(np.ceil(max_k * ratio))
    else:
        k = min(k, max_k) if k is not None else max_k
    return y_true, y_pred, k


def precision(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    k: Optional[int] = None,
    ratio: Optional[float] = None,
    ret_batch: bool = False,
) -> Union[float, list]:
    r"""Calculate the Precision score for the recommender task.

    Args:
        ``y_true`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``y_pred`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``k`` (``int``, optional): The specified top-k value. Defaults to :math:`N_{target}`.
        ``ratio`` (``float``, optional): The specified ratio of top-k value. If ``ratio`` is not ``None``, ``k`` will be ignored. Defaults to ``None``.
        ``ret_batch`` (``bool``): Whether to return the raw score list. Defaults to ``False``.
    
    Examples:
        >>> import torch
        >>> import dhg.metrics as dm
        >>> y_true = torch.tensor([0, 1, 0, 0, 1, 1])
        >>> y_pred = torch.tensor([0.8, 0.9, 0.6, 0.7, 0.4, 0.5])
        >>> dm.recommender.precision(y_true, y_pred, k=2)
        0.5
    """
    y_true, y_pred, k = _format_inputs(y_true, y_pred, k, ratio=ratio)
    assert y_true.max() == 1, "The input y_true must be binary."
    pred_seq = y_true.gather(1, torch.argsort(y_pred, dim=-1, descending=True))[:, :k]
    res_list = (pred_seq.sum(dim=1) / k).detach().cpu()
    if ret_batch:
        return [res.item() for res in res_list]
    else:
        return res_list.mean().item()


def recall(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    k: Optional[int] = None,
    ratio: Optional[float] = None,
    ret_batch: bool = False,
) -> Union[float, list]:
    r"""Calculate the Recall score for the recommender task.

    Args:
        ``y_true`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``y_pred`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``k`` (``int``, optional): The specified top-k value. Defaults to :math:`N_{target}`.
        ``ratio`` (``float``, optional): The specified ratio of top-k value. If ``ratio`` is not ``None``, ``k`` will be ignored. Defaults to ``None``.
        ``ret_batch`` (``bool``): Whether to return the raw score list. Defaults to ``False``.
    
    Examples:
        >>> import torch
        >>> import dhg.metrics as dm
        >>> y_true = torch.tensor([0, 1, 0, 0, 1, 1])
        >>> y_pred = torch.tensor([0.8, 0.9, 0.6, 0.7, 0.4, 0.5])
        >>> dm.recommender.recall(y_true, y_pred, k=5)
        0.6666666666666666
    """
    y_true, y_pred, k = _format_inputs(y_true, y_pred, k, ratio=ratio)
    assert y_true.max() == 1, "The input y_true must be binary."
    pred_seq = y_true.gather(1, torch.argsort(y_pred, dim=-1, descending=True))[:, :k]
    num_true = y_true.sum(dim=1)
    res_list = (pred_seq.sum(dim=1) / num_true).cpu()
    res_list[torch.isinf(res_list)] = 0
    res_list[torch.isnan(res_list)] = 0
    if ret_batch:
        return [res.item() for res in res_list]
    else:
        return res_list.mean().item()


def _dcg(matrix: torch.Tensor) -> torch.Tensor:
    r"""Calculate the Discounted Cumulative Gain (DCG).
    
    Args:
        ``sequence`` (``torch.Tensor``): A 2-D tensor. Size :math:`(N, K)`
    """
    assert matrix.dim() == 2, "The input must be a 2-D tensor."
    n, k = matrix.shape
    denom = (
        torch.log2(torch.arange(k, device=matrix.device) + 2.0).view(1, -1).repeat(n, 1)
    )
    return (matrix / denom).sum(dim=-1)


def ndcg(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    k: Optional[int] = None,
    ratio: Optional[float] = None,
    ret_batch: bool = False,
) -> Union[float, list]:
    r"""Calculate the Normalized Discounted Cumulative Gain (NDCG) for the recommender task.

    Args:
        ``y_true`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``y_pred`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``k`` (``int``, optional): The specified top-k value. Default to :math:`N_{target}`.
        ``ratio`` (``float``, optional): The specified ratio of top-k value. If ``ratio`` is not ``None``, ``k`` will be ignored. Defaults to ``None``.
        ``ret_batch`` (``bool``): Whether to return the raw score list. Defaults to ``False``.
    
    Examples:
        >>> import torch
        >>> import dhg.metrics as dm
        >>> y_true = torch.tensor([10, 0, 0, 1, 5])
        >>> y_pred = torch.tensor([.1, .2, .3, 4, 70])
        >>> dm.recommender.ndcg(y_true, y_pred)
        0.695694088935852
        >>> dm.recommender.ndcg(y_true, y_pred, k=3)
        0.4123818874359131
    """
    y_true, y_pred, k = _format_inputs(y_true, y_pred, k, ratio=ratio)

    pred_seq = y_true.gather(1, torch.argsort(y_pred, dim=-1, descending=True))[:, :k]
    ideal_seq = torch.sort(y_true, dim=-1, descending=True)[0][:, :k]

    pred_dcg = _dcg(pred_seq)
    ideal_dcg = _dcg(ideal_seq)

    res_list = (pred_dcg / ideal_dcg).detach().cpu()
    res_list[torch.isinf(res_list)] = 0
    res_list[torch.isnan(res_list)] = 0
    if ret_batch:
        return [res.item() for res in res_list]
    else:
        return res_list.mean().item()


class UserItemRecommenderEvaluator(BaseEvaluator):
    r"""Return the metric evaluator for recommender task on user-item bipartite graph. The supported metrics includes: ``precision``, ``recall``, ``ndcg``.
    
    Args:
        ``metric_configs`` (``List[Union[str, Dict[str, dict]]]``): The metric configurations. The key is the metric name and the value is the metric parameters.
        ``validate_index`` (``int``): The specified metric index used for validation. Defaults to ``0``.

    Examples:
        >>> import torch
        >>> import dhg.metrics as dm
        >>> evaluator = dm.UserItemRecommenderEvaluator(
                [
                    {"ndcg": {"k": 2}},
                    {"recall": {"k": 4}},
                    {"precision": {"k": 2}},
                    "precision",
                    {"precision": {"k": 6}},
                ],
                0,
            )
        >>> y_true = torch.tensor([
                [0, 1, 0, 0, 1, 1], 
                [0, 0, 1, 0, 1, 0], 
                [0, 1, 1, 1, 0, 1],
            ])
        >>> y_pred = torch.tensor([
                [0.8, 0.9, 0.6, 0.7, 0.4, 0.5], 
                [0.2, 0.6, 0.3, 0.3, 0.4, 0.6], 
                [0.7, 0.4, 0.3, 0.2, 0.8, 0.4],
            ])
        >>> evaluator.validate_add_batch(y_true, y_pred)
        >>> y_true = torch.tensor([
                [0, 1, 0, 1, 0, 1], 
                [1, 1, 0, 0, 1, 0], 
                [1, 0, 1, 0, 0, 1],
            ])
        >>> y_pred = torch.tensor([
                [0.8, 0.9, 0.9, 0.4, 0.4, 0.5], 
                [0.2, 0.6, 0.3, 0.3, 0.4, 0.6], 
                [0.7, 0.4, 0.3, 0.2, 0.8, 0.4],
            ])
        >>> evaluator.validate_add_batch(y_true, y_pred)
        >>> evaluator.validate_epoch_res()
        0.37104907135168713
        >>> y_true = torch.tensor([
                [0, 1, 0, 0, 1, 1], 
                [0, 0, 1, 0, 1, 0], 
                [0, 1, 1, 1, 0, 1],
            ])
        >>> y_pred = torch.tensor([
                [0.8, 0.9, 0.6, 0.7, 0.4, 0.5], 
                [0.2, 0.6, 0.3, 0.3, 0.4, 0.6], 
                [0.7, 0.4, 0.3, 0.2, 0.8, 0.4],
            ])
        >>> evaluator.test_add_batch(y_true, y_pred)
        >>> y_true = torch.tensor([
                [0, 1, 0, 1, 0, 1], 
                [1, 1, 0, 0, 1, 0], 
                [1, 0, 1, 0, 0, 1],
            ])
        >>> y_pred = torch.tensor([
                [0.8, 0.9, 0.9, 0.4, 0.4, 0.5], 
                [0.2, 0.6, 0.3, 0.3, 0.4, 0.6], 
                [0.7, 0.4, 0.3, 0.2, 0.8, 0.4],
            ])
        >>> evaluator.test_add_batch(y_true, y_pred)
        >>> evaluator.test_epoch_res()
        {
            'ndcg -> k@2': 0.37104907135168713, 
            'recall -> k@4': 0.638888900478681, 
            'precision -> k@2': 0.3333333333333333, 
            'precision': 0.5000000049670538, 
            'precision -> k@6': 0.5000000049670538
        }
    """

    def __init__(
        self,
        metric_configs: List[Union[str, Dict[str, dict]]],
        validate_index: int = 0,
    ):
        super().__init__("recommender", metric_configs, validate_index)

    def validate_add_batch(
        self, batch_y_true: torch.Tensor, batch_y_pred: torch.Tensor
    ):
        r"""Add batch data for validation.

        Args:
            ``batch_y_true`` (``torch.Tensor``): The ground truth data. Size :math:`(N_{batch}, -)`.
            ``batch_y_pred`` (``torch.Tensor``): The predicted data. Size :math:`(N_{batch}, -)`.
        """
        return super().validate_add_batch(batch_y_true, batch_y_pred)

    def validate_epoch_res(self):
        r"""For all added batch data, return the result of the evaluation on the specified ``validate_index``-th metric.
        """
        return super().validate_epoch_res()

    def test_add_batch(self, batch_y_true: torch.Tensor, batch_y_pred: torch.Tensor):
        r"""Add batch data for testing.

        Args:
            ``batch_y_true`` (``torch.Tensor``): The ground truth data. Size :math:`(N_{batch}, -)`.
            ``batch_y_pred`` (``torch.Tensor``): The predicted data. Size :math:`(N_{batch}, -)`.
        """
        return super().test_add_batch(batch_y_true, batch_y_pred)

    def test_epoch_res(self):
        r"""For all added batch data, return results of the evaluation on all the metrics in ``metric_configs``.
        """
        return super().test_epoch_res()
