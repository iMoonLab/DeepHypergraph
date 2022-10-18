from typing import Optional, Union, Tuple, List, Dict

import torch
import numpy as np

from dhg.metrics.base import BaseEvaluator


def available_retrieval_metrics():
    r"""Return available metrics for the retrieval task.
    
    The available metrics are: ``precision``, ``recall``, ``map``, ``ndcg``, ``mrr``, ``pr_curve``.
    """
    return ("precision", "recall", "map", "ndcg", "mrr", "pr_curve")


def _format_inputs(
    y_true: torch.Tensor, y_pred: torch.Tensor, k: Optional[int] = None, ratio: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    r"""Format the inputs
    
    Args:
        ``y_true`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``y_pred`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``k`` (``int``, optional): The specified top-k value. Defaults to :math:`N_{target}`.
        ``ratio`` (``float``, optional): The specified ratio of top-k value. If ``ratio`` is not ``None``, ``k`` will be ignored. Defaults to ``None``.
    """
    assert y_true.shape == y_pred.shape, "The shape of y_true and y_pred must be the same."
    assert y_true.dim() in (1, 2), "The input y_true must be 1-D or 2-D."
    assert y_pred.dim() in (1, 2), "The input y_pred must be 1-D or 2-D."
    assert ratio is None or (ratio > 0 and ratio <= 1), "The ratio must be in (0, 1]."
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
    r"""Calculate the Precision score for the retrieval task.

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
        >>> dm.retrieval.precision(y_true, y_pred, k=2)
        0.5
    """
    y_true, y_pred, k = _format_inputs(y_true, y_pred, k, ratio=ratio)
    assert y_true.max() == 1, "The input y_true must be binary."
    pred_seq = y_true.gather(1, torch.argsort(y_pred, dim=-1, descending=True))[:, :k]
    res_list = (pred_seq.sum(dim=1) / k).detach().cpu()
    if ret_batch:
        return res_list
    else:
        return res_list.mean().item()


def recall(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    k: Optional[int] = None,
    ratio: Optional[float] = None,
    ret_batch: bool = False,
) -> Union[float, list]:
    r"""Calculate the Recall score for the retrieval task.

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
        >>> dm.retrieval.recall(y_true, y_pred, k=5)
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
        return res_list
    else:
        return res_list.mean().item()


def ap(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    k: Optional[int] = None,
    ratio: Optional[float] = None,
    method: str = "pascal_voc",
) -> Union[float, list]:
    r"""Calculate the Average Precision (AP) for the retrieval task.

    Args:
        ``y_true`` (``torch.Tensor``): A 1-D tensor. Size :math:`(N_{target},)`.
        ``y_pred`` (``torch.Tensor``): A 1-D tensor. Size :math:`(N_{target},)`.
        ``k`` (``int``, optional): The specified top-k value. Defaults to :math:`N_{target}`.
        ``ratio`` (``float``, optional): The specified ratio of top-k value. If ``ratio`` is not ``None``, ``k`` will be ignored. Defaults to ``None``.
        ``method`` (``str``): The method to compute the AP can be ``legacy`` or ``pascal_voc``. Defaults to ``pascal_voc``.
    
    Examples:
        >>> import torch
        >>> import dhg.metrics as dm
        >>> y_true = torch.tensor([True, False, True])
        >>> y_pred = torch.tensor([0.2, 0.3, 0.5])
        >>> dm.retrieval.ap(y_true, y_pred, method="legacy")
        0.8333333730697632
    """
    assert ratio is None or (ratio > 0 and ratio <= 1), "The ratio must be in (0, 1]."
    assert method in ("legacy", "pascal_voc",), "The method must be either legacy or pascal_voc."
    assert y_true.shape == y_pred.shape, "The shape of y_true and y_pred must be the same."
    assert y_true.dim() == 1, "The input y_true must be 1-D."
    assert y_pred.dim() == 1, "The input y_pred must be 1-D."
    y_true, y_pred = y_true.detach().float(), y_pred.detach().float()
    max_k = y_true.shape[0]
    if ratio is not None:
        k = int(np.ceil(max_k * ratio))
    else:
        k = min(k, max_k) if k is not None else max_k

    pred_seq = y_true[torch.argsort(y_pred, descending=True)]
    pred_index = torch.arange(1, len(y_true) + 1, device=y_true.device)[pred_seq > 0]
    recall_seq = torch.arange(1, len(pred_index) + 1, device=y_true.device)
    res = recall_seq / pred_index
    if method == "pascal_voc":
        res = torch.flip(res, dims=(0,))
        res = torch.cummax(res, dim=0)[0]
    return res.detach().cpu().mean().item()


def map(
    y_true: torch.LongTensor,
    y_pred: torch.LongTensor,
    k: Optional[int] = None,
    ratio: Optional[float] = None,
    method: str = "pascal_voc",
    ret_batch: bool = False,
) -> Union[float, list]:
    r"""Calculate the mean Average Precision (mAP) for the retrieval task.

    Args:
        ``y_true`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``y_pred`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``k`` (``int``, optional): The specified top-k value. Defaults to :math:`N_{target}`.
        ``ratio`` (``float``, optional): The specified ratio of top-k value. If ``ratio`` is not ``None``, ``k`` will be ignored. Defaults to ``None``.
        ``method`` (``str``): The specified method: ``legacy`` or ``pascal_voc``. Defaults to ``pascal_voc``.
        ``ret_batch`` (``bool``): Whether to return the raw score list. Defaults to ``False``.
    
    Examples:
        >>> import torch
        >>> import dhg.metrics as dm
        >>> y_true = torch.tensor([
                [True, False, True, False, True],
                [False, False, False, True, True],
                [True, True, False, True, False],
                [False, True, True, False, True],
            ])
        >>> y_pred = torch.tensor([
                [0.2, 0.8, 0.5, 0.4, 0.3],
                [0.8, 0.2, 0.3, 0.9, 0.4],
                [0.2, 0.4, 0.5, 0.9, 0.8],
                [0.8, 0.2, 0.9, 0.3, 0.7],
            ])
        >>> dm.retrieval.map(y_true, y_pred, k=2, method="legacy")
        0.7055555880069733
        >>> dm.retrieval.map(y_true, y_pred, k=2, method="pascal_voc")
        0.7305555790662766
    """
    assert method in ("legacy", "pascal_voc",), "The method must be either legacy or pascal_voc."
    y_true, y_pred, k = _format_inputs(y_true, y_pred, k, ratio=ratio)
    res_list = [ap(y_true[i, :], y_pred[i, :], k, method=method) for i in range(y_true.shape[0])]
    if ret_batch:
        return res_list
    else:
        return np.mean(res_list)


def _dcg(matrix: torch.Tensor) -> torch.Tensor:
    r"""Calculate the Discounted Cumulative Gain (DCG).
    
    Args:
        ``sequence`` (``torch.Tensor``): A 2-D tensor. Size :math:`(N, K)`
    """
    assert matrix.dim() == 2, "The input must be a 2-D tensor."
    n, k = matrix.shape
    denom = torch.log2(torch.arange(k, device=matrix.device) + 2.0).view(1, -1).repeat(n, 1)
    return (matrix / denom).detach().cpu().sum(dim=-1)


def ndcg(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    k: Optional[int] = None,
    ratio: Optional[float] = None,
    ret_batch: bool = False,
) -> Union[float, list]:
    r"""Calculate the Normalized Discounted Cumulative Gain (NDCG) for the retrieval task.

    Args:
        ``y_true`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``y_pred`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``k`` (``int``, optional): The specified top-k value. Defaults to :math:`N_{target}`.
        ``ratio`` (``float``, optional): The specified ratio of top-k value. If ``ratio`` is not ``None``, ``k`` will be ignored. Defaults to ``None``.
        ``ret_batch`` (``bool``): Whether to return the raw score list. Defaults to ``False``.

    Examples:
        >>> import torch
        >>> import dhg.metrics as dm
        >>> y_true = torch.tensor([10, 0, 0, 1, 5])
        >>> y_pred = torch.tensor([.1, .2, .3, 4, 70])
        >>> dm.retrieval.ndcg(y_true, y_pred)
        0.695694088935852
        >>> dm.retrieval.ndcg(y_true, y_pred, k=3)
        0.4123818874359131
    """
    y_true, y_pred, k = _format_inputs(y_true, y_pred, k, ratio=ratio)

    pred_seq = y_true.gather(1, torch.argsort(y_pred, dim=-1, descending=True))[:, :k]
    ideal_seq = torch.sort(y_true, dim=-1, descending=True)[0][:, :k]

    pred_dcg = _dcg(pred_seq)
    ideal_dcg = _dcg(ideal_seq)

    res_list = pred_dcg / ideal_dcg
    res_list[torch.isinf(res_list)] = 0
    res_list[torch.isnan(res_list)] = 0
    if ret_batch:
        return res_list
    else:
        return res_list.mean().item()


def rr(y_true: torch.Tensor, y_pred: torch.Tensor, k: Optional[int] = None, ratio: Optional[float] = None,) -> float:
    r"""Calculate the Reciprocal Rank (RR) for the retrieval task.

    Args:
        ``y_true`` (``torch.Tensor``): A 1-D tensor. Size :math:`(N_{target},)``.
        ``y_pred`` (``torch.Tensor``): A 1-D tensor. Size :math:`(N_{target},)`.
        ``k`` (``int``, optional): The specified top-k value. Defaults to :math:`N_{target}`.
        ``ratio`` (``float``, optional): The specified ratio of top-k value. If ``ratio`` is not ``None``, ``k`` will be ignored. Defaults to ``None``.
    
    Examples:
        >>> import torch
        >>> import dhg.metrics as dm
        >>> y_true = torch.tensor([False, True, False, True])
        >>> y_pred = torch.tensor([0.2, 0.3, 0.5, 0.2])
        >>> dm.retrieval.rr(y_true, y_pred)
        0.375
        >>> dm.retrieval.rr(y_true, y_pred, k=2)
        0.5
    """
    assert y_true.shape == y_pred.shape
    assert y_true.dim() == 1, "The input y_true must be a 1-D tensor."
    assert y_pred.dim() == 1, "The input y_pred must be a 1-D tensor."
    y_true, y_pred = y_true.detach().float(), y_pred.detach().float()
    max_k = y_true.shape[0]
    if ratio is not None:
        k = int(np.ceil(max_k * ratio))
    else:
        k = min(k, max_k) if k is not None else max_k

    pred_seq = y_true[torch.argsort(y_pred, dim=-1, descending=True)][:k]
    pred_index = torch.nonzero(pred_seq).view(-1)
    res = (1 / (pred_index + 1)).mean().cpu()
    return res.mean().item()


def mrr(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    k: Optional[int] = None,
    ratio: Optional[float] = None,
    ret_batch: bool = False,
) -> Union[float, list]:
    r"""Calculate the mean Reciprocal Rank (MRR) for the retrieval task.

    Args:
        ``y_true`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``y_pred`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``k`` (``int``, optional): The specified top-k value. Defaults to :math:`N_{target}`.
        ``ratio`` (``float``, optional): The specified ratio of top-k value. If ``ratio`` is not ``None``, ``k`` will be ignored. Defaults to ``None``.
        ``ret_batch`` (``bool``): Whether to return the raw score list. Defaults to ``False``.

    Examples:
        >>> import torch
        >>> import dhg.metrics as dm
        >>> y_true = torch.tensor([False, True, False, True])
        >>> y_pred = torch.tensor([0.2, 0.3, 0.5, 0.2])
        >>> dm.retrieval.mrr(y_true, y_pred)
        0.375
        >>> dm.retrieval.mrr(y_true, y_pred, k=2)
        0.5
    """
    y_true, y_pred, k = _format_inputs(y_true, y_pred, k, ratio=ratio)
    res_list = [rr(y_true[i, :], y_pred[i, :], k) for i in range(y_true.shape[0])]
    if ret_batch:
        return res_list
    else:
        return np.mean(res_list)


def _pr_curve(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    k: Optional[int] = None,
    ratio: Optional[float] = None,
    method: str = "pascal_voc",
    n_points: int = 11,
) -> tuple:
    r"""Calculate the Precision-Recall Curve for the retrieval task.

    Args:
        ``y_true`` (``torch.Tensor``): A 1-D tensor. Size :math:`(N_{target},)`.
        ``y_pred`` (``torch.Tensor``): A 1-D tensor. Size :math:`(N_{target},)`.
        ``k`` (``int``, optional): The specified top-k value. Defaults to :math:`N_{target}`.
        ``ratio`` (``float``, optional): The specified ratio of top-k value. If ``ratio`` is not ``None``, ``k`` will be ignored. Defaults to ``None``.
        ``method`` (``str``, optional): The method to compute the PR curve can be "legacy" or "pascal_voc". Defaults to "pascal_voc".
        ``n_points`` (``int``): The number of points to compute the PR curve. Defaults to ``11``.

    Examples:
        >>> import torch
        >>> import dhg.metrics as dm
        >>> y_true = torch.tensor([0, 1, 0, 1, 0, 0, 1, 0, 1, 0])
        >>> y_pred = torch.tensor([0.23, 0.76, 0.01, 0.91, 0.13, 0.45, 0.12, 0.03, 0.38, 0.11])
        >>> precision_coor, recall_coor = dm.retrieval.pr_curve(y_true, y_pred)
        >>> precision_coor
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.75, 0.75, 0.75, 0.5714285969734192]
        >>> recall_coor
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    """
    assert method in ("legacy", "pascal_voc",), "The method must be either legacy or pascal_voc."
    assert y_true.shape == y_pred.shape, "The shape of y_true and y_pred must be the same."
    assert y_true.dim() == 1, "The input y_true must be 1-D."
    assert y_pred.dim() == 1, "The input y_pred must be 1-D."
    y_true, y_pred = y_true.detach().float(), y_pred.detach().float()
    max_k = y_true.shape[0]
    if ratio is not None:
        k = int(np.ceil(max_k * ratio))
    else:
        k = min(k, max_k) if k is not None else max_k

    pred_seq = y_true[torch.argsort(y_pred, descending=True)]
    pred_index = torch.arange(1, len(y_true) + 1, device=y_true.device)[pred_seq > 0]
    recall_seq = torch.arange(1, len(pred_index) + 1, device=y_true.device)
    res = recall_seq / pred_index
    if method == "pascal_voc":
        res = torch.flip(res, dims=(0,))
        res = torch.cummax(res, dim=0)[0]
        res = torch.flip(res, dims=(0,))
    res = res.cpu().numpy()
    recall_coor = np.linspace(0, 1, n_points)
    recall_index = (recall_coor * (torch.sum(y_true).item() - 1)).astype(int)
    precision_coor = res[recall_index]
    return precision_coor.tolist(), recall_coor.tolist()


def pr_curve(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    k: Optional[int] = None,
    ratio: Optional[float] = None,
    method: str = "pascal_voc",
    n_points: int = 11,
    ret_batch: bool = False,
) -> tuple:
    r"""Calculate the Precision-Recall Curve for the retrieval task.

    Args:
        ``y_true`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``y_pred`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``k`` (``int``, optional): The specified top-k value. Defaults to :math:`N_{target}`.
        ``ratio`` (``float``, optional): The specified ratio of top-k value. If ``ratio`` is not ``None``, ``k`` will be ignored. Defaults to ``None``.
        ``method`` (``str``, optional): The method to compute the PR curve can be ``"legacy"`` or ``"pascal_voc"``. Defaults to ``"pascal_voc"``.
        ``n_points`` (``int``): The number of points to compute the PR curve. Defaults to ``11``.
        ``ret_batch`` (``bool``): Whether to return the raw score list. Defaults to ``False``.

    Examples:
        >>> import torch
        >>> import dhg.metrics as dm
        >>> y_true = torch.tensor(
                [
                    [0, 1, 0, 1, 0, 0, 1, 0, 1, 0], 
                    [1, 0, 1, 0, 0, 1, 0, 1, 0, 0], 
                    [0, 1, 0, 0, 1, 0, 0, 0, 1, 1],
                ]
            )
        >>> y_pred = torch.tensor(
                [
                    [0.23, 0.76, 0.01, 0.91, 0.13, 0.45, 0.12, 0.03, 0.38, 0.11],
                    [0.33, 0.47, 0.21, 0.87, 0.23, 0.65, 0.22, 0.13, 0.58, 0.21],
                    [0.43, 0.57, 0.31, 0.77, 0.33, 0.85, 0.32, 0.23, 0.78, 0.31],
                ]
            )
        >>> precision_coor, recall_coor = dm.retrieval.pr_curve(y_true, y_pred, method="legacy")
        >>> precision_coor
        [0.6666, 0.6666, 0.6666, 0.6666, 0.6333, 0.6333, 0.6333, 0.5416, 0.5416, 0.5416, 0.4719]
        >>> recall_coor
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        >>> precision_coor, recall_coor = dm.retrieval.pr_curve(y_true, y_pred, method="pascal_voc")
        >>> precision_coor
        [0.6666, 0.6666, 0.6666, 0.6666, 0.6333, 0.6333, 0.6333, 0.5500, 0.5500, 0.5500, 0.4719]
        >>> recall_coor
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    """
    assert method in ("legacy", "pascal_voc",), "The method must be either legacy or pascal_voc."
    y_true, y_pred, k = _format_inputs(y_true, y_pred, k, ratio=ratio)
    precision_coor_list, recall_coor_list = [], []
    for i in range(y_true.shape[0]):
        precision_coor, recall_coor = _pr_curve(y_true[i, :], y_pred[i, :], k=k, method=method, n_points=n_points)
        precision_coor_list.append(precision_coor)
        recall_coor_list.append(recall_coor)
    if ret_batch:
        return precision_coor_list, recall_coor_list
    else:
        precision_coor = np.mean(precision_coor_list, axis=0)
        recall_coor = np.mean(recall_coor_list, axis=0)
        return precision_coor.tolist(), recall_coor.tolist()


class RetrievalEvaluator(BaseEvaluator):
    r"""Return the metric evaluator for retrieval task. The supported metrics includes: ``precision``, ``recall``, ``map``, ``ndcg``, ``mrr``, ``pr_curve``.
    
    Args:
        ``metric_configs`` (``List[Union[str, Dict[str, dict]]]``): The metric configurations. The key is the metric name and the value is the metric parameters.
        ``validate_index`` (``int``): The specified metric index used for validation. Defaults to ``0``.

    Examples:
        >>> import torch
        >>> import dhg.metrics as dm
        >>> evaluator = dm.RetrievalEvaluator(
                [
                    {"recall": {"k": 2}},
                    {"recall": {"k": 4}},
                    {"recall": {"ratio": 0.1}},
                    {"precision": {"k": 4}},
                    {"ndcg": {"k": 4}},
                    "pr_curve",
                    {"pr_curve": {"k": 4, "method": "legacy"}},
                    {"pr_curve": {"k": 4, "method": "pascal_voc", "n_points": 21}},
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
        0.2222222238779068
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
            'recall -> k@2': 0.2222222238779068, 
            'recall -> k@4': 0.6388888955116272, 
            'recall -> ratio@0.1000': 0.1666666716337204, 
            'precision -> k@4': 0.4583333432674408, 
            'ndcg -> k@4': 0.5461218953132629, 
            'pr_curve': [
                [0.7944444517294565, 0.7944444517294565, 0.7944444517294565, 0.7944444517294565, 0.7944444517294565, 0.5888889034589132, 0.5888889034589132, 0.5888889034589132, 0.5888889034589132, 0.5888889034589132, 0.5611111223697662], 
                [0.0, 0.09999999999999999, 0.19999999999999998, 0.30000000000000004, 0.39999999999999997, 0.5, 0.6000000000000001, 0.7000000000000001, 0.7999999999999999, 0.9, 1.0]
            ], 
            'pr_curve -> k@4 | method@legacy': [
                [0.6944444477558136, 0.6944444477558136, 0.6944444477558136, 0.6944444477558136, 0.7222222238779068, 0.4833333392937978, 0.4833333392937978, 0.5000000099341074, 0.5000000099341074, 0.5000000099341074, 0.5611111223697662], 
                [0.0, 0.09999999999999999, 0.19999999999999998, 0.30000000000000004, 0.39999999999999997, 0.5, 0.6000000000000001, 0.7000000000000001, 0.7999999999999999, 0.9, 1.0]
            ], 
            'pr_curve -> k@4 | method@pascal_voc | n_points@21': [
                [0.7944444517294565, 0.7944444517294565, 0.7944444517294565, 0.7944444517294565, 0.7944444517294565, 0.7944444517294565, 0.7944444517294565, 0.7944444517294565, 0.7944444517294565, 0.7944444517294565, 0.5888889034589132, 0.5888889034589132, 0.5888889034589132, 0.5888889034589132, 0.5888889034589132, 0.5888889034589132, 0.5888889034589132, 0.5888889034589132, 0.5888889034589132, 0.5888889034589132, 0.5611111223697662], 
                [0.0, 0.049999999999999996, 0.09999999999999999, 0.15000000000000002, 0.19999999999999998, 0.25, 0.30000000000000004, 0.35000000000000003, 0.39999999999999997, 0.45, 0.5, 0.5499999999999999, 0.6000000000000001, 0.65, 0.7000000000000001, 0.75, 0.7999999999999999, 0.85, 0.9, 0.9500000000000001, 1.0]
            ]
        }
    """

    def __init__(
        self, metric_configs: List[Union[str, Dict[str, dict]]], validate_index: int = 0,
    ):
        super().__init__("retrieval", metric_configs, validate_index)

    def validate_add_batch(self, batch_y_true: torch.Tensor, batch_y_pred: torch.Tensor):
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
