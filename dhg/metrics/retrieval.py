from typing import Optional, Union, Tuple

import torch
import numpy as np


__all__ = [
    "available_retrieval_metrics",
    "precision",
    "recall",
    "ap" "map",
    "ndcg",
    "rr",
    "mrr",
    "pr_curve",
]


def available_retrieval_metrics():
    r"""Return available metrics for the retrieval task.
    
    The available metrics are: ``precision``, ``recall``, ``ap``, ``map``, ``ndcg``, ``rr``, ``mrr``, ``pr_curve``.
    """
    return ("precision", "recall", "ap", "map", "ndcg", "rr", "mrr", "pr_curve")


def _format_inputs(
    y_true: torch.Tensor, y_pred: torch.Tensor, k: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    r"""Format the inputs
    
    Args:
        ``y_true`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``y_pred`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``k`` (``int``, optional): The specified top-k value. Default to :math:`N_{target}`.
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
    k = min(k, max_k) if k is not None else max_k
    return y_true, y_pred, k


def precision(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    k: Optional[int] = None,
    ret_batch: bool = False,
) -> Union[float, list]:
    r"""Compute the Precision score for the retrieval task.

    Args:
        ``y_true`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``y_pred`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``k`` (``int``, optional): The specified top-k value. Defaults to :math:`N_{target}`.
        ``ret_batch`` (``bool``): Whether to return the raw score list. Defaults to ``False``.
    
    Examples:
        >>> import torch
        >>> import dhg.metrics as dm
        >>> y_true = torch.tensor([0, 1, 0, 0, 1, 1])
        >>> y_pred = torch.tensor([0.8, 0.9, 0.6, 0.7, 0.4, 0.5])
        >>> dm.retrieval.precision(y_true, y_pred, k=2)
        0.5
    """
    y_true, y_pred, k = _format_inputs(y_true, y_pred, k)
    assert y_true.max() == 1, "The input y_true must be binary."
    pred_seq = y_true.gather(1, torch.argsort(y_pred, dim=-1, descending=True))[:, :k]
    res_list = pred_seq.sum(dim=1) / k
    if ret_batch:
        return res_list
    else:
        return res_list.mean().item()


def recall(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    k: Optional[int] = None,
    ret_batch: bool = False,
) -> Union[float, list]:
    r"""Compute the Recall score for the retrieval task.

    Args:
        ``y_true`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``y_pred`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``k`` (``int``, optional): The specified top-k value. Defaults to :math:`N_{target}`.
        ``ret_batch`` (``bool``): Whether to return the raw score list. Defaults to ``False``.

    
    Examples:
        >>> import torch
        >>> import dhg.metrics as dm
        >>> y_true = torch.tensor([0, 1, 0, 0, 1, 1])
        >>> y_pred = torch.tensor([0.8, 0.9, 0.6, 0.7, 0.4, 0.5])
        >>> dm.retrieval.recall(y_true, y_pred, k=5)
        0.6666666666666666
    """
    y_true, y_pred, k = _format_inputs(y_true, y_pred, k)
    assert y_true.max() == 1, "The input y_true must be binary."
    pred_seq = y_true.gather(1, torch.argsort(y_pred, dim=-1, descending=True))[:, :k]
    num_true = y_true.sum(dim=1)
    res_list = pred_seq.sum(dim=1) / num_true
    if ret_batch:
        return res_list
    else:
        return res_list.mean().item()


def ap(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    k: Optional[int] = None,
    method: str = "pascal_voc",
) -> Union[float, list]:
    r"""Compute the Average Precision (AP) for the retrieval task.

    Args:
        ``y_true`` (``torch.Tensor``): A 1-D tensor. Size :math:`(N_{target},)`.
        ``y_pred`` (``torch.Tensor``): A 1-D tensor. Size :math:`(N_{target},)`.
        ``k`` (``int``, optional): The specified top-k value. Defaults to :math:`N_{target}`.
        ``method`` (``str``): The method to compute the AP can be ``legacy`` or ``pascal_voc``. Defaults to ``pascal_voc``.
    
    Examples:
        >>> import torch
        >>> import dhg.metrics as dm
        >>> y_true = torch.tensor([True, False, True])
        >>> y_pred = torch.tensor([0.2, 0.3, 0.5])
        >>> dm.retrieval.ap(y_true, y_pred, method="legacy")
        0.8333333730697632
    """
    assert method in (
        "legacy",
        "pascal_voc",
    ), "The method must be either legacy or pascal_voc."
    assert (
        y_true.shape == y_pred.shape
    ), "The shape of y_true and y_pred must be the same."
    assert y_true.dim() == 1, "The input y_true must be 1-D."
    assert y_pred.dim() == 1, "The input y_pred must be 1-D."
    y_true, y_pred = y_true.detach().float(), y_pred.detach().float()
    max_k = y_true.shape[0]
    k = min(k, max_k) if k is not None else max_k

    pred_seq = y_true[torch.argsort(y_pred, descending=True)]
    pred_index = torch.arange(1, len(y_true) + 1, device=y_true.device)[pred_seq > 0]
    recall_seq = torch.arange(1, len(pred_index) + 1, device=y_true.device)
    res = recall_seq / pred_index
    if method == "pascal_voc":
        res = torch.flip(res, dims=(0,))
        res = torch.cummax(res, dim=0)[0]
    return res.mean().item()


def map(
    y_true: torch.LongTensor,
    y_pred: torch.LongTensor,
    k: Optional[int] = None,
    method: str = "pascal_voc",
    ret_batch: bool = False,
) -> Union[float, list]:
    r"""Compute the mean Average Precision (mAP) for the retrieval task.

    Args:
        ``y_true`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``y_pred`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``k`` (``int``, optional): The specified top-k value. Default to :math:`N_{target}`.
        ``method`` (``str``): The specified method: ``legacy`` or ``pascal_voc``. Defaults to ``pascal_voc``.
        ``ret_batch`` (``bool``): Whether to return the raw score list. Defaults to ``False``.
    
    Examples:
        >>> import torch
        >>> import dhg.metrics as dm
        >>> y_true = torch.tensor([
                [True, False, True, False, True],
                [False, False, False, True, True],
                [True, True, False, True, False],
            ])
        >>> y_pred = torch.tensor([
                [0.2, 0.3, 0.5, 0.4, 0.3],
                [0.8, 0.2, 0.3, 0.5, 0.4],
                [0.2, 0.4, 0.5, 0.2, 0.8],
            ])
        >>> dm.retrieval.map(y_true, y_pred, method="legacy")
        0.587037056684494
    """
    assert method in (
        "legacy",
        "pascal_voc",
    ), "The method must be either legacy or pascal_voc."
    y_true, y_pred, k = _format_inputs(y_true, y_pred, k)
    res_list = [ap(y_true[i, :], y_pred[i, :], k) for i in range(y_true.shape[0])]
    if ret_batch:
        return res_list
    else:
        return np.mean(res_list)


def _dcg(matrix: torch.Tensor) -> torch.Tensor:
    r"""Compute the Discounted Cumulative Gain (DCG).
    
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
    ret_batch: bool = False,
) -> Union[float, list]:
    r"""Compute the Normalized Discounted Cumulative Gain (NDCG) for the retrieval task.

    Args:
        ``y_true`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``y_pred`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``k`` (``int``, optional): The specified top-k value. Default to :math:`N_{target}`.
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
    y_true, y_pred, k = _format_inputs(y_true, y_pred, k)

    pred_seq = y_true.gather(1, torch.argsort(y_pred, dim=-1, descending=True))[:, :k]
    ideal_seq = torch.sort(y_true, dim=-1, descending=True)[0][:, :k]

    pred_dcg = _dcg(pred_seq)
    ideal_dcg = _dcg(ideal_seq)

    res_list = pred_dcg / ideal_dcg
    res_list[torch.isinf(res_list)] = 0
    if ret_batch:
        return res_list
    else:
        return res_list.mean().item()


def rr(y_true: torch.Tensor, y_pred: torch.Tensor, k: Optional[int] = None) -> float:
    r"""Compute the Reciprocal Rank (RR) for the retrieval task.

    Args:
        ``y_true`` (``torch.Tensor``): A 1-D tensor. Size :math:`(N_{target},)``.
        ``y_pred`` (``torch.Tensor``): A 1-D tensor. Size :math:`(N_{target},)`.
        ``k`` (``int``, optional): The specified top-k value. Default to :math:`N_{target}`.
    
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
    k = min(k, max_k) if k is not None else max_k

    pred_seq = y_true[torch.argsort(y_pred, dim=-1, descending=True)][:k]
    pred_index = torch.nonzero(pred_seq).view(-1)
    res = (1 / (pred_index + 1)).mean()
    return res.mean().item()


def mrr(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    k: Optional[int] = None,
    ret_batch: bool = False,
) -> Union[float, list]:
    r"""Compute the mean Reciprocal Rank (MRR) for the retrieval task.

    Args:
        ``y_true`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``y_pred`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``k`` (``int``, optional): The specified top-k value. Default to :math:`N_{target}`.
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
    y_true, y_pred, k = _format_inputs(y_true, y_pred, k)
    res_list = [rr(y_true[i, :], y_pred[i, :], k) for i in range(y_true.shape[0])]
    if ret_batch:
        return res_list
    else:
        return np.mean(res_list)


def _pr_curve(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    k: Optional[int] = None,
    method: str = "pascal_voc",
    n_points: int = 11,
) -> tuple:
    r"""Compute the Precision-Recall Curve for the retrieval task.

    Args:
        ``y_true`` (``torch.Tensor``): A 1-D tensor. Size :math:`(N_{target},)`.
        ``y_pred`` (``torch.Tensor``): A 1-D tensor. Size :math:`(N_{target},)`.
        ``k`` (``int``, optional): The specified top-k value. Default to :math:`N_{target}`.
        ``method`` (``str``, optional): The method to compute the PR curve can be "legacy" or "pascal_voc". Default to "pascal_voc".
        ``n_points`` (``int``): The number of points to compute the PR curve. Default to ``11``.

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
    assert method in (
        "legacy",
        "pascal_voc",
    ), "The method must be either legacy or pascal_voc."
    assert (
        y_true.shape == y_pred.shape
    ), "The shape of y_true and y_pred must be the same."
    assert y_true.dim() == 1, "The input y_true must be 1-D."
    assert y_pred.dim() == 1, "The input y_pred must be 1-D."
    y_true, y_pred = y_true.detach().float(), y_pred.detach().float()
    max_k = y_true.shape[0]
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
    method: str = "pascal_voc",
    n_points: int = 11,
    ret_batch: bool = False,
) -> tuple:
    r"""Compute the Precision-Recall Curve for the retrieval task.

    Args:
        ``y_true`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``y_pred`` (``torch.Tensor``): A 1-D tensor or 2-D tensor. Size :math:`(N_{target},)` or :math:`(N_{samples}, N_{target})`.
        ``k`` (``int``, optional): The specified top-k value. Default to :math:`N_{target}`.
        ``method`` (``str``, optional): The method to compute the PR curve can be "legacy" or "pascal_voc". Default to "pascal_voc".
        ``n_points`` (``int``): The number of points to compute the PR curve. Default to ``11``.
        ``ret_batch`` (``bool``): Whether to return the raw score list. Defaults to ``False``.

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
    assert method in (
        "legacy",
        "pascal_voc",
    ), "The method must be either legacy or pascal_voc."
    y_true, y_pred, k = _format_inputs(y_true, y_pred, k)
    precision_coor_list, recall_coor_list = [], []
    for i in range(y_true.shape[0]):
        precision_coor, recall_coor = _pr_curve(
            y_true[i, :], y_pred[i, :], k, method, n_points
        )
        precision_coor_list.append(precision_coor)
        recall_coor_list.append(recall_coor)
    if ret_batch:
        return precision_coor_list, recall_coor_list
    else:
        precision_coor = np.mean(precision_coor_list, axis=0)
        recall_coor = np.mean(recall_coor_list, axis=0)
        return precision_coor.tolist(), recall_coor.tolist()
