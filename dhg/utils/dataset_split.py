import random
from typing import Union, Optional

import torch
import numpy as np

from dhg.structure.graphs import BiGraph


def _idx2mask(num_v: int, train_idx: list, test_idx: list, val_idx: Optional[list] = None):
    train_mask, test_mask = torch.zeros(num_v, dtype=torch.bool), torch.zeros(num_v, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    if val_idx is not None:
        val_mask = torch.zeros(num_v, dtype=torch.bool)
        val_mask[val_idx] = True
        return train_mask, val_mask, test_mask
    else:
        return train_mask, test_mask


def split_by_num(
    num_v: int,
    v_label: Union[list, torch.Tensor, np.ndarray],
    train_num: int,
    val_num: Optional[int] = None,
    test_num: Optional[int] = None,
):
    r"""Split the dataset by the number of vertices in each category, and return the masks of [``train_mask`` and ``test_mask``] or [``train_mask``, ``val_mask`` and ``test_mask``].

    Args:
        num_v (``int``): The number of vertices.
        v_label (``Union[list, torch.Tensor, np.ndarray]``): The vertex labels.
        train_num (``int``): The number of vertices in the training set for each category.
        val_num (``Optional[int]``, optional): The number of vertices in the validation set for each category. If set to ``None``, this function will only return the masks of ``train_mask`` and ``test_mask``. Defaults to ``None``.
        test_num (``Optional[int]``, optional): The number of vertices in the test set for each category. If set to ``None``, except for the training and validation sets, the remaining all vertices will be used for testing. Defaults to ``None``.
    
    Examples:
        >>> import numpy as np
        >>> from dhg.utils import split_by_num
        >>> num_v = 100
        >>> v_label = np.random.randint(0, 3, num_v) # 3 categories
        >>> train_num, val_num, test_num = 10, 2, 5
        >>> train_mask, val_mask, test_mask = split_by_num(num_v, v_label, train_num, val_num, test_num)
        >>> train_mask.sum(), val_mask.sum(), test_mask.sum()
        (tensor(30), tensor(6), tensor(15))
        >>> train_mask, val_mask, test_mask = split_by_num(num_v, v_label, train_num, val_num)
        >>> train_mask.sum(), val_mask.sum(), test_mask.sum()
        (tensor(30), tensor(6), tensor(64))
    """
    if isinstance(v_label, list):
        v_label = np.array(v_label)
    if isinstance(v_label, torch.Tensor):
        v_label = v_label.detach().cpu().numpy()
    assert isinstance(v_label, np.ndarray)
    v_label = v_label.squeeze().astype(int)
    assert v_label.ndim == 1
    if v_label.min() == 1:
        v_label -= 1
    num_classes = np.unique(v_label).shape[0]

    train_idx, test_idx = [], []
    val_num = val_num if val_num is not None else 0
    if val_num != 0:
        val_idx = []
    else:
        val_idx = None

    for lbl_idx in range(num_classes):
        lbl_v_idx = np.where(v_label == lbl_idx)[0]
        random.shuffle(lbl_v_idx)
        train_idx.extend(lbl_v_idx[:train_num])
        if val_num != 0:
            val_idx.extend(lbl_v_idx[train_num : train_num + val_num])
        if test_num is not None:
            test_idx.extend(lbl_v_idx[train_num + val_num : train_num + val_num + test_num])
        else:
            test_idx.extend(lbl_v_idx[train_num + val_num :])

    return _idx2mask(num_v, train_idx, test_idx, val_idx)


def split_by_ratio(
    num_v: int,
    v_label: Union[list, torch.Tensor, np.ndarray],
    train_ratio: float,
    val_ratio: Optional[float] = None,
    test_ratio: Optional[float] = None,
):
    r"""Split the dataset by the ratio of vertices in each category, and return the masks of [``train_mask`` and ``test_mask``] or [``train_mask``, ``val_mask`` and ``test_mask``].

    Args:
        num_v (``int``): The number of vertices.
        v_label (``Union[list, torch.Tensor, np.ndarray]``): The vertex labels.
        train_ratio (``float``): The ratio of vertices in the training set for each category.
        val_ratio (``Optional[float]``, optional): The ratio of vertices in the validation set for each category. If set to ``None``, this function will only return the masks of ``train_mask`` and ``test_mask``. Defaults to ``None``.
        test_ratio (``Optional[float]``, optional): The ratio of vertices in the test set for each category. If set to ``None``, except for the training and validation sets, the remaining all vertices will be used for testing. Defaults to ``None``.
    
    Examples:
        >>> import numpy as np
        >>> from dhg.utils import split_by_ratio
        >>> num_v = 100
        >>> v_label = np.random.randint(0, 3, num_v) # 3 categories
        >>> train_ratio, val_ratio, test_ratio = 0.6, 0.1, 0.2
        >>> train_mask, val_mask, test_mask = split_by_ratio(num_v, v_label, train_ratio, val_ratio, test_ratio)
        >>> train_mask.sum(), val_mask.sum(), test_mask.sum()
        (tensor(59), tensor(9), tensor(18))
        >>> train_mask, val_mask, test_mask = split_by_ratio(num_v, v_label, train_ratio, val_ratio)
        >>> train_mask.sum(), val_mask.sum(), test_mask.sum()
        (tensor(59), tensor(9), tensor(32))
    """
    if isinstance(v_label, list):
        v_label = np.array(v_label)
    if isinstance(v_label, torch.Tensor):
        v_label = v_label.detach().cpu().numpy()
    assert isinstance(v_label, np.ndarray)
    v_label = v_label.squeeze().astype(int)
    assert v_label.ndim == 1
    if v_label.min() == 1:
        v_label -= 1
    num_classes = np.unique(v_label).shape[0]

    val_ratio = val_ratio if val_ratio is not None else 0
    if test_ratio is not None:
        assert train_ratio + val_ratio + test_ratio <= 1
    else:
        assert train_ratio + val_ratio < 1

    train_idx, test_idx = [], []
    if val_ratio != 0:
        val_idx = []
    else:
        val_idx = None

    for lbl_idx in range(num_classes):
        lbl_v_idx = np.where(v_label == lbl_idx)[0]
        _num = lbl_v_idx.shape[0]
        random.shuffle(lbl_v_idx)
        train_num = int(_num * train_ratio)
        val_num = int(_num * val_ratio)
        train_idx.extend(lbl_v_idx[:train_num])
        if val_ratio != 0:
            val_idx.extend(lbl_v_idx[train_num : train_num + val_num])
        if test_ratio is not None:
            test_num = int(_num * test_ratio)
            test_idx.extend(lbl_v_idx[train_num + val_num : train_num + val_num + test_num])
        else:
            test_idx.extend(lbl_v_idx[train_num + val_num :])

    return _idx2mask(num_v, train_idx, test_idx, val_idx)


def split_by_num_for_UI_bigraph(g: BiGraph, train_num: int):
    r"""Split the User-Item bipartite graph by the number of the items connected to each user. This function will return two adjacency matrices for training and testing, respectively.

    Args:
        g (``BiGraph``): The User-Item bipartite graph.
        train_num (``int``): The number of items for the training set for each user.
    
    Examples:
        >>> import dhg
        >>> from dhg.utils import edge_list_to_adj_list, split_by_num_for_UI_bigraph
        >>> g = dhg.random.bigraph_Gnm(5, 8, 20)
        >>> edge_list_to_adj_list(g.e[0])
        [[3, 4, 0, 6, 5], [0, 5, 1, 4, 3, 6], [2, 2, 5, 1], [1, 0, 6, 5, 1, 4, 7], [4, 5, 7]]
        >>> train_num = 3
        >>> train_adj, test_adj = split_by_num_for_UI_bigraph(g, train_num)
        >>> train_adj
        [[0, 1, 3, 4], [1, 6, 0, 5], [2, 1, 2, 5], [3, 6, 4, 5], [4, 5, 7]]
        >>> test_adj
        [[0, 5, 6], [1, 1, 4, 7], [3, 0]]
    """
    train_adj_list, test_adj_list = [], []
    for u_idx in range(g.num_u):
        cur_train, cur_test = [u_idx], [u_idx]
        nbr_v_list = g.nbr_v(u_idx)
        random.shuffle(nbr_v_list)
        _num = len(nbr_v_list)
        if _num == 0:
            continue
        if _num <= train_num:
            cur_train.extend(nbr_v_list)
            train_adj_list.append(cur_train)
        else:
            cur_train.extend(nbr_v_list[:train_num])
            cur_test.extend(nbr_v_list[train_num:])
            train_adj_list.append(cur_train)
            test_adj_list.append(cur_test)
    return train_adj_list, test_adj_list


def split_by_ratio_for_UI_bigraph(g: BiGraph, train_ratio: float):
    r"""Split the User-Item bipartite graph by ratio of the items connected to each user. This function will return two adjacency matrices for training and testing, respectively.

    Args:
        g (``BiGraph``): The User-Item bipartite graph.
        train_ratio (``float``): The ratio of items for the training set for each user.
    
    Examples:
        >>> import dhg
        >>> from dhg.utils import edge_list_to_adj_list, split_by_ratio_for_UI_bigraph
        >>> g = dhg.random.bigraph_Gnm(5, 8, 20)
        >>> edge_list_to_adj_list(g.e[0])
        [[4, 0, 6, 5, 4], [3, 4, 7, 0, 3, 6, 2], [2, 2, 5, 0, 6], [1, 0, 3, 1, 7], [0, 3, 6]]
        >>> train_ratio = 0.8
        >>> train_adj, test_adj = split_by_ratio_for_UI_bigraph(g, train_ratio)
        >>> train_adj
        [[0, 6], [1, 3, 0, 1], [2, 2, 6, 5], [3, 0, 4, 3, 6], [4, 0, 4, 6]]
        >>> test_adj
        [[0, 3], [1, 7], [2, 0], [3, 2, 7], [4, 5]]
    """
    train_adj_list, test_adj_list = [], []
    for u_idx in range(g.num_u):
        cur_train, cur_test = [u_idx], [u_idx]
        nbr_v_list = g.nbr_v(u_idx)
        random.shuffle(nbr_v_list)
        _num = len(nbr_v_list)
        if _num == 0:
            continue
        train_num = int(_num * train_ratio)
        cur_train.extend(nbr_v_list[:train_num])
        cur_test.extend(nbr_v_list[train_num:])
        train_adj_list.append(cur_train)
        test_adj_list.append(cur_test)
    return train_adj_list, test_adj_list

