from itertools import product

import torch

from . import remove_negative_index, self_loop_add, pairwise_euclidean_distance


def neighbor_grid(input_size, self_loop=False, neigh_funs__mask_funs=None):
    """

    :param input_size: w \times h matrix
    :param self_loop:
    :return:
    """
    assert len(input_size) == 2, \
        f"grid neighbor input size error, must be a matrix with two dimensions"
    node_num = input_size[0] * input_size[1]
    node_set = torch.arange(node_num)

    if neigh_funs__mask_funs is None:
        neigh_funs = [
            lambda _idx, _w, _h: _idx - _w - 1,
            lambda _idx, _w, _h: _idx - _w,
            lambda _idx, _w, _h: _idx - _w + 1,

            lambda _idx, _w, _h: _idx - 1,
            lambda _idx, _w, _h: _idx + 1,

            lambda _idx, _w, _h: _idx + _w - 1,
            lambda _idx, _w, _h: _idx + _w,
            lambda _idx, _w, _h: _idx + _w + 1,
        ]
        neigh_mask_funs = [
            lambda _w, _h: (node_set // _w == 0) | (node_set % _w == 0),
            lambda _w, _h: (node_set // _w == 0),
            lambda _w, _h: (node_set // _w == 0) | (node_set % _w == _w - 1),

            lambda _w, _h: (node_set % _w == 0),
            lambda _w, _h: (node_set % _w == _w - 1),

            lambda _w, _h: (node_set % _w == 0) | (node_set // _w == _h - 1),
            lambda _w, _h: (node_set // _w == _h - 1),
            lambda _w, _h: (node_set // _w == _h - 1) | (node_set % _w == _w - 1),
        ]
    else:
        neigh_funs, neigh_mask_funs = neigh_funs__mask_funs

    neigh_masks = [neigh_mask_fun(input_size[0], input_size[1]) for neigh_mask_fun in neigh_mask_funs]
    neigh_num = len(neigh_funs)

    node_idx = []
    for neigh_fun, neigh_mask in zip(neigh_funs, neigh_masks):
        _tmp = neigh_fun(node_set, input_size[0], input_size[1])
        _tmp[neigh_mask] = -1
        node_idx.append(_tmp)
    # neigh_num * input_size
    node_idx = torch.stack(node_idx, dim=1).reshape(-1)

    hyedge_idx = node_set.unsqueeze(0).repeat(neigh_num, 1).transpose(1, 0).reshape(-1)

    # construct sparse hypergraph adjacency matrix from (node_idx,hyedge_idx) pair.
    H = torch.stack([node_idx, hyedge_idx])

    H = remove_negative_index(H)

    if self_loop:
        H = self_loop_add(H)

    return H


def neighbor_distance(x: torch.Tensor, k_nearest, dis_metric=pairwise_euclidean_distance):
    """
    construct hyperedge for each node in x matrix. Each hyperedge contains a node and its k-1 nearest neighbors.
    :param x: N x C matrix. N denotes node number, and C is the feature dimension.
    :param k_nearest:
    :return:
    """
    assert len(x.shape) == 2

    # N x C
    node_num = x.size(0)
    dis_matrix = dis_metric(x)
    _, nn_idx = torch.topk(dis_matrix, k_nearest, dim=1, largest=False)
    hyedge_idx = torch.arange(node_num).unsqueeze(0).repeat(k_nearest, 1).transpose(1, 0).reshape(-1)
    H = torch.stack([nn_idx.reshape(-1), hyedge_idx])
    return H


def gather_patch_ft(x: torch.Tensor, patch_size):
    """

    :param x: 1 x C x M x N
    :param patch_size: row x column
    :return:
    """
    assert len(x.shape) == 4
    assert len(patch_size) == 2

    # C x M x N
    x = x.view([x.size(1), x.size(2), x.size(3)])
    # M x N x C
    x = x.permute([1, 2, 0])

    x_row_num, x_col_num = x.shape[0], x.shape[1]
    out = []

    center_row, center_col = (patch_size[0] + 1) // 2 - 1, (patch_size[1] + 1) // 2 - 1

    ft_tmp = torch.zeros(x.size(0) + patch_size[0] - 1, x.size(1) + patch_size[1] - 1, x.size(2))
    ft_tmp[center_row:center_row + x_row_num, center_col:center_col + x_col_num] = x

    for _row, _col in product(range(patch_size[0]), range(patch_size[1])):
        out.append(ft_tmp[_row:_row + x_row_num, _col:_col + x_col_num])

    out = torch.cat(out, dim=2).permute([2, 0, 1]).unsqueeze(0)
    return out
