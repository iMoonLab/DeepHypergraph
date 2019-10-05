from functools import reduce

import torch

from . import remove_negative_index, add_self_loop, pairwise_euclidean_distance


def grid_neighbor(input_size, self_loop=False, neigh_funs__mask_funs=None):
    """

    :param input_size: w \times h matrix
    :param self_loop:
    :return:
    """
    input_size = torch.tensor(input_size)
    assert len(input_size.shape) == 1 and input_size.shape[0] == 2, \
        f"grid neighbor input size error, must be a matrix with two dimensions"
    node_num = reduce(lambda x, y: x * y, input_size)
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
        H = add_self_loop(H)

    return H


def distance_neighbor(x, k_nearest):
    """
    construct hyperedge for each node in x matrix. Each hyperedge contains a node and its k-1 nearest neighbors.
    :param x: N x C matrix. N denotes node number, and C is the feature dimension.
    :param k_nearest:
    :return:
    """
    assert isinstance(x, torch.Tensor)

    x = x.squeeze()
    if len(x.shape) == 2:
        x = x.unsqueeze(0)

    # x (N x C)
    node_num = x.size(0)
    dis_matrix = pairwise_euclidean_distance(x)
    _, nn_idx = torch.topk(dis_matrix, k_nearest, dim=1, largest=False)
    hyedge_idx = torch.arange(node_num).unsqueeze(0).repeat(k_nearest, 1).transpose(1, 0).reshape(-1)
    H = torch.stack([nn_idx.reshape(-1), hyedge_idx])
    return H
