import torch

from .degree import degree_hyedge


def contiguous_hyedge_idx(H):
    node_idx, hyedge_idx = H
    DE = degree_hyedge(H)
    zero_idx = torch.where(DE == 0)[0]

    bias = torch.zeros_like(hyedge_idx)
    for _idx in zero_idx:
        bias[hyedge_idx > _idx] -= 1

    hyedge_idx += bias
    return torch.stack([node_idx, hyedge_idx])


def filter_node_index(H, idx_range):
    node_idx, edge_idx = H
    mask = (node_idx >= idx_range[0]) & (node_idx < idx_range[1])
    return contiguous_hyedge_idx(H[:, mask])


def remove_negative_index(H):
    node_idx, _ = H
    mask = node_idx >= 0
    return contiguous_hyedge_idx(H[:, mask])
