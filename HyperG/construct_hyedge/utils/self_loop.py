import torch

from .count import node_count
from .degree import hyedge_degree
from .verify import contiguous_hyedge_idx
from HyperG.construct_hygraph import hyedge_concat


def remove_self_loop(H):
    node_idx, hyedge_idx = H
    DE = hyedge_degree(H)
    loop_edge_idx = torch.where(DE == 1)[0]

    mask = torch.ones_like(hyedge_idx).bool()
    for loop_idx in loop_edge_idx:
        mask = mask & (hyedge_idx != loop_idx)

    H = H[:, mask]
    return contiguous_hyedge_idx(H)


def add_self_loop(H):
    H = remove_self_loop(H)
    node_num = node_count(H)

    loop_node_idx = torch.arange(node_num)
    loop_hyedge_idx = torch.arange(node_num)
    loop_H = torch.stack([loop_node_idx, loop_hyedge_idx])

    return hyedge_concat([H, loop_H])
