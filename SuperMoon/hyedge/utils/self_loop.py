import torch

from .count import count_node
from .degree import degree_hyedge
from .verify import contiguous_hyedge_idx


def self_loop_remove(H, hyedge_num=None):
    node_idx, hyedge_idx = H
    DE = degree_hyedge(H, hyedge_num)
    loop_edge_idx = torch.where(DE == 1)[0]

    mask = torch.ones_like(hyedge_idx).bool()
    for loop_idx in loop_edge_idx:
        mask = mask & (hyedge_idx != loop_idx)

    H = H[:, mask]
    return contiguous_hyedge_idx(H)


def self_loop_add(H, node_num=None):
    H = self_loop_remove(H)
    node_num = count_node(H, node_num=node_num)

    loop_node_idx = torch.arange(node_num)
    loop_hyedge_idx = torch.arange(node_num)
    loop_H = torch.stack([loop_node_idx, loop_hyedge_idx])

    from SuperMoon.hygraph import hyedge_concat
    return hyedge_concat([H, loop_H])
