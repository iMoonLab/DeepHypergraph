import torch

from .count import count_hyedge, count_node


def degree_node(H):
    node_idx, edge_idx = H
    node_num = count_node(H)
    return torch.zeros(node_num).scatter_add(0, node_idx, torch.ones_like(node_idx).float()).long()


def degree_hyedge(H):
    node_idx, hyedge_idx = H
    edge_num = count_hyedge(H)
    return torch.zeros(edge_num).scatter_add(0, hyedge_idx, torch.ones_like(hyedge_idx).float()).long()
