import torch

from .count import hyedge_count, node_count


def node_degree(H):
    node_idx, edge_idx = H
    node_num = node_count(H)
    return torch.zeros(node_num).scatter_add(0, node_idx, torch.ones_like(node_idx).float()).long()


def hyedge_degree(H):
    node_idx, hyedge_idx = H
    edge_num = hyedge_count(H)
    return torch.zeros(edge_num).scatter_add(0, hyedge_idx, torch.ones_like(hyedge_idx).float()).long()
