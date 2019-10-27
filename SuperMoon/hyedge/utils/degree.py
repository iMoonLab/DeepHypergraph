import torch

from .count import count_hyedge, count_node


def degree_node(H, node_num=None):
    node_idx, edge_idx = H
    node_num = count_node(H, node_num)
    src = torch.ones_like(node_idx).float().to(H.device)
    out = torch.zeros(node_num).to(H.device)
    return out.scatter_add(0, node_idx, src).long()
    # return torch.zeros(node_num).scatter_add(0, node_idx, torch.ones_like(node_idx).float()).long()


def degree_hyedge(H: torch.Tensor, hyedge_num=None):
    node_idx, hyedge_idx = H
    hyedge_num = count_hyedge(H, hyedge_num=hyedge_num)
    src = torch.ones_like(hyedge_idx).float().to(H.device)
    out = torch.zeros(hyedge_num).to(H.device)
    return out.scatter_add(0, hyedge_idx, src).long()
