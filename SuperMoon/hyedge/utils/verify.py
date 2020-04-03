import torch

from .degree import degree_hyedge


def contiguous_hyedge_idx(H):
    node_idx, hyedge_idx = H
    unorder_pairs = [(hyedge_id, sequence_id) for sequence_id, hyedge_id in enumerate(hyedge_idx.numpy().tolist())]
    unorder_pairs.sort(key=lambda x: x[0])
    new_hyedge_id = -1
    pre_hyedge_id = None
    new_hyedge_idx = list()
    sequence_idx = list()
    for (hyedge_id, sequence_id) in unorder_pairs:
        if hyedge_id != pre_hyedge_id:
            new_hyedge_id += 1
            pre_hyedge_id = hyedge_id
        new_hyedge_idx.append(new_hyedge_id)
        sequence_idx.append(sequence_id)
    hyedge_idx[sequence_idx] = torch.LongTensor(new_hyedge_idx)
    return torch.stack([node_idx, hyedge_idx])


def filter_node_index(H, idx_range):
    node_idx, edge_idx = H
    mask = (node_idx >= idx_range[0]) & (node_idx < idx_range[1])
    return contiguous_hyedge_idx(H[:, mask])


def remove_negative_index(H):
    node_idx, _ = H
    mask = node_idx >= 0
    return contiguous_hyedge_idx(H[:, mask])
