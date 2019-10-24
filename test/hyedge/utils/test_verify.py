import torch

from SuperMoon.hyedge import filter_node_index, remove_negative_index, contiguous_hyedge_idx


def test_filter_node_index():
    H = torch.tensor([
        [-2, -1, 0, 1, 2, 3, 1],
        [0, 0, 0, 1, 1, 1, 2]
    ])
    H_new = filter_node_index(H, [0, 2])
    node_idx, edge_idx = H_new
    assert torch.all(node_idx >= 0)
    assert torch.all(node_idx < 2)
    assert torch.all(node_idx == torch.tensor([0, 1, 1]))
    assert torch.all(edge_idx == torch.tensor([0, 1, 2]))


def test_remove_negative_index():
    H = torch.tensor([
        [-2, -1, 0, 1, 2, 3, 1],
        [0, 0, 0, 1, 1, 1, 2]
    ])
    H_new = remove_negative_index(H)
    node_idx, edge_idx = H_new
    assert torch.all(node_idx >= 0)
    assert torch.all(node_idx == torch.tensor([0, 1, 2, 3, 1]))
    assert torch.all(edge_idx == torch.tensor([0, 1, 1, 1, 2]))


def test_contiguous_hyedge_idx():
    H = torch.tensor([
        [-2, -1, 0, 1, 2, 3, 1],
        [0, 0, 0, 2, 4, 4, 6]
    ])
    H_new = contiguous_hyedge_idx(H)
    assert torch.all(H_new == torch.tensor([
        [-2, -1, 0, 1, 2, 3, 1],
        [0, 0, 0, 1, 2, 2, 3]
    ]))
