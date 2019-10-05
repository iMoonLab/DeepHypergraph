import torch

from HyperG.construct_hyedge import hyedge_degree


def test_edge_degree():
    H = torch.tensor([
        [2, 1, 0, 1, 2, 3, 1],
        [0, 0, 0, 1, 1, 1, 2]
    ])
    assert torch.all(hyedge_degree(H) == torch.tensor([3, 3, 1]))
