import torch

from SuperMoon.hyedge import degree_hyedge


def test_edge_degree():
    H = torch.tensor([
        [2, 1, 0, 1, 2, 3, 1],
        [0, 0, 0, 1, 1, 1, 2]
    ])
    assert torch.all(degree_hyedge(H) == torch.tensor([3, 3, 1]))
