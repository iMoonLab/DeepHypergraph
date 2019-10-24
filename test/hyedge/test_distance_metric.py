import torch

from SuperMoon.hyedge import pairwise_euclidean_distance


def test_pairwise_euclidean_distance():
    x = torch.tensor([
        [1.0, 2.0],
        [1.0, 0.0],
        [2.0, 2.0]
    ])
    dis = pairwise_euclidean_distance(x)
    assert torch.all(dis == torch.tensor([
        [0.0, 4.0, 1.0],
        [4.0, 0.0, 5.0],
        [1.0, 5.0, 0.0],
    ]))
