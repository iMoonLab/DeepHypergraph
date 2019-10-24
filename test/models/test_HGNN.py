import torch

from SuperMoon.models import HGNN


def test_HGNN():
    x = torch.tensor([
        [0.1, 2.0],
        [0.4, 0.4],
        [0.2, 0.6],
        [0.5, 0.2],
        [0.6, 0.8]
    ])
    H = torch.tensor([
        [0, 3, 4, 2, 2, 1, 3, 3, 2, 3],
        [0, 0, 0, 1, 1, 2, 3, 4, 4, 4]
    ])
    model = HGNN(2, 4, [3])
    assert model(x, H).shape == (5, 4)
