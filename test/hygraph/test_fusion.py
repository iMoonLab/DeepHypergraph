import torch

from HyperG.hygraph import hyedge_concat


def test_hyedge_concat():
    H0 = torch.tensor([
        [0, 2, 3, 2],
        [0, 1, 2, 2]
    ])
    H1 = torch.tensor([
        [0, 3, 2, 1],
        [0, 1, 2, 3]
    ])
    H = hyedge_concat([H0, H1])
    assert torch.all(H == torch.tensor([
        [0, 2, 3, 2, 0, 3, 2, 1],
        [0, 1, 2, 2, 3, 4, 5, 6]
    ]))
