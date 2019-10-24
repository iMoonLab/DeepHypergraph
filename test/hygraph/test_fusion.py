import torch

from SuperMoon.hygraph import hyedge_concat


def test_hyedge_concat():
    H0 = torch.tensor([
        [0, 2, 3, 2],
        [0, 1, 2, 2]
    ])
    H1 = torch.tensor([
        [0, 3, 2, 1],
        [0, 1, 2, 3]
    ])
    H_same_node = hyedge_concat([H0, H1])
    assert torch.all(H_same_node == torch.tensor([
        [0, 2, 3, 2, 0, 3, 2, 1],
        [0, 1, 2, 2, 3, 4, 5, 6]
    ]))

    H0 = torch.tensor([
        [0, 2, 3, 2],
        [0, 1, 2, 2]
    ])
    H1 = torch.tensor([
        [0, 3, 2, 1],
        [0, 1, 2, 3]
    ])
    H_not_same_node = hyedge_concat([H0, H1], same_node=False)
    assert torch.all(H_not_same_node == torch.tensor([
        [0, 2, 3, 2, 4, 7, 6, 5],
        [0, 1, 2, 2, 3, 4, 5, 6]
    ]))
