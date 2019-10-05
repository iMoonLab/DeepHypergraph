import torch

from HyperG.construct_hyedge import grid_neighbor, distance_neighbor


def test_grid_neighbor():
    H = grid_neighbor((3, 3), False)
    assert torch.all(H == torch.tensor([
        [1, 3, 4, 0, 2, 3, 4, 5, 1, 4, 5, 0, 1, 4, 6, 7, 0, 1, 2, 3, 5, 6, 7, 8, 1, 2, 4, 7, 8, 3, 4, 7, 3, 4, 5, 6, 8,
         4, 5, 7],
        [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 7,
         8, 8, 8]
    ]))

    H = grid_neighbor((3, 3), True)
    assert torch.all(H == torch.tensor([
        [1, 3, 4, 0, 2, 3, 4, 5, 1, 4, 5, 0, 1, 4, 6, 7, 0, 1, 2, 3, 5, 6, 7, 8, 1, 2, 4, 7, 8, 3, 4, 7, 3, 4, 5, 6, 8,
         4, 5, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 7,
         8, 8, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    ]))


def test_distance_neighbor():
    x = torch.tensor([
        [1.0, 2.0],
        [1.0, 0.0],
        [2.0, 2.0]
    ])
    H = distance_neighbor(x, 2)
    assert torch.all(H == torch.tensor([
        [0, 2, 1, 0, 2, 0],
        [0, 0, 1, 1, 2, 2]
    ]))
