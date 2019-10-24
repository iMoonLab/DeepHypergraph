import torch

from SuperMoon.hyedge import self_loop_remove, self_loop_add


def test_remove_self_loop():
    H = torch.tensor([
        [2, 1, 0, 1, 2, 3, 1],
        [0, 0, 0, 1, 2, 2, 3]
    ])
    H_new = self_loop_remove(H)
    assert torch.all(H_new == torch.tensor([
        [2, 1, 0, 2, 3],
        [0, 0, 0, 1, 1]
    ]))


def test_add_self_loop():
    H = torch.tensor([
        [2, 1, 0, 1, 2, 3, 1],
        [0, 0, 0, 1, 2, 2, 3]
    ])
    H_new = self_loop_add(H)
    assert torch.all(H_new == torch.tensor([
        [2, 1, 0, 2, 3, 0, 1, 2, 3],
        [0, 0, 0, 1, 1, 2, 3, 4, 5]
    ]))
