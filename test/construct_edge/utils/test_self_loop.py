import torch

from HyperG.construct_hyedge import remove_self_loop, add_self_loop


def test_remove_self_loop():
    H = torch.tensor([
        [2, 1, 0, 1, 2, 3, 1],
        [0, 0, 0, 1, 2, 2, 3]
    ])
    H_new = remove_self_loop(H)
    assert torch.all(H_new == torch.tensor([
        [2, 1, 0, 2, 3],
        [0, 0, 0, 1, 1]
    ]))


def test_add_self_loop():
    H = torch.tensor([
        [2, 1, 0, 1, 2, 3, 1],
        [0, 0, 0, 1, 2, 2, 3]
    ])
    H_new = add_self_loop(H)
    assert torch.all(H_new == torch.tensor([
        [2, 1, 0, 2, 3, 0, 1, 2, 3],
        [0, 0, 0, 1, 1, 2, 3, 4, 5]
    ]))
