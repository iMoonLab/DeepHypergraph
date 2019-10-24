import torch

from SuperMoon.conv import HyConv


def test_HyConv():
    in_ch, out_ch = (16, 32)
    x = torch.rand(4, 16)
    H = torch.tensor([
        [0, 1, 2, 1, 2, 0, 3],
        [0, 0, 0, 1, 1, 2, 2]
    ])
    hyconv = HyConv(in_ch, out_ch)
    assert hyconv(x, H).size() == (4, 32)
