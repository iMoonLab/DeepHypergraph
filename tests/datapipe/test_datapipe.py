import torch
import pytest
import dhg.datapipe as dp

def test_norm():
    ft = torch.rand(5, 16)
    ft_norm = dp.norm_ft(ft, ord=1)
    row_norm = 1 / ft.sum(dim=1, keepdim=True)
    row_norm[torch.isinf(row_norm)] = 0
    _ft_norm = ft * row_norm
    assert pytest.approx(ft_norm) == _ft_norm
