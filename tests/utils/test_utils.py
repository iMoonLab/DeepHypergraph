import torch
import pytest
from dhg import utils


def test_sparse_softmax():
    a = torch.rand(5, 10)
    a_sparse = a.clone().to_sparse()
    res_sparse = torch.sparse.softmax(a_sparse, dim=1)
    res_dense = torch.softmax(a, dim=1)
    assert pytest.approx(res_dense) == res_sparse.to_dense()

def test_C():
    assert pytest.approx(utils.C(5, 2)) == 10
    assert pytest.approx(sum(utils.C(100, i) for i in range(101))) == 2 ** 100
