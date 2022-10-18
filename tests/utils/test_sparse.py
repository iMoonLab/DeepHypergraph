from re import A
import pytest
import torch
from dhg.utils import sparse_dropout


def test_sparse_dropout():
    a = (torch.rand(10, 20) > 0.7).float()

    idx = torch.nonzero(a).T
    data = a[idx[0], idx[1]]
    coo = torch.sparse_coo_tensor(idx, data, a.shape).coalesce()

    dropped = sparse_dropout(coo, 0.3)

    assert coo.size() == dropped.size()

    assert (dropped._values()!=0).sum() == pytest.approx(coo._nnz() * 0.7, 0.15)

    for i in range(10):
        for j in range(20):
            assert dropped[i, j] == a[i, j] or dropped[i, j] == 0
