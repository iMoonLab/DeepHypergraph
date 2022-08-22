import pytest

import torch
import dhg.metrics.recommender as dm


def test_recall():
    y_true = torch.tensor([0, 1, 0, 0, 1, 1])
    y_pred = torch.tensor([0.8, 0.9, 0.6, 0.7, 0.4, 0.5])
    # sorted: [1, 0, 0, 0, 1, 1]
    assert dm.recall(y_true, y_pred, k=2) == pytest.approx(1 / 3)
    assert dm.recall(y_true, y_pred, k=3) == pytest.approx(1 / 3)
    assert dm.recall(y_true, y_pred, k=5) == pytest.approx(2 / 3)


def test_precision():
    y_true = torch.tensor([0, 1, 0, 0, 1, 1])
    y_pred = torch.tensor([0.8, 0.9, 0.6, 0.7, 0.4, 0.5])
    # sorted: [1, 0, 0, 0, 1, 1]
    assert dm.precision(y_true, y_pred, k=2) == pytest.approx(1 / 2)
    assert dm.precision(y_true, y_pred, k=3) == pytest.approx(1 / 3)
    assert dm.precision(y_true, y_pred, k=5) == pytest.approx(2 / 5)


def test_ndcg():
    # TODO
    pass
