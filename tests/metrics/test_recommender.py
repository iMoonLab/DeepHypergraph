import pytest

import torch
import math
from sklearn.metrics import ndcg_score
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
    y_true = torch.tensor([[0, 1, 2, 3, 4], [2, 0, 1, 4, 3]])
    y_score = torch.tensor([[0.8, 0.9, 0.6, 0.7, 0.4], [0.4, 0.5, 0.6, 0.7, 0.8]])
    assert dm.ndcg(y_true, y_score, k=2) == pytest.approx(ndcg_score(y_true, y_score, k=2))
    assert dm.ndcg(y_true, y_score, k=3) == pytest.approx(ndcg_score(y_true, y_score, k=3))
    assert dm.ndcg(y_true, y_score, k=4) == pytest.approx(ndcg_score(y_true, y_score, k=4))
    assert dm.ndcg(y_true, y_score, k=5) == pytest.approx(ndcg_score(y_true, y_score, k=5))

    y_true = torch.tensor([0, 1, 0, 0, 1, 1])
    y_pred = torch.tensor([0.8, 0.9, 0.6, 0.7, 0.4, 0.5])
    assert dm.ndcg(y_true, y_pred, k=2) == pytest.approx((1 / math.log2(2)) /  (1 / math.log2(2) + 1 / math.log2(3)))
    assert dm.ndcg(y_true, y_pred, k=3) == pytest.approx((1 / math.log2(2)) /  (1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)))
    assert dm.ndcg(y_true, y_pred, k=5) == pytest.approx((1 / math.log2(2) + 1 / math.log2(6)) /  (1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)))
    
    y_true = torch.tensor([3, 2, 3, 0, 1, 2, 3, 2])
    y_pred = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    assert dm.ndcg(y_true, y_pred, k=6) == pytest.approx(0.785, abs=1e-4)
