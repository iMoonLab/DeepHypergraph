import pytest

import numpy as np
from sklearn.metrics import average_precision_score, ndcg_score
import torch
import dhg.metrics.retrieval as dm


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


def test_ap():
    y_true = torch.tensor([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1])
    y_pred = torch.tensor(
        [
            0.23,
            0.76,
            0.01,
            0.91,
            0.13,
            0.45,
            0.12,
            0.03,
            0.38,
            0.11,
            0.03,
            0.09,
            0.65,
            0.07,
            0.12,
            0.24,
            0.10,
            0.23,
            0.46,
            0.08,
        ]
    )
    assert pytest.approx(dm.ap(y_true, y_pred)) == 0.6620671153068542
    assert pytest.approx(dm.ap(y_true, y_pred, method="legacy")) == 0.6501623392105103


def test_map():
    y_true = torch.tensor([
        [True, False, True, False, True],
        [False, False, False, True, True],
        [True, True, False, True, False],
        [False, True, True, False, True],
    ])
    y_pred = torch.tensor([
        [0.2, 0.8, 0.5, 0.4, 0.3],
        [0.8, 0.2, 0.3, 0.9, 0.4],
        [0.2, 0.4, 0.5, 0.9, 0.8],
        [0.8, 0.2, 0.9, 0.3, 0.7],
    ])
    ap = []
    for i in range(y_true.shape[0]):
        ap.append(average_precision_score(y_true[i], y_pred[i]))
    map = np.mean(ap)
    assert dm.map(y_true, y_pred, method='legacy') == pytest.approx(map)



def test_ndcg():
    y_true = torch.tensor([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1])
    y_pred = torch.tensor([0.23, 0.76, 0.01, 0.91, 0.13, 0.45, 0.12, 0.03, 0.38, 0.11, 0.03, 0.09, 0.65, 0.07, 0.12, 0.24, 0.10, 0.23, 0.46, 0.08])
    assert dm.ndcg(y_true, y_pred, k=5) == pytest.approx(ndcg_score(y_true.reshape(1, -1), y_pred.reshape(1, -1), k=5))
    pass


def test_rr():
    # TODO
    pass


def test_mrr():
    # TODO
    pass


def test_pr_curve():
    y_true = torch.tensor([0, 1, 0, 1, 0, 0, 1, 0, 1, 0])
    y_pred = torch.tensor([0.23, 0.76, 0.01, 0.91, 0.13, 0.45, 0.12, 0.03, 0.38, 0.11])
    precision_coor, recall_coor = dm.pr_curve(y_true, y_pred)
    assert pytest.approx(precision_coor) == [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.75,
        0.75,
        0.75,
        0.5714285969734192,
    ]
    assert pytest.approx(recall_coor) == [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]
    precision_coor, recall_coor = dm.pr_curve(y_true, y_pred, n_points=21)
    assert pytest.approx(precision_coor) == [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.75,
        0.75,
        0.75,
        0.75,
        0.75,
        0.75,
        0.5714285969734192,
    ]
    assert pytest.approx(recall_coor) == [
        0.0,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.30,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        1.0,
    ]
