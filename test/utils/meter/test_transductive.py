import torch

from SuperMoon.utils.meter import trans_class_acc, trans_iou_socre


def test_trans_class_acc():
    pred = torch.tensor([
        [0.2, 0.8],
        [0.3, 0.7],
        [0.1, 0.9],
        [0.4, 0.6],
        [0.2, 0.8],
    ])
    target = torch.tensor([1, 0, 1, 0, 1])
    mask = torch.tensor([1, 1, 1, 1, 0])
    acc = trans_class_acc(pred, target, mask)
    assert acc == 0.5


def test_trans_iou_score():
    pred = torch.tensor([
        [0.2, 0.8],
        [0.3, 0.7],
        [0.1, 0.9],
        [0.4, 0.6],
        [0.2, 0.8],
    ])
    target = torch.tensor([1, 0, 1, 0, 1])
    mask = torch.tensor([1, 1, 1, 1, 0])
    iou = trans_iou_socre(pred, target, mask)
    assert (iou[0] - 0.5) < 1e-5
