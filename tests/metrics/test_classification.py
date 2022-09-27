import pytest

import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import dhg.metrics.classification as dm


def test_accuracy():
    y_true = torch.tensor([0, 1, 2, 0, 1, 3])
    y_pred = torch.tensor([0, 2, 1, 0, 2, 3])
    assert dm.accuracy(y_true, y_pred) == 0.5


def test_f1_score():
    y_true = torch.tensor([0, 1, 2, 0, 1, 3])
    y_pred = torch.tensor([0, 2, 1, 0, 2, 3])
    assert dm.f1_score(y_true, y_pred, 'macro') == f1_score(y_true, y_pred, average='macro')
    assert dm.f1_score(y_true, y_pred, 'micro') == f1_score(y_true, y_pred, average='micro')
    assert dm.f1_score(y_true, y_pred, 'weighted') == f1_score(y_true, y_pred, average='weighted')


def test_confusion_matrix():
    y_true = torch.tensor([0, 1, 2, 0, 1, 3])
    y_pred = torch.tensor([0, 2, 1, 0, 2, 3])
    assert np.all(dm.confusion_matrix(y_true, y_pred) == confusion_matrix(y_true, y_pred))
