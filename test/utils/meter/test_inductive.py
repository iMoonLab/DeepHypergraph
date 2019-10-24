import torch

from SuperMoon.utils.meter import CIndexMeter


def test_CIndexMeter():
    c_index = CIndexMeter()
    preds = [0.4, 0.3, 0.6]
    targets = [0.5, 0.2, 0.4]
    for pred, target in zip(preds, targets):
        c_index.add(torch.tensor(pred), torch.tensor(target))
    assert c_index.value() - 2.0 / 3 < 1e-6
