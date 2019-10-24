import torch

from SuperMoon.models import ResNet_HGNN


def test_ResNet_HGNN():
    x = torch.randn(1, 3, 224, 224)

    n_class = 40
    depth = 34
    k_nearest = 5
    hiddens = [512]
    model = ResNet_HGNN(n_class, depth, k_nearest, hiddens, pretrained=False)

    assert model(x).size() == (1, 40)
