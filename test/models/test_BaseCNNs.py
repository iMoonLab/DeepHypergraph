import pytest
import torch

from SuperMoon.models import ResNetFeature, ResNetClassifier


@pytest.mark.parametrize('depth', [18, 34, 50, 101, 152])
def test_ResNetFeature(depth):
    x = torch.rand(1, 3, 224, 224)
    model = ResNetFeature(depth, pretrained=False)
    if depth in [18, 34]:
        assert model(x).size() == (1, 512, 7, 7)
    else:
        assert model(x).size() == (1, 2048, 7, 7)


def test_ResNetClassifier():
    n_class = 40
    len_feature = 512
    x = torch.rand(1, len_feature, 7, 7)
    classfier = ResNetClassifier(n_class, len_feature)
    assert classfier(x).size() == (1, 40)
