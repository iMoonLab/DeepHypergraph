import torch.nn as nn
from SuperMoon.models import ResNetFeature, ResNetClassifier


class ResNet(nn.Module):
    def __init__(self, n_class, depth=34, pretrained=True):
        super().__init__()

        self.ft_layers = ResNetFeature(depth=depth, pretrained=pretrained)
        self.cls_layers = ResNetClassifier(n_class=n_class, len_feature=self.ft_layers.len_feature)

    def forward(self, x):
        x = self.ft_layers(x)
        x = self.cls_layers(x)

        return x
