import torch.nn as nn
import torch.nn.functional as F

from SuperMoon.conv import HyConv
from SuperMoon.hyedge import neighbor_distance
from SuperMoon.models import ResNetFeature, ResNetClassifier


class ResNet_HGNN(nn.Module):
    def __init__(self, n_class, depth, k_nearest, hiddens=[512], dropout=0.5, pretrained=True):
        super().__init__()
        self.dropout = dropout
        self.k_nearest = k_nearest
        self.ft_layers = ResNetFeature(depth=depth, pretrained=pretrained)

        # hypergraph convolution for feature refine
        self.hyconvs = []
        dim_in = self.ft_layers.len_feature
        for h in hiddens:
            dim_out = h
            self.hyconvs.append(HyConv(dim_in, dim_out))
            dim_in = dim_out
        self.hyconvs = nn.ModuleList(self.hyconvs)

        self.cls_layers = ResNetClassifier(n_class=n_class, len_feature=dim_in)

    def forward(self, x):
        x = self.ft_layers(x)

        assert x.size(0) == 1, 'when construct hypergraph, only support batch size = 1!'
        x = x.view(x.size(1), x.size(2) * x.size(3))
        # -> N x C
        x = x.permute(1, 0)
        H = neighbor_distance(x, k_nearest=self.k_nearest)
        # Hypergraph Convs
        for hyconv in self.hyconvs:
            x = hyconv(x, H)
            x = F.leaky_relu(x, inplace=True)
            x = F.dropout(x, self.dropout, training=self.training)
        # N x C -> 1 x C x N
        x = x.permute(1, 0).unsqueeze(0)

        x = self.cls_layers(x)

        return x
