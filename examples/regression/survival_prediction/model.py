import torch.nn.functional as F
from torch import nn
import torch

from SuperMoon.conv import HyConv


class HGNN_reg(nn.Module):
    def __init__(self, in_ch, n_target, hiddens=[16], dropout=0.5) -> None:
        super().__init__()
        self.dropout = dropout
        _in = in_ch
        self.hyconvs = []
        for _h in hiddens:
            _out = _h
            self.hyconvs.append(HyConv(_in, _out))
            _in = _out
        self.hyconvs = nn.ModuleList(self.hyconvs)
        self.last_fc = nn.Linear(_in, n_target)

    def forward(self, x, H, hyedge_weight=None):
        for hyconv in self.hyconvs:
            x = hyconv(x, H)
            x = F.leaky_relu(x, inplace=True)
            x = F.dropout(x, self.dropout)

        # N x C -> C
        x = x.mean(dim=0)

        # C -> 1 x C
        x = x.unsqueeze(0)
        # 1 x C -> 1 x n_target
        x = self.last_fc(x)
        # 1 x n_target -> n_target
        x = x.squeeze(0)

        return torch.sigmoid(x)
