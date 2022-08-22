import torch
import torch.nn as nn

import dhg
from dhg.nn import GCNConv


class GCN(nn.Module):
    r"""The GCN model proposed in `Semi-Supervised Classification with Graph Convolutional Networks <https://arxiv.org/pdf/1609.02907>`_ paper (ICLR 2017).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): Dropout ratio. Defaults to ``0.5``.
    """
    def __init__(self, in_channels: int,
                 hid_channels: int,
                 num_classes: int,
                 use_bn: bool = False,
                 drop_rate: float = 0.5) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate))
        self.layers.append(GCNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True))

    def forward(self, X: torch.Tensor, g: "dhg.Graph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``g`` (``dhg.Graph``): The graph structure that contains :math:`N` vertices.
        """        
        for layer in self.layers:
            X = layer(X, g)
        return X
