import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

import dhg
from dhg.nn import HGNNPConv


class HGNNP(nn.Module):
    r"""The HGNN :sup:`+` model proposed in `HGNN+: General Hypergraph Neural Networks <https://ieeexplore.ieee.org/document/9795251>`_ paper (IEEE T-PAMI 2022).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to ``0.5``.
        ``use_checkpoint`` (``bool``): If set to ``True``, use gradient checkpointing to save memory at the cost of computation. Defaults to ``False``.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.layers = nn.ModuleList()
        self.layers.append(
            HGNNPConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        )
        self.layers.append(
            HGNNPConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)
        )

    def forward(self, X: torch.Tensor, hg: "dhg.Hypergraph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                X = checkpoint(layer, X, hg, use_reentrant=False)
            else:
                X = layer(X, hg)
        return X
