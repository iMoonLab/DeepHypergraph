import torch
import torch.nn as nn

import dhg
from dhg.nn import GATConv, MultiHeadWrapper


class GAT(nn.Module):
    r"""The GAT model proposed in `Graph Attention Networks <https://arxiv.org/pdf/1710.10903>`_ paper (ICLR 2018).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``num_heads`` (``int``): The Number of attention head in each layer.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): The dropout probability. Defaults to ``0.5``.
        ``atten_neg_slope`` (``float``): Hyper-parameter of the ``LeakyReLU`` activation of edge attention. Defaults to 0.2.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        num_heads: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        atten_neg_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.drop_layer = nn.Dropout(drop_rate)
        self.multi_head_layer = MultiHeadWrapper(
            num_heads,
            "concat",
            GATConv,
            in_channels=in_channels,
            out_channels=hid_channels,
            use_bn=use_bn,
            drop_rate=drop_rate,
            atten_neg_slope=atten_neg_slope,
        )
        # The original implementation has applied activation layer after the final layer.
        # Thus, we donot set ``is_last`` to ``True``.
        self.out_layer = GATConv(
            hid_channels * num_heads,
            num_classes,
            use_bn=use_bn,
            drop_rate=drop_rate,
            atten_neg_slope=atten_neg_slope,
            is_last=False,
        )

    def forward(self, X: torch.Tensor, g: "dhg.Graph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``g`` (``dhg.Graph``): The graph structure that contains :math:`N` vertices.
        """
        X = self.drop_layer(X)
        X = self.multi_head_layer(X=X, g=g)
        X = self.drop_layer(X)
        X = self.out_layer(X, g)
        return X
