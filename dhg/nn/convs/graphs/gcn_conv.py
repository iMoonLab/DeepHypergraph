import torch
import torch.nn as nn

from dhg.structure.graphs import Graph


class GCNConv(nn.Module):
    r"""The GCN convolution layer proposed in `Semi-Supervised Classification with Graph Convolutional Networks <https://arxiv.org/pdf/1609.02907>`_ paper (ICLR 2017).

    Matrix Format:

    .. math::
        \mathbf{X}^{\prime} = \sigma \left( \mathbf{\hat{D}}^{-\frac{1}{2}} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-\frac{1}{2}} \mathbf{X} \mathbf{\Theta} \right),

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` and :math:`\mathbf{\hat{D}}_{ii} = \sum_j \mathbf{\hat{A}}_{ij}`.

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``out_channels`` (``int``): :math:`C_{out}` is the number of output channels.
        ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): The dropout probability. If ``dropout <= 0``, the layer will not drop values. Defaults to ``0.5``.
        ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, g: Graph) -> torch.Tensor:
        r"""The forward function.

        Args:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            g (``dhg.Graph``): The graph structure that contains :math:`N` vertices.
        """
        X = self.theta(X)
        X = g.smoothing_with_GCN(X)
        if not self.is_last:
            X = self.act(X)
            if self.bn is not None:
                X = self.bn(X)
            X = self.drop(X)
        return X
