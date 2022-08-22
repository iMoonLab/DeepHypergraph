import torch
import torch.nn as nn

from dhg.structure.hypergraphs import Hypergraph


class HNHNConv(nn.Module):
    r"""The HNHN convolution layer proposed in `HNHN: Hypergraph Networks with Hyperedge Neurons <https://arxiv.org/pdf/2006.12278.pdf>`_ paper (ICML 2020).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``out_channels`` (int): :math:`C_{out}` is the number of output channels.
        ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): If set to a positive number, the layer will use dropout. Defaults to ``0.5``.
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
        self.theta_v2e = nn.Linear(in_channels, out_channels, bias=bias)
        self.theta_e2v = nn.Linear(out_channels, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        r"""The forward function.

        Args:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(|\mathcal{V}|, C_{in})`.
            hg (``dhg.Hypergraph``): The hypergraph structure that contains :math:`|\mathcal{V}|` vertices.
        """
        # v -> e
        X = self.theta_v2e(X)
        if self.bn is not None:
            X = self.bn(X)
        Y = self.act(hg.v2e(X, aggr="mean"))
        # e -> v
        Y = self.theta_e2v(Y)
        X = hg.e2v(Y, aggr="mean")
        if not self.is_last:
            X = self.drop(self.act(X))
        return X
