import torch
import torch.nn as nn

from dhg.structure.graphs import Graph


class GraphSAGEConv(nn.Module):
    r"""The GraphSAGE convolution layer proposed in `Inductive Representation Learning on Large Graphs <https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf>`_ paper (NeurIPS 2017).

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j
        
    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``out_channels`` (``int``): :math:`C_{out}` is the number of output channels.
        ``aggr`` (``str``): The neighbor aggregation method. Currently, only mean aggregation is supported. Defaults to "mean".
        ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): The dropout probability. If ``dropout <= 0``, the layer will not drop values. Defaults to ``0.5``.
        ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = "mean",
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ):
        super().__init__()
        assert aggr in ["mean"], "Currently, only mean aggregation is supported."
        self.aggr = aggr
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        if aggr == "mean":
            self.theta = nn.Linear(in_channels * 2, out_channels, bias=bias)
        else:
            raise NotImplementedError()

    def forward(self, X: torch.Tensor, g: Graph) -> torch.Tensor:
        r"""The forward function.

        Args:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N_v, C_{in})`.
            g (``dhg.Graph``): The graph structure that contains :math:`N_v` vertices.
        """
        if self.aggr == "mean":
            X_nbr = g.v2v(X, aggr="mean")
            X = torch.cat([X, X_nbr], dim=1)
        else:
            raise NotImplementedError()
        X = self.theta(X)
        if self.bn is not None:
            X = self.bn(X)
        if not self.is_last:
            X = self.drop(self.act(X))
        return X
