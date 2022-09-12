from typing import Optional

import torch
import torch.nn as nn

from dhg.structure.graphs import Graph
from dhg.structure.hypergraphs import Hypergraph


class HyperGCNConv(nn.Module):
    r"""The HyperGCN convolution layer proposed in `HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs <https://papers.nips.cc/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf>`_ paper (NeurIPS 2019).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``out_channels`` (int): :math:`C_{out}` is the number of output channels.
        ``use_mediator`` (``str``): Whether to use mediator to transform the hyperedges to edges in the graph. Defaults to ``False``.
        ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): If set to a positive number, the layer will use dropout. Defaults to ``0.5``.
        ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_mediator: bool = False,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.use_mediator = use_mediator
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(
        self, X: torch.Tensor, hg: Hypergraph, cached_g: Optional[Graph] = None
    ) -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
            ``cached_g`` (``dhg.Graph``): The pre-transformed graph structure from the hypergraph structure that contains :math:`N` vertices. If not provided, the graph structure will be transformed for each forward time. Defaults to ``None``.
        """
        X = self.theta(X)
        if self.bn is not None:
            X = self.bn(X)
        if cached_g is None:
            g = Graph.from_hypergraph_hypergcn(
                hg, X, self.use_mediator, device=X.device
            )
            X = g.smoothing_with_GCN(X)
        else:
            X = cached_g.smoothing_with_GCN(X)
        if not self.is_last:
            X = self.drop(self.act(X))
        return X
