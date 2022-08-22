import torch
import torch.nn as nn

from dhg.nn import GraphSAGEConv
from dhg.structure.graphs import Graph


class GraphSAGE(nn.Module):
    r"""The GraphSAGE model proposed in `Inductive Representation Learning on Large Graphs <https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf>`_ paper (NIPS 2017).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``aggr`` (``str``): The neighbor aggregation method. Currently, only mean aggregation is supported. Defaults to "mean".
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): The dropout probability. Defaults to 0.5.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        aggr: str = "mean",
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphSAGEConv(in_channels, hid_channels, aggr=aggr, use_bn=use_bn, drop_rate=drop_rate))
        self.layers.append(GraphSAGEConv(hid_channels, num_classes, aggr=aggr, use_bn=use_bn, is_last=True))

    def forward(self, X: torch.Tensor, g: "Graph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``g`` (``dhg.Graph``): The graph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            X = layer(X, g)
        return X
