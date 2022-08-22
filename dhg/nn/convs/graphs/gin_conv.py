import torch
import torch.nn as nn

from dhg.structure.graphs import Graph


class GINConv(nn.Module):
    r"""The GIN convolution layer proposed in `How Powerful are Graph Neural Networks? <https://arxiv.org/pdf/1810.00826>`_ paper (ICLR 2019).

    Sparse Format:

    .. math::
        \mathbf{x}^{\prime}_i = MLP \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right).

    Matrix Format:

    .. math::
        \mathbf{X}^{\prime} = MLP \left( \left( \mathbf{A} +
        (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right).

    Args:
        ``MLP`` (``nn.Module``): The neural network to be applied after message passing, i.e. ``nn.Linear``, ``nn.Sequential``.
        ``eps`` (``float``): The epsilon value.
        ``train_eps`` (``bool``): If set to ``True``, the epsilon value will be trainable.
    """

    def __init__(self, MLP: nn.Module, eps: float = 0.0, train_eps: bool = False):
        super().__init__()
        self.MLP = MLP
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.eps = eps

    def forward(self, X: torch.Tensor, g: Graph) -> torch.Tensor:
        r"""The forward function.

        Args:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N_v, C_{in})`.
            g (``dhg.Graph``): The graph structure that contains :math:`N_v` vertices.
        """
        X = (1 + self.eps) * X + g.v2v(X, aggr="sum")
        X = self.MLP(X)
        return X
