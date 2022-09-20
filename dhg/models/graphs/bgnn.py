from typing import Tuple

import torch
import torch.nn as nn

from dhg.structure.graphs import BiGraph


class BGNN_Adv(nn.Module):
    r"""The BGNN-Adv model proposed in `Cascade-BGNN: Toward Efficient Self-supervised Representation Learning on Large-scale Bipartite Graphs <https://arxiv.org/pdf/1906.11994.pdf>`_ paper (TNNLS 2020).

    Args:
        ``u_dim`` (``int``): The dimension of the vertex feature in set :math:`U`.
        ``v_dim`` (``int``): The dimension of the vertex feature in set :math:`V`.
        ``layer_depth`` (``int``): The depth of layers.
    """

    def __init__(self, u_dim: int, v_dim: int, layer_depth: int = 3,) -> None:

        super().__init__()
        self.layer_depth = layer_depth
        self.layers = nn.ModuleList()

        for _idx in range(layer_depth):
            if _idx % 2 == 0:
                self.layers.append(nn.Linear(v_dim, u_dim))
            else:
                self.layers.append(nn.Linear(u_dim, v_dim))

    def forward(self, X_u: torch.Tensor, X_v: torch.Tensor, g: BiGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""The forward function.

        Args:
            ``X_u`` (``torch.Tensor``): The feature matrix of vertices in set :math:`U`.
            ``X_v`` (``torch.Tensor``): The feature matrix of vertices in set :math:`V`.
            ``g`` (``BiGraph``): The bipartite graph.
        """
        last_X_u, last_X_v = X_u, X_v
        for _idx in range(self.layer_depth):
            if _idx % 2 == 0:
                _tmp = self.layers[_idx](last_X_v)
                last_X_u = g.v2u(_tmp, aggr="sum")
            else:
                _tmp = self.layers[_idx](last_X_u)
                last_X_v = g.u2v(_tmp, aggr="sum")
        return last_X_u

    def train_with_cascaded(self):
        pass

    def train_with_end2end(self):
        pass


class BGNN_MLP(nn.Module):
    r"""The BGNN-MLP model proposed in `Cascade-BGNN: Toward Efficient Self-supervised Representation Learning on Large-scale Bipartite Graphs <https://arxiv.org/pdf/1906.11994.pdf>`_ paper (TNNLS 2020).

    Args:
        ``u_dim`` (``int``): The dimension of the vertex feature in set :math:`U`.
        ``v_dim`` (``int``): The dimension of the vertex feature in set :math:`V`.
        ``layer_depth`` (``int``): The depth of layers.
    """

    def __init__(self, u_dim: int, v_dim: int, layer_depth: int = 3,) -> None:

        super().__init__()
        self.layer_depth = layer_depth
        self.layers = nn.ModuleList()

        for _idx in range(layer_depth):
            if _idx % 2 == 0:
                self.layers.append(nn.Linear(v_dim, u_dim))
            else:
                self.layers.append(nn.Linear(u_dim, v_dim))

    def forward(self, X_u: torch.Tensor, X_v: torch.Tensor, g: BiGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""The forward function.

        Args:
            ``X_u`` (``torch.Tensor``): The feature matrix of vertices in set :math:`U`.
            ``X_v`` (``torch.Tensor``): The feature matrix of vertices in set :math:`V`.
            ``g`` (``BiGraph``): The bipartite graph.
        """
        last_X_u, last_X_v = X_u, X_v
        for _idx in range(self.layer_depth):
            if _idx % 2 == 0:
                _tmp = self.layers[_idx](last_X_v)
                last_X_u = g.v2u(_tmp, aggr="sum")
            else:
                _tmp = self.layers[_idx](last_X_u)
                last_X_v = g.u2v(_tmp, aggr="sum")
        return last_X_u

    def train_with_cascaded(self):
        pass

    def train_with_end2end(self):
        pass
