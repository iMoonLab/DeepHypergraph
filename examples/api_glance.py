import torch
import torch.nn as nn
import torch.nn.functional as F

from dhg import Graph, Hypergraph


class GCNConv(nn.Module):
    def __init__(self,):
        super().__init__()
        ...
        self.reset_parameters()

    def forward(self, X: torch.Tensor, g: Graph) -> torch.Tensor:
        # apply the trainable parameters ``theta`` to the input ``X``  
        X = self.theta(X)
        # smooth the input ``X`` with the GCN's Laplacian
        X = g.smoothing_with_GCN(X)
        X = F.relu(X)
        return X


class GATConv(nn.Module):
    def __init__(self,):
        super().__init__()
        ...
        self.reset_parameters()

    def forward(self, X: torch.Tensor, g: Graph) -> torch.Tensor:
        # apply the trainable parameters ``theta`` to the input ``X``
        X = self.theta(X)
        # compute attention weights for each edge
        x_for_src = self.atten_src(X)
        x_for_dst = self.atten_dst(X)
        e_atten_score = x_for_src[g.e_src] + x_for_dst[g.e_dst]
        e_atten_score = F.leaky_relu(e_atten_score).squeeze()
        # apply ``e_atten_score`` to each edge in the graph ``g``, aggragete neighbor messages
        #  with ``softmax_then_sum``, and perform vertex->vertex message passing in graph 
        #  with message passing function ``v2v()``
        X = g.v2v(X, aggr="softmax_then_sum", e_weight=e_atten_score)
        X = F.elu(X)
        return X


class HGNNConv(nn.Module):
    def __init__(self,):
        super().__init__()
        ...
        self.reset_parameters()

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        # apply the trainable parameters ``theta`` to the input ``X``
        X = self.theta(X)
        # smooth the input ``X`` with the HGNN's Laplacian
        X = hg.smoothing_with_HGNN(X)
        X = F.relu(X)
        return X


class HGNNPConv(nn.Module):
    def __init__(self,):
        super().__init__()
        ...
        self.reset_parameters()

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        # apply the trainable parameters ``theta`` to the input ``X``
        X = self.theta(X)
        # perform vertex->hyperedge->vertex message passing in hypergraph
        #  with message passing function ``v2v``, which is the combination
        #  of message passing function ``v2e()`` and ``e2v()``
        X = hg.v2v(X, aggr="mean")
        X = F.leaky_relu(X)
        return X
