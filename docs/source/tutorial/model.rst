Building Model
======================

.. hint:: 

    - Author: `Yifan Feng (丰一帆) <https://fengyifan.site/>`_
    - Proof: Xinwei Zhang

DHG provides various spectral-based and spatial-based operations and correlation structures for building your models. 
Here, we divide the model into four categories:

- :ref:`Spectral-based Model <build_spectral_based_model>`
- :ref:`Spatial-based Model <build_spatial_based_model>`
- :ref:`Hybrid Operation Model <build_hybrid_operation_model>`
- :ref:`Hybrid Structure Model <build_hybrid_structure_model>`

.. important:: 

    Before we start, you should have learned some basic usages about learning on :doc:`low-order </start/low_order/index>` and :doc:`high-order </start/high_order/index>` structures with DHG.

.. _build_spectral_based_model:

Building Spectral-based Model
------------------------------

In the following examples, we will build two typical spectral-based models (`GCN <https://arxiv.org/pdf/1609.02907>`_ 
and `HGNN <https://arxiv.org/pdf/1809.09401>`_ ) on graph and hypergraph, respectively.

**Building GCN Model**

The GCN model first computes the Laplacian matrix with expanded adjacency matrix, then performs feature smoothing on vertices for each GCN convolution layer.
For a given :py:class:`graph structure <dhg.Graph>`, the GCN's Laplacian matrix has been pre-computed and stored in :py:attr:`L_GCN <dhg.Graph.L_GCN>` attribute, which can be formed as:

.. math::

    \mathcal{L}_{GCN} = \mathbf{\hat{D}}_v^{-\frac{1}{2}} \mathbf{\hat{A}} \mathbf{\hat{D}}_v^{-\frac{1}{2}},

where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` and :math:`\mathbf{\hat{D}}_{ii} = \sum_j \mathbf{\hat{A}}_{ij}`, 
and :math:`\mathbf{A}` is the adjacency matrix of the graph. Then, the convolution layer of GCN can be formulated as:

.. math::
    \mathbf{X}^{\prime} = \sigma \left( \mathbf{\hat{D}}^{-\frac{1}{2}} \mathbf{\hat{A}}
    \mathbf{\hat{D}}^{-\frac{1}{2}} \mathbf{X} \mathbf{\Theta} \right),

where :math:`\mathbf{X}` is the input vertex feature matrix, :math:`\mathbf{\Theta}` is the learnable parameters of the GCN convolution layer.

DHG also provides function :py:func:`smoothing_with_GCN <dhg.Graph.smoothing_with_GCN>` that applies GCN's Laplacian matrix to smooth vertex features.
Then, the convolution layer of GCN can be implemented as:

.. code-block:: python

    import dhg
    import torch
    import torch.nn as nn

    class GCNConv(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            drop_rate: float = 0.5,
        ):
            super().__init__()
            self.act = nn.ReLU(inplace=True)
            self.drop = nn.Dropout(drop_rate)
            self.theta = nn.Linear(in_channels, out_channels, bias=bias)

        def forward(self, X: torch.Tensor, g: dhg.Graph) -> torch.Tensor:
            X = self.theta(X)
            X_ = g.smoothing_with_GCN(X)
            X_ = self.drop(self.act(X_))
            return X_

Finally, the GCN model can be implemented by stacking multiple GCNConv layers.

**Building HGNN model**

The HGNN model first computes the Laplacian matrix of the given hypergraph, then performs feature smoothing on vertices for each HGNN convolution layer.
For a given :py:class:`hypergraph structure <dhg.Hypergraph>`, the HGNN's Laplacian matrix have been pre-computed 
and stored in :py:attr:`L_HGNN <dhg.Hypergraph.L_HGNN>` attribute, which can be formed as:


.. math::
    
    \mathcal{L}_{HGNN} = \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}}

where :math:`\mathbf{H}` is the hypergraph incidence matrix, :math:`\mathbf{W}_e` is a diagonal hyperedge weight matrix, 
:math:`\mathbf{D}_v` is a diagonal vertex degree matrix, :math:`\mathbf{D}_e` is a diagonal hyperedge degree matrix.
Then, the convolution layer of HGNN can be implemented as:


.. math::
    
    \mathbf{X}^{\prime} = \sigma \left( \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} 
    \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}} \mathbf{X} \mathbf{\Theta} \right).


where :math:`\mathbf{X}` is the input vertex feature matrix, :math:`\mathbf{\Theta}` is the learnable parameters of the HGNN convolution layer.

DHG also provides function :py:func:`smoothing_with_HGNN <dhg.Hypergraph.smoothing_with_HGNN>` that applies HGNN's Laplacian matrix to smooth vertex features.
Then, the convolution layer of HGNN can be implemented as:

.. code-block:: python

    import dhg
    import torch
    import torch.nn as nn

    class HGNNConv(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            drop_rate: float = 0.5,
        ):
            super().__init__()
            self.act = nn.ReLU(inplace=True)
            self.drop = nn.Dropout(drop_rate)
            self.theta = nn.Linear(in_channels, out_channels, bias=bias)

        def forward(self, X: torch.Tensor, hg: dhg.Hypergraph) -> torch.Tensor:
            X = self.theta(X)
            X_ = hg.smoothing_with_HGNN(X)
            X_ = self.drop(self.act(X_))
            return X_

Finally, the HGNN model can be implemented by stacking multiple HGNNConv layers.


.. _build_spatial_based_model:

Building Spatial-based Model
-----------------------------

In the following examples, we will build four different spatial-based models. 

- The first two models are `GraphSAGE <https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf>`_ 
  and `HGNN+ <https://ieeexplore.ieee.org/document/9795251>`_, which perform general message passing 
  from vertex to vertex via edges or from vertex set to vertex set via hyperedges.
- The last two models are `GAT <https://arxiv.org/pdf/1710.10903>`_ and a hypergraph convolution with different hyperedge weights model, 
  which show you how to use **different edge/hyperedge weights** on message aggregation from vertex to vertex or from vertex set to vertex set.


**Building GraphSAGE model**

The GraphSAGE is a general message passing model that combines vertex features and their neighbors' features to form a new vertex feature, 
which can be implemented as follows:

.. code-block:: python

    import dhg
    import torch
    import torch.nn as nn

    class GraphSAGEConv(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            aggr: str = "mean",
            bias: bool = True,
            drop_rate: float = 0.5,
        ):
            super().__init__()
            assert aggr in ["mean"], "Currently, only mean aggregation is supported."
            self.aggr = aggr
            self.act = nn.ReLU(inplace=True)
            self.drop = nn.Dropout(drop_rate)
            if aggr == "mean":
                self.theta = nn.Linear(in_channels * 2, out_channels, bias=bias)
            else:
                raise NotImplementedError()

        def forward(self, X: torch.Tensor, g: dhg.Graph) -> torch.Tensor:
            if self.aggr == "mean":
                X_nbr = g.v2v(X, aggr="mean")
                X = torch.cat([X, X_nbr], dim=1)
            else:
                raise NotImplementedError()
            X_ = self.theta(X)
            X_ = self.drop(self.act(X_))
            return X_

Finally, the GraphSAGE model can be implemented by stacking multiple GraphSAGEConv layers.


**Building HGNN+ model**

The HGNN+ is a general message passing model that passes messages from vertex to hyperedge to vertex, which can be implemented as following:

.. code-block:: python

    import dhg
    import torch
    import torch.nn as nn

    class HGNNPConv(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            drop_rate: float = 0.5,
        ):
            super().__init__()
            self.act = nn.ReLU(inplace=True)
            self.drop = nn.Dropout(drop_rate)
            self.theta = nn.Linear(in_channels, out_channels, bias=bias)

        def forward(self, X: torch.Tensor, hg: dhg.Hypergraph) -> torch.Tensor:
            X = self.theta(X)
            Y = hg.v2e(X, aggr="mean")
            X_ = hg.e2v(Y, aggr="mean")
            X_ = self.drop(self.act(X_))
            return X_

Finally, the HGNN+ model can be implemented by stacking multiple HGNNPConv layers.

**Building GAT model**

DHG provides a special and convenient way to implement weighted neighborhood aggregation from vertex to vertex.
In graph, each edge has its source and target index. 
Given vertex features ``X``, graph ``g``, and linear layers ``atten_src`` and ``atten_dst``, you can compute the edge weight by follows:

.. code-block:: python

    >>> x_for_src = atten_src(X)
    >>> x_for_dst = atten_dst(X)
    >>> e_atten_weight = x_for_src[g.e_src] + x_for_dst[g.e_dst]

Besides, DHG provides ``softmax_then_sum`` aggregation function for neighbor messages aggregation. 
It can normalize the messages from neighbors with ``softmax`` and then sum them to update the center vertex's message.

Then, the GATConv model can be implemented as follows:

.. code-block:: python

    import dhg
    import torch
    import torch.nn as nn

    class GATConv(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            drop_rate: float = 0.5,
            atten_neg_slope: float = 0.2,
        ):
            super().__init__()
            self.atten_dropout = nn.Dropout(drop_rate)
            self.atten_act = nn.LeakyReLU(atten_neg_slope)
            self.act = nn.ELU(inplace=True)
            self.theta = nn.Linear(in_channels, out_channels, bias=bias)
            self.atten_src = nn.Linear(out_channels, 1, bias=False)
            self.atten_dst = nn.Linear(out_channels, 1, bias=False)

        def forward(self, X: torch.Tensor, g: dhg.Graph) -> torch.Tensor:
            X = self.theta(X)
            x_for_src = self.atten_src(X)
            x_for_dst = self.atten_dst(X)
            e_atten_score = x_for_src[g.e_src] + x_for_dst[g.e_dst]
            e_atten_score = self.atten_dropout(self.atten_act(e_atten_score).squeeze())
            X_ = g.v2v(X, aggr="softmax_then_sum", e_weight=e_atten_score)
            X_ = self.act(X_)
            return X_

Finally, the GAT model can be implemented by stacking multiple GATConv layers.


**Building hypergraph convolution with different hyperedge weights model**

Like varying the edge weights in the graph, hyperedge weights can also be varied in the message passing from vertex to hyperedge to vertex.
But the difference is that the hyperedge weights are more complex than the edge weights in the graph.
Due to the two stages (vertex to hyperedge and hyperedge to vertex) of message passing in the hypergraph,
varying the hyperedge weights can also be split into two stages: vertex to hyperedge stage and hyperedge to vertex stage.

- In the first stage, the hyperedge weights are controlled by the **source vertex index** (:py:attr:`v2e_src <dhg.Hypergraph.v2e_src>`) 
  and the **target hyperedge index** (:py:attr:`v2e_dst <dhg.Hypergraph.v2e_dst>`).
- In the second stage, the hyperedge weights are controlled by the **source hyperedge index** (:py:attr:`e2v_src <dhg.Hypergraph.e2v_src>`) 
  and the **target vertex index**  (:py:attr:`e2v_dst <dhg.Hypergraph.e2v_dst>`).

In hypergraph, the two message passing stages are symmetric. 
Thus, the same vertex and hyperedge attention layer can be used in the two stages.
Given the vertex features ``X``, hyperedge features ``Y``, hypergraph ``hg``, and linear layers ``atten_vertex`` and ``atten_hyperedge``, 
you can compute the hyperedge weights for the two stages as follows: 

.. code-block:: python

    >>> x_for_vertex = atten_vertex(X)
    >>> y_for_hyperedge = atten_hyperedge(Y)
    >>> v2e_atten_weight = x_for_vertex[hg.v2e_src] + y_for_hyperedge[hg.v2e_dst]
    >>> e2v_atten_weight = y_for_hyperedge[hg.e2v_src] + x_for_vertex[hg.e2v_dst]

Finally, a hypergraph convolution with different hyperedge weights model can be implemented as follows:

.. code-block:: python

    import dhg
    import torch
    import torch.nn as nn

    class HGATConv(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            drop_rate: float = 0.5,
            atten_neg_slope: float = 0.2,
        ):
            super().__init__()
            self.atten_dropout = nn.Dropout(drop_rate)
            self.atten_act = nn.LeakyReLU(atten_neg_slope)
            self.act = nn.ELU(inplace=True)
            self.theta_vertex = nn.Linear(in_channels, out_channels, bias=bias)
            self.theta_hyperedge = nn.Linear(in_channels, out_channels, bias=bias)
            self.atten_vertex = nn.Linear(out_channels, 1, bias=False)
            self.atten_hyperedge = nn.Linear(out_channels, 1, bias=False)

        def forward(self, X: torch.Tensor, Y: torch.Tensor, hg: dhg.Hypergraph) -> torch.Tensor:
            X = self.theta_vertex(X)
            Y = self.theta_hyperedge(Y)
            x_for_vertex = self.atten_vertex(X)
            y_for_hyperedge = self.atten_hyperedge(Y)
            v2e_atten_score = x_for_vertex[hg.v2e_src] + y_for_hyperedge[hg.v2e_dst]
            e2v_atten_score = y_for_hyperedge[hg.e2v_src] + x_for_vertex[hg.e2v_dst]
            v2e_atten_score = self.atten_dropout(self.atten_act(v2e_atten_score).squeeze())
            e2v_atten_score = self.atten_dropout(self.atten_act(e2v_atten_score).squeeze())
            Y_ = hg.v2e(X, aggr="softmax_then_sum", v2e_weight=v2e_atten_score)
            X_ = hg.e2v(Y_, aggr="softmax_then_sum", e2v_weight=e2v_atten_score)
            X_ = self.act(X_)
            Y_ = self.act(Y_)
            return X_, Y_

Finally, the hypergraph convolution with different hyperedge weights model can be implemented by stacking multiple HGATConv layers.


.. _build_hybrid_operation_model:

Building Hybrid Operation Model
--------------------------------

A hybrid operation model means that the spectral-based convolution or spatial-based convolution can simultaneously be used to embed the correlation into the vertex features.
Given a correlation structure like graph ``g``, you can implement a hybrid operation model as follows:

.. code-block:: python

    import dhg
    import torch
    import torch.nn as nn

    class HOMConv(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            drop_rate: float = 0.5,
        ):
            super().__init__()
            self.act = nn.ReLU(inplace=True)
            self.drop = nn.Dropout(drop_rate)
            self.theta = nn.Linear(in_channels, out_channels, bias=bias)

        def forward(self, X: torch.Tensor, g: dhg.Graph) -> torch.Tensor:
            X = self.theta(X)
            X_spectral = g.smoothing_with_GCN(X)
            X_spatial = g.v2v(X, aggr="mean")
            X_ = (X_spectral + X_spatial) / 2
            X_ = self.drop(self.act(X_))
            return X_

Finally, the hybrid operation model can be implemented by stacking multiple HOMConv layers.

.. _build_hybrid_structure_model:

Building Hybrid Structure Model
-------------------------------------

The hybrid structure model is a model that supports multiple types of correlation structures as input.
Given a set of vertices and vertex feature ``X``, assume that you have constructed low-order structure like graph ``g`` 
and high-order like hypergraph ``hg``. A hybrid structure model can be implemented as follows:

.. code-block:: python

    import dhg
    import torch
    import torch.nn as nn

    class HSMConv(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            bias: bool = True,
            drop_rate: float = 0.5,
        ):
            super().__init__()
            self.act = nn.ReLU(inplace=True)
            self.drop = nn.Dropout(drop_rate)
            self.theta = nn.Linear(in_channels, out_channels, bias=bias)

        def forward(self, X: torch.Tensor, g: dhg.Graph, hg: dhg.Hypergraph) -> torch.Tensor:
            X = self.theta(X)
            X_g = g.v2v(X, aggr="mean")
            X_hg = hg.v2v(X, aggr="mean")
            X_ = (X_g + X_hg) / 2
            X_ = self.drop(self.act(X_))
            return X_

Finally, the hybrid structure model can be implemented by stacking multiple HSMConv layers.

