Building Model
======================

DHG provides various spectral-based and spatial-based operations and correlation structures for building your models. 
Here, we divide the model into four categories:

- :ref:`Spectral-based Model <build_spectral_based_model>`
- :ref:`Spatial-based Model <build_spatial_based_model>`
- :ref:`Hybrid Operation Model <build_hybrid_operation_model>`
- :ref:`Hybrid Operation on Hybrid Structure Model <build_hybrid_operation_on_hybrid_structure_model>`

.. important:: 

    Before we start, you should have learned some basic usages about learning on :doc:`low-order </start/low_order/index>` and :doc:`high-order </start/high_order/index>` structures with DHG.

.. _build_spectral_based_model:

Building Spectral-based Model
------------------------------

In the following examples, we will build two typical spectral-based models (`GCN <https://arxiv.org/pdf/1609.02907>`_ 
and `HGNN <https://arxiv.org/pdf/1809.09401>`_ ) on simple graph and simple hypergraph, respectively.

**Building GCN Model**

The GCN model first computes the Laplacian matrix with expanded adjacency matrix, then performs feature smoothing on vertices for each GCN convolution layer.
For a given :py:class:`simple graph structure <dhg.Graph>`, the GCN's Laplacian matrix have been pre-computed and stored in :py:attr:`L_GCN <dhg.Graph.L_GCN>` attribute, which can be formed as:

.. math::

    \mathcal{L}_{GCN} = \mathbf{\hat{D}}_v^{-\frac{1}{2}} \mathbf{\hat{A}} \mathbf{\hat{D}}_v^{-\frac{1}{2}},

where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` and :math:`\mathbf{\hat{D}}_{ii} = \sum_j \mathbf{\hat{A}}_{ij}`, 
and :math:`\mathbf{A}` is the adjacency matrix of the simple graph. Then, the convolution layer of GCN can be formulated as:

.. math::
    \mathbf{X}^{\prime} = \sigma \left( \mathbf{\hat{D}}^{-\frac{1}{2}} \mathbf{\hat{A}}
    \mathbf{\hat{D}}^{-\frac{1}{2}} \mathbf{X} \mathbf{\Theta} \right),

where :math:`\mathbf{X}` is the input vertex feature matrix, :math:`\mathbf{\Theta}` is the learnable parameters of the GCN convolution layer.

DHG also provide function :py:func:`smoothing_with_GCN <dhg.Graph.smoothing_with_GCN>` that applies GCN's Laplacian matrix to smooth vertex features.
Then, the convolution layer of GCN can be implamented as:

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
            X = g.smoothing_with_GCN(X)
            X = self.drop(self.act(X))
            return X

Finally, the GCN model can be implemented with stacking multiple GCNConv layers.

**Building HGNN model**

The HGNN model first computes the Laplacian matrix of the given simple hypergraph, then performs feature smoothing on vertices for each HGNN convolution layer.
For a given :py:class:`simple hypergraph structure <dhg.Hypergraph>`, the HGNN's Laplacian matrix have been pre-computed 
and stored in :py:attr:`L_HGNN <dhg.Hypergraph.L_HGNN>` attribute, which can be formed as:


.. math::
    
    \mathcal{L}_{HGNN} = \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}}

where :math:`\mathbf{H}` is the hypergraph incidence matrix, :math:`\mathbf{W}_e` is a diagonal hyperedge weight matrix, 
:math:`\mathbf{D}_v` is a diagonal vertex degree matrix, :math:`\mathbf{D}_e` is a diagonal hyperedge degree matrix.
Then, the convolution layer of HGNN can be implamented as:


.. math::
    
    \mathbf{X}^{\prime} = \sigma \left( \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} 
    \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}} \mathbf{X} \mathbf{\Theta} \right).


where :math:`\mathbf{X}` is the input vertex feature matrix, :math:`\mathbf{\Theta}` is the learnable parameters of the HGNN convolution layer.

DHG also provide function :py:func:`smoothing_with_HGNN <dhg.Hypergraph.smoothing_with_HGNN>` that applies HGNN's Laplacian matrix to smooth vertex features.
Then, the convolution layer of HGNN can be implamented as:

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
            X = hg.smoothing_with_HGNN(X)
            X = self.drop(self.act(X))
            return X

Finally, the HGNN model can be implemented with stacking multiple HGNNConv layers.


.. _build_spatial_based_model:

Building Spatial-based Model
-----------------------------

In the following examples, we will build four different spatial-based models. 

- The first two models are `GraphSAGE <https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf>`_ 
  and `HGNN+ <https://ieeexplore.ieee.org/document/9795251>`_, which perform general message passing 
  from vertex to vertex via edges or from vertex set to vertex set via hyperedges.
- The last two models are `GAT <https://arxiv.org/pdf/1710.10903>`_ and a hypergraph convolution with different hyperedge weights model, 
  which show you how to use **different edge/hyperedge weights** on message aggretaion from vertex to vertex or from vertex set to vertex set.


**Building GraphSAGE model**

The GraphSAGE is a general message passing model that conbines vertex feature and their neighbors' features to form a new vertex feature, 
which can be implamented as follows:

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
            X = self.theta(X)
            X = self.drop(self.act(X))
            return X


**Building HGNN+ model**

Comming soon...

**Building GAT model**

Comming soon...


**Building hypergraph convolution with different hyperedge weights model**

Comming soon...



.. _build_hybrid_operation_model:

Building Hybrid Operation Model
--------------------------------

Comming soon...

.. _build_hybrid_operation_on_hybrid_structure_model:

Building Hybrid Operation on Hybrid Structure Model
-----------------------------------------------------

Comming soon...


.. 1. select your correlation
.. 2. determinate your type

.. Building Spectral-Based Model
.. ---------------------------


.. GCN
.. ++++++

.. HGNN
.. +++++++

.. Building Spatial-Based Model
.. -----------------------------

.. GAT 
.. +++++

.. GraphSAGE
.. +++++++++++++++


.. HGNNP
.. ++++++++++



.. Examples
.. --------------
