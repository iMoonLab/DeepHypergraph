构建模型
======================

.. hint:: 

    - 作者:  `丰一帆 <https://fengyifan.site/>`_
    - 翻译:  颜杰龙
    - 校对： `丰一帆 <https://fengyifan.site/>`_ 、张欣炜

DHG提供多种基于谱域和基于空域的操作及关联结构来构建模型。
在这里，我们把模型分为四类：

- :ref:`基于谱域的模型 <zh_build_spectral_based_model>`
- :ref:`基于空域的模型 <zh_build_spatial_based_model>`
- :ref:`混合操作模型 <zh_build_hybrid_operation_model>`
- :ref:`混合关联结构模型 <zh_build_hybrid_structure_model>`

.. important:: 

    在开始之前，您需要了解一些使用DHG在 :doc:`低阶 </start/low_order/index>` 和 :doc:`高阶 </start/high_order/index>` 关联结构上学习的基础用法。

.. _zh_build_spectral_based_model:

构建基于谱域的模型
------------------------------

在如下的例子中，我们分别在图和超图上构建基于谱域的典型模型（`GCN <https://arxiv.org/pdf/1609.02907>`_ 和 `HGNN <https://arxiv.org/pdf/1809.09401>`_ ）。

**构建GCN模型**

GCN模型首先计算扩展邻接矩阵的拉普拉斯矩阵，然后在每一个GCN卷积层对顶点特征平滑。
给定 :py:class:`图关联结构 <dhg.Graph>` ， GCN的拉普拉斯矩阵会被预先计算并且存储在属性 :py:attr:`L_GCN <dhg.Graph.L_GCN>` 中，其计算公式为：

.. math::

    \mathcal{L}_{GCN} = \mathbf{\hat{D}}_v^{-\frac{1}{2}} \mathbf{\hat{A}} \mathbf{\hat{D}}_v^{-\frac{1}{2}},

其中 :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` ， :math:`\mathbf{\hat{D}}_{ii} = \sum_j \mathbf{\hat{A}}_{ij}` ，
:math:`\mathbf{A}` 为图的邻接矩阵。
然后，GCN的卷积层可以表示为：

.. math::
    \mathbf{X}^{\prime} = \sigma \left( \mathbf{\hat{D}}^{-\frac{1}{2}} \mathbf{\hat{A}}
    \mathbf{\hat{D}}^{-\frac{1}{2}} \mathbf{X} \mathbf{\Theta} \right),

其中 :math:`\mathbf{X}` 为输入的顶点特征矩阵， :math:`\mathbf{\Theta}` 为GCN卷积层的可学习参数。

DHG也提供 :py:func:`smoothing_with_GCN <dhg.Graph.smoothing_with_GCN>`  函数使用GCN的拉普拉斯矩阵来对顶点特征平滑。
然后，GCN的卷积层可以实现为：

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

最后，通过多个GCNConv层的叠加可以实现GCN模型。

**构建HGNN模型**

HGNN模型首先计算给定超图的拉普拉斯矩阵，然后在每一个HGNN卷积层对顶点特征平滑。
给定 :py:class:`超图关联结构 <dhg.Hypergraph>`，HGNN的拉普拉斯矩阵会被预先计算并且存储在属性 :py:attr:`L_HGNN <dhg.Hypergraph.L_HGNN>` 中，其计算公式为：


.. math::
    
    \mathcal{L}_{HGNN} = \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}}

其中 :math:`\mathbf{H}` 为超图关联矩阵， :math:`\mathbf{W}_e` 为超边权重对角矩阵，
:math:`\mathbf{D}_v` 为顶点度数对角矩阵， :math:`\mathbf{D}_e` 为超边度数对角矩阵。
然后，HGNN的卷积层可以实现为：

.. math::
    
    \mathbf{X}^{\prime} = \sigma \left( \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} 
    \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}} \mathbf{X} \mathbf{\Theta} \right).


其中 :math:`\mathbf{X}` 为输入的顶点特征矩阵， :math:`\mathbf{\Theta}` 为HGNN卷积层的可学习参数。

DHG也提供 :py:func:`smoothing_with_HGNN <dhg.Hypergraph.smoothing_with_HGNN>` 函数使用HGNN的拉普拉斯矩阵来对顶点特征平滑。
然后，HGNN的卷积层可以实现为：

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

最后，通过多个HGNNConv层的叠加可以实现HGNN模型。


.. _zh_build_spatial_based_model:

构建基于空域的模型
-----------------------------

在如下的例子中，我们将会构建四种不同基于空域的模型。

- 前两个模型为 `GraphSAGE <https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf>`_
  和 `HGNN+ <https://ieeexplore.ieee.org/document/9795251>`_ ，其执行通过边从顶点到顶点或者通过超边从顶点集到顶点集的通用消息传递。
- 后两个模型为 `GAT <https://arxiv.org/pdf/1710.10903>`_ 和 具有不同超边权重的超图卷积模型，
  其展示了如何使用 **不同的边/超边权重** 来从顶点到顶点或者从顶点集到顶点集进行消息聚合。


**构建GraphSAGE模型**

GraphSAGE是一个通用的消息传递模型，其通过结合顶点特征以及它们邻居的特征来形成新的顶点特征，
其可以用如下方式实现：

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

最后，通过多个GraphSAGEConv层的叠加可以实现GraphSAGE模型。


**构建HGNN+模型**

HGNN+是一个通用的消息传递模型，其以从顶点到超边再到顶点的方式传播消息，可以用如下方式实现：

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

最后，通过多个HGNNPConv层的叠加可以实现HGNN+模型。

**构建GAT模型**

DHG提供一种特殊且方便的方式来实现从顶点到顶点的加权邻域聚合。
在图中，每条边有其源点和汇点索引。
给定顶点特征 ``X`` 、图 ``g`` 以及线性层 ``atten_src`` 和 ``atten_dst`` ，可以用以下方式计算边权：

.. code-block:: python

    >>> x_for_src = atten_src(X)
    >>> x_for_dst = atten_dst(X)
    >>> e_atten_weight = x_for_src[g.e_src] + x_for_dst[g.e_dst]

除此之外，DHG提供 ``softmax_then_sum`` 聚合函数用于邻域消息聚合。
该函数可以使用 ``softmax`` 对邻居的消息归一化，然后将它们相加来更新中心顶点的消息。

然后，GATConv模型可以实现为：

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

最后，通过多个GATConv层的叠加可以实现GAT模型。


**构建具有不同超边权重的超图卷积模型**

像在图中改变权重一样，超边权重也可以在从顶点到超边再到顶点的消息传递中改变。
但不同的是，超边权重比图中的边权更复杂。
由于超图中的消息传递分为两个阶段（从顶点到超边和从超边到顶点），
改变超边权重也可以分为两个阶段：从顶点到超边阶段和从超边到顶点阶段。

- 在第一阶段，超边权重由 **源点索引** (:py:attr:`v2e_src <dhg.Hypergraph.v2e_src>`)和 **目标超边索引** (:py:attr:`v2e_dst <dhg.Hypergraph.v2e_dst>`) 控制。
- 在第二阶段，超边权重由 **源超边索引** (:py:attr:`e2v_src <dhg.Hypergraph.e2v_src>`)和 **目标顶点索引** (:py:attr:`e2v_dst <dhg.Hypergraph.e2v_dst>`)控制。

在超图中，消息传递的两阶段是对称的。
因此，两阶段中可以使用相同的顶点和超边注意力层，
给定顶点特征 ``X`` 、 超边特征 ``Y`` 、超图 ``hg`` 和线性层 ``atten_vertex`` 及 ``atten_hyperedge`` ，
可以用以下方式计算两阶段超边边权：

.. code-block:: python

    >>> x_for_vertex = atten_vertex(X)
    >>> y_for_hyperedge = atten_hyperedge(Y)
    >>> v2e_atten_weight = x_for_vertex[hg.v2e_src] + y_for_hyperedge[hg.v2e_dst]
    >>> e2v_atten_weight = y_for_hyperedge[hg.e2v_src] + x_for_vertex[hg.e2v_dst]

然后，具有不同超边权重的超图卷积模型可以实现为：

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

最后，通过多个HGATConv层的叠加可以实现具有不同超边权重的超图卷积模型。


.. _zh_build_hybrid_operation_model:

构建混合操作模型
--------------------------------

混合操作模型意味着可以同时使用基于谱域的卷积或基于空域的卷积来将相关性嵌入到顶点特征中。
给定关联结构如图 ``g`` ，混合操作模型可以实现为：

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

最后，通过多个HOMConv层的叠加可以实现混合操作模型。

.. _zh_build_hybrid_structure_model:

构建混合关联结构模型
-------------------------------------

混合关联结构模型是支持多类型关联结构作为输入的模型。
给定顶点集和顶点特征 ``X`` ，假设您已经构造了低阶关联结构（如图 ``g`` ） 和高阶关联结构（如超图 ``hg`` ），
混合关联结构模型可以实现为：

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

最后，通过多个HSMConv层的叠加可以实现混合关联结构模型。

