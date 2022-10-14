构建关联结构
===================================

.. hint:: 

    - 作者:  `丰一帆 <https://fengyifan.site/>`_
    - 翻译:  颜杰龙
    - 校对： `丰一帆 <https://fengyifan.site/>`_ 、张欣炜

关联结构是 **DHG** 的核心。
在这一节中，我们介绍不同关联结构的基本构建方法以及它们的一些结构转换函数，如：

- 将高阶关联结构简化为低阶关联结构
- 将低阶关联结构提升到高阶关联结构

构建低阶关联结构
-----------------------

当前版本DHG实现的低阶关联结构包括图、有向图、二分图，以后我们将增加更多的低阶关联结构。

.. _zh_build_graph:

构建图
+++++++++++++++++++++++

`图 <https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)>`_ 为无自环无重边的图，边 ``(x, y)`` 和边 ``(y, x)`` 视为相同的边。
可以使用如下方式构建：

- 边列表 (**默认**) :py:class:`dhg.Graph`
- 邻接列表 :py:meth:`dhg.Graph.from_adj_list`
- 从超图关联结构简化而来
  
  - 星扩展 :py:meth:`dhg.Graph.from_hypergraph_star`
  - 团扩展 :py:meth:`dhg.Graph.from_hypergraph_clique`
  - 基于 `HyperGCN <https://arxiv.org/pdf/1809.02589.pdf>`_ 的扩展 :py:meth:`dhg.Graph.from_hypergraph_hypergcn`

常用方法
^^^^^^^^^^^^^^^^^^^

使用 :py:class:`dhg.Graph` 类 **从边列表构建一个图**

.. code-block:: python

    >>> import dhg
    >>> g = dhg.Graph(5, [(0, 1), (0, 2), (1, 2), (3, 4)])
    >>> g
    Graph(num_v=5, num_e=4)
    >>> g.v
    [0, 1, 2, 3, 4]
    >>> g.e
    ([(0, 1), (0, 2), (1, 2), (3, 4)], [1.0, 1.0, 1.0, 1.0])
    >>> e_list, e_weight = g.e
    >>> e_list
    [(0, 1), (0, 2), (1, 2), (3, 4)]
    >>> e_weight
    [1.0, 1.0, 1.0, 1.0]
    >>> g.e_both_side
    ([(0, 1), (0, 2), (1, 2), (3, 4), (1, 0), (2, 0), (2, 1), (4, 3)], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    >>> # print the adjacency matrix
    >>> g.A.to_dense()
    tensor([[0., 1., 1., 0., 0.],
            [1., 0., 1., 0., 0.],
            [1., 1., 0., 0., 0.],
            [0., 0., 0., 0., 1.],
            [0., 0., 0., 1., 0.]])

.. image:: ../../_static/img/build_structure_graph_from_edge_list.png
    :align: center
    :alt: Customize size
    :height: 400px


可以发现图的邻接矩阵是一个对称矩阵。
:py:attr:`g.e <dhg.Graph.e>` 属性会返回两个列表的元组，第一个列表是边列表，第二个列表是每条边的权重。
:py:attr:`g.e_both_side <dhg.Graph.e_both_side>` 属性会返回图里所有边及其对应的对称形式。

.. important::

    图里的边是无序对，也就意味着边 ``(0, 1)`` 和边 ``(1, 0)`` 是同一条边。
    增加边 ``(0, 1)`` 和边 ``(1, 0)`` 等同于增加边 ``(0, 1)`` 两次。


.. code-block:: python

    >>> g = dhg.Graph(5, [(0, 1), (0, 2), (2, 0), (3, 4)])
    >>> g.e
    ([(0, 1), (0, 2), (3, 4)], [1.0, 1.0, 1.0])
    >>> g.add_edges([(0, 1), (4, 3)])
    >>> g.e
    ([(0, 1), (0, 2), (3, 4)], [1.0, 1.0, 1.0])


.. note:: 

    如果增加的边有重边，这些重边将根据指定的 ``merge_op`` 合并。

.. code-block:: python

    >>> g = dhg.Graph(5, [(0, 1), (0, 2), (0, 2), (3, 4)], merge_op="mean")
    >>> g.e
    ([(0, 1), (0, 2), (3, 4)], [1.0, 1.0, 1.0])
    >>> g = dhg.Graph(5, [(0, 1), (0, 2), (0, 2), (3, 4)], merge_op="sum")
    >>> g.e
    ([(0, 1), (0, 2), (3, 4)], [1.0, 2.0, 1.0])
    >>> g.add_edges([(1, 0), (3, 2)], merge_op="mean")
    >>> g.e
    ([(0, 1), (0, 2), (3, 4), (2, 3)], [1.0, 2.0, 1.0, 1.0])
    >>> g.add_edges([(1, 0), (2, 3)], merge_op="sum")
    >>> g.e
    ([(0, 1), (0, 2), (3, 4), (2, 3)], [2.0, 2.0, 1.0, 2.0])


如果你分别设置 ``merge_op`` 为 ``mean`` 和 ``sum`` ，你会发现最后一条边的权重分别是 ``1.0`` 和 ``2.0`` 。

使用 :py:meth:`dhg.Graph.from_adj_list` 函数 **从邻接列表构建一个图**

邻接列表是一个嵌套列表，每一个内层列表包含两个部分。
第一个部分是列表的 **第一个元素** ，代表源点的索引。
第二个部分是列表的 **剩余元素** ，代表汇点的索引。
例如，假设包含5个顶点的图，其邻接列表为：

.. code-block:: 

    [[0, 1, 2], [0, 3], [1, 2], [3, 4]]

那么，该邻接列表转换的边列表为：

.. code-block:: 

    [(0, 1), (0, 2), (0, 3), (1, 2), (3, 4)]

我们可以根据邻接列表构建图，如：

.. code-block:: python

    >>> g = dhg.Graph.from_adj_list(5, [[0, 1, 2], [1, 3], [4, 3, 0, 2, 1]])
    >>> g.e
    ([(0, 1), (0, 2), (1, 3), (3, 4), (0, 4), (2, 4), (1, 4)], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    >>> g.A.to_dense()
    tensor([[0., 1., 1., 0., 1.],
            [1., 0., 0., 1., 1.],
            [1., 0., 0., 0., 1.],
            [0., 1., 0., 0., 1.],
            [1., 1., 1., 1., 0.]])


.. image:: ../../_static/img/build_structure_graph_from_adj.png
    :align: center
    :alt: Customize size
    :height: 400px


从高阶关联结构简化而来
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

我们首先定义一个超图：

.. code-block:: python

    >>> hg = dhg.Hypergraph(5, [(0, 1, 2), (1, 3, 2), (1, 2), (0, 3, 4)])
    >>> hg.e
    ([(0, 1, 2), (1, 2, 3), (1, 2), (0, 3, 4)], [1.0, 1.0, 1.0, 1.0])
    >>> # print hypergraph incidence matrix
    >>> hg.H.to_dense()
    tensor([[1., 0., 0., 1.],
            [1., 1., 1., 0.],
            [1., 1., 1., 0.],
            [0., 1., 0., 1.],
            [0., 0., 0., 1.]])

**星扩展** :py:meth:`dhg.Graph.from_hypergraph_star`

星扩展会在图内将超图的超边视为虚拟顶点。
每一个虚拟顶点连接超边内所有的顶点。
:py:meth:`dhg.Graph.from_hypergraph_star` 函数会返回两个值。
第一个值是简化得到的图，第二个值为表示顶点是否为实际顶点的 ``vertex mask`` 。
``vertex mask`` 为 ``True`` 代表着该顶点为实际顶点，为 ``False`` 表示顶点为从超边转换的虚拟顶点。

.. code-block:: python

    >>> g, v_mask = dhg.Graph.from_hypergraph_star(hg)
    >>> g
    Graph(num_v=9, num_e=11)
    >>> g.e[0]
    [(0, 5), (0, 8), (1, 5), (1, 6), (1, 7), (2, 5), (2, 6), (2, 7), (3, 6), (3, 8), (4, 8)]
    >>> v_mask
    tensor([ True,  True,  True,  True,  True, False, False, False, False])
    >>> g.A.to_dense()
    tensor([[0., 0., 0., 0., 0., 1., 0., 0., 1.],
            [0., 0., 0., 0., 0., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 1.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1.],
            [1., 1., 1., 0., 0., 0., 0., 0., 0.],
            [0., 1., 1., 1., 0., 0., 0., 0., 0.],
            [0., 1., 1., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 1., 1., 0., 0., 0., 0.]])

.. image:: ../../_static/img/build_structure_graph_from_star_expansion.png
    :align: center
    :alt: Customize size
    :height: 400px


**团扩展** :py:meth:`dhg.Graph.from_hypergraph_clique`

和星扩展不同的是，团扩展不会在图内增加虚拟顶点。
它将超图内的超边简化为图的边。
对于每一条超边，星扩展会增加边把超边内的顶点两两连接。

.. code-block:: python

    >>> g = dhg.Hypergraph.from_hypergraph_clique(hg)
    >>> g = dhg.Graph.from_hypergraph_clique(hg)
    >>> g
    Graph(num_v=5, num_e=8)
    >>> g.e
    ([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (2, 3), (3, 4)], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    >>> g.A.to_dense()
    tensor([[0., 1., 1., 1., 1.],
            [1., 0., 1., 1., 0.],
            [1., 1., 0., 1., 0.],
            [1., 1., 1., 0., 1.],
            [1., 0., 0., 1., 0.]])

.. image:: ../../_static/img/build_structure_graph_from_clique_expansion.png
    :align: center
    :alt: Customize size
    :height: 400px


**基于HyperGCN的扩展** :py:meth:`dhg.Graph.from_hypergraph_hypergcn`

在论文 `HyperGCN <https://arxiv.org/pdf/1809.02589.pdf>`_ 中， 作者介绍了一种将超图的超边简化为图的边的方法，如下图所示。

.. image:: ../../_static/img/hypergcn.png
    :align: center
    :alt: hypergcn
    :height: 200px


.. code-block:: python

    >>> X = torch.tensor(([[0.6460, 0.0247],
                           [0.9853, 0.2172],
                           [0.7791, 0.4780],
                           [0.0092, 0.4685],
                           [0.9049, 0.6371]]))
    >>> g = dhg.Graph.from_hypergraph_hypergcn(hg, X)
    >>> g
    Graph(num_v=5, num_e=4)
    >>> g.e
    ([(0, 2), (2, 3), (1, 2), (3, 4)], [0.3333333432674408, 0.3333333432674408, 0.5, 0.3333333432674408])
    >>> g.A.to_dense()
    tensor([[0.0000, 0.0000, 0.3333, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.5000, 0.0000, 0.0000],
            [0.3333, 0.5000, 0.0000, 0.3333, 0.0000],
            [0.0000, 0.0000, 0.3333, 0.0000, 0.3333],
            [0.0000, 0.0000, 0.0000, 0.3333, 0.0000]])
    >>> g = dhg.Graph.from_hypergraph_hypergcn(hg, X, with_mediator=True)
    >>> g
    Graph(num_v=5, num_e=6)
    >>> g.e
    ([(1, 2), (0, 1), (2, 3), (1, 3), (3, 4), (0, 3)], [0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408])
    >>> g.A.to_dense()
    tensor([[0.0000, 0.3333, 0.0000, 0.3333, 0.0000],
            [0.3333, 0.0000, 0.3333, 0.3333, 0.0000],
            [0.0000, 0.3333, 0.0000, 0.3333, 0.0000],
            [0.3333, 0.3333, 0.3333, 0.0000, 0.3333],
            [0.0000, 0.0000, 0.0000, 0.3333, 0.0000]])


.. image:: ../../_static/img/build_structure_graph_from_hypergcn.png
    :align: center
    :alt: Customize size
    :height: 400px


.. _zh_build_directed_graph:

构建有向图
+++++++++++++++++++++++

`有向图 <https://en.wikipedia.org/wiki/Directed_graph>`_ 为包含有向边的图, 边 ``(x, y)`` 和边 ``(y, x)`` 可以同时存在。
可以使用如下方式构建：

- 边列表 (**默认**) :py:class:`dhg.DiGraph`
- 邻接列表 :py:meth:`dhg.DiGraph.from_adj_list`
- 使用特征的k近邻 :py:meth:`dhg.DiGraph.from_feature_kNN`


常用方法
^^^^^^^^^^^^^^^^^^^
.. note:: 

    有向图同样支持在构建或增加边时，根据 ``merge_op`` 合并重边。

使用 :py:class:`dhg.DiGraph` 类 **从边列表构建一个有向图**

.. code-block:: python

    >>> import dhg
    >>> g = dhg.DiGraph(5, [(0, 3), (2, 4), (4, 2), (3, 1)])
    >>> g
    Directed Graph(num_v=5, num_e=4)
    >>> g.e
    ([(0, 3), (2, 4), (4, 2), (3, 1)], [1.0, 1.0, 1.0, 1.0])
    >>> # print the adjacency matrix
    >>> g.A.to_dense()
    tensor([[0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1.],
            [0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 0.]])

.. image:: ../../_static/img/build_structure_digraph_from_edge_list.png
    :align: center
    :alt: Customize size
    :height: 400px

可以发现有向图的邻接矩阵不是一个对称矩阵。

使用 :py:meth:`dhg.DiGraph.from_adj_list` 函数 **从邻接列表构建一个有向图**

.. code-block:: python

    >>> g = dhg.DiGraph.from_adj_list(5, [(0, 3, 4), (2, 1, 3), (3, 0)])
    >>> g
    Directed Graph(num_v=5, num_e=5)
    >>> g.e
    ([(0, 3), (0, 4), (2, 1), (2, 3), (3, 0)], [1.0, 1.0, 1.0, 1.0, 1.0])
    >>> # print the adjacency matrix
    >>> g.A.to_dense()
    tensor([[0., 0., 0., 1., 1.],
            [0., 0., 0., 0., 0.],
            [0., 1., 0., 1., 0.],
            [1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]])


.. image:: ../../_static/img/build_structure_digraph_from_adj.png
    :align: center
    :alt: Customize size
    :height: 400px


使用 :py:meth:`dhg.DiGraph.from_feature_kNN` 函数 **根据特征的k近邻构建有向图**

.. code-block:: python

    >>> X = torch.tensor(([[0.6460, 0.0247],
                           [0.9853, 0.2172],
                           [0.7791, 0.4780],
                           [0.0092, 0.4685],
                           [0.9049, 0.6371]]))
    >>> g = dhg.DiGraph.from_feature_kNN(X, k=2)
    >>> g
    Directed Graph(num_v=5, num_e=10)
    >>> g.e
    ([(0, 1), (0, 2), (1, 2), (1, 0), (2, 4), (2, 1), (3, 2), (3, 0), (4, 2), (4, 1)], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    >>> g.A.to_dense()
    tensor([[0., 1., 1., 0., 0.],
            [1., 0., 1., 0., 0.],
            [0., 1., 0., 0., 1.],
            [1., 0., 1., 0., 0.],
            [0., 1., 1., 0., 0.]], dtype=torch.float64)

.. image:: ../../_static/img/build_structure_digraph_from_knn.png
    :align: center
    :alt: Customize size
    :height: 400px


从高阶关联结构简化而来
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

期待您的贡献！

.. _zh_build_bipartite_graph:

构建二分图
+++++++++++++++++++++++

`二分图 <https://en.wikipedia.org/wiki/Bipartite_graph>`_ 包含两种类型的顶点以及连接不同类型顶点的边，
其分为 :math:`\mathcal{U}` 顶点集和 :math:`\mathcal{V}` 顶点集。
可以使用如下方式构建：

- 边列表 (**默认**) :py:class:`dhg.BiGraph`
- 邻接列表 :py:meth:`dhg.BiGraph.from_adj_list`
- 超图 :py:meth:`dhg.BiGraph.from_hypergraph`

常用方法
^^^^^^^^^^^^^^^^^^^
.. note:: 

    二分图同样支持在构建或增加边时，根据 ``merge_op`` 合并重边。

使用 :py:class:`dhg.BiGraph` 类 **从边列表构建一个二分图**

.. code-block:: python

    >>> import dhg
    >>> g = dhg.BiGraph(5, 4, [(0, 3), (4, 2), (1, 1), (2, 0)])
    >>> g
    Bipartite Graph(num_u=5, num_v=4, num_e=4)
    >>> g.e
    ([(0, 3), (4, 2), (1, 1), (2, 0)], [1.0, 1.0, 1.0, 1.0])
    >>> # print the bipartite adjacency matrix
    >>> g.B.to_dense()
    tensor([[0., 0., 0., 1.],
            [0., 1., 0., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 1., 0.]])
    >>> # print the adjacency matrix
    >>> g.A.to_dense()
    tensor([[0., 0., 0., 0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0.]])

.. image:: ../../_static/img/build_structure_bigraph_from_edge_list.png
    :align: center
    :alt: Customize size
    :height: 400px


使用 :py:meth:`dhg.BiGraph.from_adj_list` 函数 **从邻接列表构建一个二分图**

.. code-block:: python

    >>> g = dhg.BiGraph.from_adj_list(5, 4, [(0, 3, 2), (4, 2, 0), (1, 1, 2)])
    >>> g
    Bipartite Graph(num_u=5, num_v=4, num_e=6)
    >>> g.e
    ([(0, 3), (0, 2), (4, 2), (4, 0), (1, 1), (1, 2)], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    >>> g.B.to_dense()
    tensor([[0., 0., 1., 1.],
            [0., 1., 1., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [1., 0., 1., 0.]])
    >>> g.A.to_dense()
    tensor([[0., 0., 0., 0., 0., 0., 0., 1., 1.],
            [0., 0., 0., 0., 0., 0., 1., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 1., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0.],
            [1., 1., 0., 0., 1., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0.]])


.. image:: ../../_static/img/build_structure_bigraph_from_adj.png
    :align: center
    :alt: Customize size
    :height: 400px


从高阶关联结构简化而来
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

我们首先定义一个超图：

.. code-block:: python

    >>> hg = dhg.Hypergraph(5, [(0, 1, 2), (1, 3, 2), (1, 2), (0, 3, 4)])
    >>> hg.e
    ([(0, 1, 2), (1, 2, 3), (1, 2), (0, 3, 4)], [1.0, 1.0, 1.0, 1.0])
    >>> # print hypergraph incidence matrix
    >>> hg.H.to_dense()
    tensor([[1., 0., 0., 1.],
            [1., 1., 1., 0.],
            [1., 1., 1., 0.],
            [0., 1., 0., 1.],
            [0., 0., 0., 1.]])

使用函数 :py:meth:`dhg.BiGraph.from_hypergraph` **从超图构建一个二分图**

.. code-block:: python

    >>> g = dhg.BiGraph.from_hypergraph(hg, vertex_as_U=True)
    >>> g
    Bipartite Graph(num_u=5, num_v=4, num_e=11)
    >>> g.e
    ([(0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (0, 3), (3, 3), (4, 3)], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    >>> g.B.to_dense()
    tensor([[1., 0., 0., 1.],
            [1., 1., 1., 0.],
            [1., 1., 1., 0.],
            [0., 1., 0., 1.],
            [0., 0., 0., 1.]])
    >>> g = dhg.BiGraph.from_hypergraph(hg, vertex_as_U=False)
    >>> g
    Bipartite Graph(num_u=4, num_v=5, num_e=11)
    >>> g.e
    ([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (3, 0), (3, 3), (3, 4)], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    >>> g.B.to_dense()
    tensor([[1., 1., 1., 0., 0.],
            [0., 1., 1., 1., 0.],
            [0., 1., 1., 0., 0.],
            [1., 0., 0., 1., 1.]])


.. image:: ../../_static/img/build_structure_bigraph_from_hypergraph.png
    :align: center
    :alt: Customize size
    :height: 400px


构建高阶关联结构
-----------------------

当前版本DHG实现的高阶关联结构包括超图，以后我们将增加更多的高阶关联结构。

.. _zh_build_hypergraph:

构建超图
++++++++++++++++++++++++++
`超图 <https://en.wikipedia.org/wiki/Hypergraph>`_ 是超边中不含方向信息的超图。
超图内的每条超边可以连接两个或更多的顶点，其可以用所有顶点的子集表示。
可以使用如下方式构建：

- 超边列表 (**默认**) :py:class:`dhg.Hypergraph`
- 使用特征的k近邻 :py:meth:`dhg.Hypergraph.from_feature_kNN`
- 从低阶关联结构提升

  - 图 :py:meth:`dhg.Hypergraph.from_graph`
  - 图顶点的k阶邻居 :py:meth:`dhg.Hypergraph.from_graph_kHop`
  - 二分图 :py:meth:`dhg.Hypergraph.from_bigraph`


常用方法
^^^^^^^^^^^^^^^^^^^

使用 :py:class:`dhg.Hypergraph` 类 **从边列表构建一个超图**

.. code-block:: python

    >>> hg = dhg.Hypergraph(5, [(0, 1, 2), (2, 3), (0, 4)])
    >>> hg
    Hypergraph(num_v=5, num_e=3)
    >>> hg.e
    ([(0, 1, 2), (2, 3), (0, 4)], [1.0, 1.0, 1.0])
    >>> # print the incidence matrix of the hypergraph
    >>> hg.H.to_dense()
    tensor([[1., 0., 1.],
            [1., 0., 0.],
            [1., 1., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])

.. image:: ../../_static/img/build_structure_hypergraph_from_edge_list.png
    :align: center
    :alt: Customize size
    :height: 400px


.. important:: 

    超图里面的超边是顶点的无序集，也就意味着超边 ``(0, 1, 2)`` 、超边 ``(0, 2, 1)`` 和超边 ``(2, 1, 0)`` 是同一条超边。

.. code-block:: python

    >>> hg = dhg.Hypergraph(5, [(0, 2, 1), (2, 3), (0, 4)])
    >>> hg.e
    ([(0, 1, 2), (2, 3), (0, 4)], [1.0, 1.0, 1.0])
    >>> hg.H.to_dense()
    tensor([[1., 0., 1.],
            [1., 0., 0.],
            [1., 1., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])
    >>> hg = dhg.Hypergraph(5, [(1, 0, 2), (2, 3), (0, 4)])
    >>> hg.e
    ([(0, 1, 2), (2, 3), (0, 4)], [1.0, 1.0, 1.0])
    >>> hg.H.to_dense()
    tensor([[1., 0., 1.],
            [1., 0., 0.],
            [1., 1., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])

.. note:: 

    如果增加的超边有重边，这些重边将根据指定的 ``merge_op`` 合并。

.. code-block:: python

    >>> hg = dhg.Hypergraph(5, [(0, 1, 2), (2, 3), (2, 3), (0, 4)], merge_op="mean")
    >>> hg.e
    ([(0, 1, 2), (2, 3), (0, 4)], [1.0, 1.0, 1.0])
    >>> hg = dhg.Hypergraph(5, [(0, 1, 2), (2, 3), (2, 3), (0, 4)], merge_op="sum")
    >>> hg.e
    ([(0, 1, 2), (2, 3), (0, 4)], [1.0, 2.0, 1.0])
    >>> hg.add_hyperedges([(0, 2, 1), (0, 4)], merge_op="mean")
    >>> hg.e
    ([(0, 1, 2), (2, 3), (0, 4)], [1.0, 2.0, 1.0])
    >>> hg.add_hyperedges([(0, 2, 1), (0, 4)], merge_op="sum")
    >>> hg.e
    ([(0, 1, 2), (2, 3), (0, 4)], [2.0, 2.0, 2.0])

如果你分别设置 ``merge_op`` 为 ``mean`` 和 ``sum`` ，你会发现最后一条超边的权重分别是 ``1.0`` 和 ``2.0`` 。
You can find the weight of the last hyperedge is ``1.0`` and ``2.0``, if you set the ``merge_op`` to ``mean`` and ``sum``, respectively.


使用 :py:meth:`dhg.Hypergraph.from_feature_kNN` 函数 **根据特征的k近邻构建超图**

.. code-block:: python

    >>> X = torch.tensor([[0.0658, 0.3191, 0.0204, 0.6955],
                          [0.1144, 0.7131, 0.3643, 0.4707],
                          [0.2250, 0.0620, 0.0379, 0.2848],
                          [0.0619, 0.4898, 0.9368, 0.7433],
                          [0.5380, 0.3119, 0.6462, 0.4311],
                          [0.2514, 0.9237, 0.8502, 0.7592],
                          [0.9482, 0.6812, 0.0503, 0.4596],
                          [0.2652, 0.3859, 0.8645, 0.7619],
                          [0.4683, 0.8260, 0.9798, 0.2933],
                          [0.6308, 0.1469, 0.0304, 0.2073]])
    >>> hg = dhg.Hypergraph.from_feature_kNN(X, k=3)
    >>> hg
    Hypergraph(num_v=10, num_e=9)
    >>> hg.e
    ([(0, 1, 2), (0, 1, 5), (0, 2, 9), (3, 5, 7), (4, 7, 8), (4, 6, 9), (3, 4, 7), (4, 5, 8), (2, 6, 9)], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    >>> hg.H.to_dense()
    tensor([[1., 1., 1., 0., 0., 0., 0., 0., 0.],
            [1., 1., 0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 1., 0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 1., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 1., 1., 1., 1., 0.],
            [0., 1., 0., 1., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 1.],
            [0., 0., 0., 1., 1., 0., 1., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 1., 0.],
            [0., 0., 1., 0., 0., 1., 0., 0., 1.]])


.. image:: ../../_static/img/build_structure_hypergraph_from_knn.png
    :align: center
    :alt: Customize size
    :height: 400px


.. note:: 

    重边根据 ``mean`` 操作合并。


从低阶关联结构提升得到
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

使用 :py:meth:`dhg.Hypergraph.from_graph` 函数 **从图构建一个超图**


.. code-block:: python

    >>> g = dhg.Graph(5, [(0, 1), (1, 2), (2, 3), (1, 4)])
    >>> g.e
    ([(0, 1), (1, 2), (2, 3), (1, 4)], [1.0, 1.0, 1.0, 1.0])
    >>> g.A.to_dense()
    tensor([[0., 1., 0., 0., 0.],
            [1., 0., 1., 0., 1.],
            [0., 1., 0., 1., 0.],
            [0., 0., 1., 0., 0.],
            [0., 1., 0., 0., 0.]])
    >>> hg = dhg.Hypergraph.from_graph(g)
    >>> hg.e
    ([(0, 1), (1, 2), (2, 3), (1, 4)], [1.0, 1.0, 1.0, 1.0])
    >>> hg.H.to_dense()
    tensor([[1., 0., 0., 0.],
            [1., 1., 0., 1.],
            [0., 1., 1., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]])

.. image:: ../../_static/img/build_structure_hypergraph_from_graph.png
    :align: center
    :alt: Customize size
    :height: 400px


使用 :py:meth:`dhg.Hypergraph.from_graph_kHop` 函数 **根据图顶点的k阶邻居构建一个超图**

.. code-block:: python

    >>> g = dhg.Graph(5, [(0, 1), (1, 2), (2, 3), (1, 4)])
    >>> g.e
    ([(0, 1), (1, 2), (2, 3), (1, 4)], [1.0, 1.0, 1.0, 1.0])
    >>> g.A.to_dense()
    tensor([[0., 1., 0., 0., 0.],
            [1., 0., 1., 0., 1.],
            [0., 1., 0., 1., 0.],
            [0., 0., 1., 0., 0.],
            [0., 1., 0., 0., 0.]])
    >>> hg = dhg.Hypergraph.from_graph_kHop(g, k=1)
    >>> hg.e
    ([(0, 1), (0, 1, 2, 4), (1, 2, 3), (2, 3), (1, 4)], [1.0, 1.0, 1.0, 1.0, 1.0])
    >>> hg.H.to_dense()
    tensor([[1., 1., 0., 0., 0.],
            [1., 1., 1., 0., 1.],
            [0., 1., 1., 1., 0.],
            [0., 0., 1., 1., 0.],
            [0., 1., 0., 0., 1.]])
    >>> hg = dhg.Hypergraph.from_graph_kHop(g, k=2)
    >>> hg.e
    ([(0, 1, 2, 4), (0, 1, 2, 3, 4), (1, 2, 3)], [1.0, 1.0, 1.0])
    >>> hg.H.to_dense()
    tensor([[1., 1., 0.],
            [1., 1., 1.],
            [1., 1., 1.],
            [0., 1., 1.],
            [1., 1., 0.]])


.. image:: ../../_static/img/build_structure_hypergraph_from_khop.png
    :align: center
    :alt: Customize size
    :height: 400px


使用 :py:meth:`dhg.Hypergraph.from_bigraph` 函数 **从二分图构建一个超图**

    .. code-block:: python

        >>> g = dhg.BiGraph(4, 3, [(0, 1), (1, 1), (2, 1), (3, 0), (1, 2)])
        >>> g
        Bipartite Graph(num_u=4, num_v=3, num_e=5)
        >>> g.e
        ([(0, 1), (1, 1), (2, 1), (3, 0), (3, 2)], [1.0, 1.0, 1.0, 1.0, 1.0])
        >>> g.B.to_dense()
        tensor([[0., 1., 0.],
                [0., 1., 0.],
                [0., 1., 0.],
                [1., 0., 1.]])
        >>> hg = dhg.Hypergraph.from_bigraph(g, U_as_vertex=True)
        >>> hg
        Hypergraph(num_v=4, num_e=3)
        >>> hg.e
        ([(3,), (0, 1, 2), (1,)], [1.0, 1.0, 1.0])
        >>> hg.H.to_dense()
        tensor([[0., 1., 0.],
                [0., 1., 1.],
                [0., 1., 0.],
                [1., 0., 0.]])
        >>> hg = dhg.Hypergraph.from_bigraph(g, U_as_vertex=False)
        >>> hg
        Hypergraph(num_v=3, num_e=3)
        >>> hg.e
        ([(1,), (1, 2), (0,)], [1.0, 1.0, 1.0])
        >>> hg.H.to_dense()
        tensor([[0., 0., 1.],
                [1., 1., 0.],
                [0., 1., 0.]])

.. image:: ../../_static/img/build_structure_hypergraph_from_bigraph.png
    :align: center
    :alt: Customize size
    :height: 400px

