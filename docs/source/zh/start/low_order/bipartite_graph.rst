
.. _zh_start_learning_on_bipartite_graph:

二分图上的表示学习
==============================

.. hint:: 

    - 作者:  `丰一帆 <https://fengyifan.site/>`_
    - 翻译:  颜杰龙
    - 校对： `丰一帆 <https://fengyifan.site/>`_ 、张欣炜

定义
-------------------------
`二分图 <https://en.wikipedia.org/wiki/Bipartite_graph>`_ 可以表示为 :math:`\mathcal{G} = (\mathcal{U}, \mathcal{V}, \mathcal{E})`，
其分为 :math:`\mathcal{U}` 和 :math:`\mathcal{V}` 两部分， 通过二分图内的边 :math:`\mathcal{E}` 连接。

- :math:`\mathcal{U}` 是一个 **顶点** 集(也可以称为 **节点** 或者 **点**);
- :math:`\mathcal{V}` 是另一个 **顶点** 集(也可以称为 **节点** 或者 **点**);
- :math:`\mathcal{E} \subseteq \{ (x, y) \mid x \in \mathcal{U}~and~y \in \mathcal{V} \}` 是 **边** 集(也可以称为 **连接** 或者 **线**),

当对两类不同对线之间的关系建模时，通常会自然出现二分图。
例如，足球运动员和俱乐部之间的二分图，如果该球员曾经为该俱乐部效力，那么在该球员和该俱乐部间有一条边，
这是从隶属网络的自然例子，也是一种用于社交网络分析的二分图。
另一个例子是表示<用户-物品>交互的二分图，其中用户观看/点击/查看/喜欢/调整一个项目可以建模不同的场景，例如电影/新闻/商品/酒店/产品/服务推荐。



结构构建
-------------------------
二分图的关联结构可以通过以下方法构建。 详细参考 :ref:`这里 <zh_build_bipartite_graph>`。

- 边列表 (**默认**) :py:class:`dhg.BiGraph`
- 邻接表 :py:meth:`dhg.BiGraph.from_adj_list`
- 超图 :py:meth:`dhg.BiGraph.from_hypergraph`

在如下的例子中，我们随机生成一个二分图关联结构和两个特征矩阵，并对此结构进行一些基本的学习操作。

    .. code:: python

        >>> import torch
        >>> import dhg
        >>> # Generate a random bipartite graph with 3 vertices in set U, 5 vertices in set V, and 8 edges
        >>> g = dhg.random.bigraph_Gnm(3, 5, 8)
        >>> # Generate feature matrix for vertices in set U and set V, respectively.
        >>> X_u, X_v = torch.rand(3, 2), torch.rand(5, 2)
        >>> # Print information about the bipartite graph and feature
        >>> g 
        Bipartite Graph(num_u=3, num_v=5, num_e=8)
        >>> # Print edges in the graph
        >>> g.e[0]
        [(2, 4), (0, 4), (0, 3), (2, 0), (1, 4), (2, 3), (2, 2), (1, 3)]
        >>> # Print vertex features
        >>> X_u
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594]])
        >>> X_v
        tensor([[0.7933, 0.7811],
                [0.4643, 0.6329],
                [0.6689, 0.2302],
                [0.8003, 0.7353],
                [0.7477, 0.5585]])

.. Structure Visualization
.. -------------------------------

.. Draw the bipartite graph structure

..     .. code:: python

..         >>> fig = g.draw(edge_style="line")
..         >>> fig.show()

..     Here is the image.


基于谱域的学习
-----------------------------

使用GCN的拉普拉斯进行平滑
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python
        
        >>> # Print the Laplacian matrix defined by GCN that associated with the bipartite graph structure
        >>> g.L_GCN.to_dense()
        tensor([[0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2887, 0.2887],
                [0.0000, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.2887, 0.2887],
                [0.0000, 0.0000, 0.2000, 0.3162, 0.0000, 0.3162, 0.2236, 0.2236],
                [0.0000, 0.0000, 0.3162, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.3162, 0.0000, 0.0000, 0.5000, 0.0000, 0.0000],
                [0.2887, 0.2887, 0.2236, 0.0000, 0.0000, 0.0000, 0.2500, 0.0000],
                [0.2887, 0.2887, 0.2236, 0.0000, 0.0000, 0.0000, 0.0000, 0.2500]])
        >>> # Concate the vertex features
        >>> X = torch.cat([X_u, X_v], dim=0)
        >>> # Print the vertex features
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329],
                [0.6689, 0.2302],
                [0.8003, 0.7353],
                [0.7477, 0.5585]])
        >>> X_ = g.smoothing_with_GCN(X)
        >>> # Print the new vertex features
        >>> X_
        tensor([[0.5788, 0.6808],
                [0.6998, 0.5005],
                [0.8138, 0.6810],
                [0.4050, 0.5042],
                [0.4643, 0.6329],
                [0.3428, 0.2288],
                [0.5392, 0.6403],
                [0.5261, 0.5961]])
        >>> # Print the new vertex feautres in set U and set V, respectively
        >>> X_u_, X_v_ = torch.split(X_, [g.num_u, g.num_v], dim=0)
        >>> X_u_
        tensor([[0.5788, 0.6808],
                [0.6998, 0.5005],
                [0.8138, 0.6810]])
        >>> X_v_
        tensor([[0.4050, 0.5042],
                [0.4643, 0.6329],
                [0.3428, 0.2288],
                [0.5392, 0.6403],
                [0.5261, 0.5961]])


基于空域的学习
----------------------------

从 :math:`U` 内顶点到 :math:`V` 内顶点的消息传递
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python

        >>> # Print the messages of vertices in set U
        >>> X_u
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594]])
        >>> X_v_ = g.u2v(X_u, aggr="mean")
        >>> # Print the new messages of vertices in set V
        >>> X_v_
        tensor([[0.0262, 0.3594],
                [0.0000, 0.0000],
                [0.0262, 0.3594],
                [0.3936, 0.5542],
                [0.3936, 0.5542]])

从 :math:`U` 内顶点到 :math:`V` 内顶点依赖边权的消息传递
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python

        >>> # Print the messages of vertices in set U
        >>> X_u
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594]])
        >>> g.e_weight
        tensor([1., 1., 1., 1., 1., 1., 1., 1.])
        >>> # Generate random edge weights
        >>> e_weight = torch.rand(len(g.e_weight))
        >>> e_weight
        tensor([0.6226, 0.8429, 0.6105, 0.1248, 0.8265, 0.2117, 0.8574, 0.4282])
        >>> X_v_ = g.u2v(X_u, e_weight=e_weight, aggr="mean")
        >>> # Print the new messages of vertices in set V
        >>> X_v_
        tensor([[1.7913e-02, 2.4547e-01],
                [0.0000e+00, 0.0000e+00],
                [1.1753e-03, 1.6106e-02],
                [1.5306e+00, 2.3305e+00],
                [6.1360e-01, 1.3660e+00]])

从 :math:`V` 内顶点到 :math:`U` 内顶点的消息传递
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python

        >>> # Print the messages of vertices in set V
        >>> X_v
        tensor([[0.7933, 0.7811],
                [0.4643, 0.6329],
                [0.6689, 0.2302],
                [0.8003, 0.7353],
                [0.7477, 0.5585]])
        >>> X_u_ = g.v2u(X_v, aggr="mean")
        >>> # Print the new messages of vertices in set U
        >>> X_u_
        tensor([[0.7740, 0.6469],
                [0.7740, 0.6469],
                [0.7526, 0.5763]])

从 :math:`V` 内顶点到 :math:`U` 内顶点依赖边权的消息传递
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python

        >>> # Print the messages of vertices in set V
        >>> X_v
        tensor([[0.7933, 0.7811],
                [0.4643, 0.6329],
                [0.6689, 0.2302],
                [0.8003, 0.7353],
                [0.7477, 0.5585]])
        >>> g.e_weight
        tensor([1., 1., 1., 1., 1., 1., 1., 1.])
        >>> # Generate random edge weights
        >>> e_weight = torch.rand(len(g.e_weight))
        >>> e_weight
        tensor([0.6226, 0.8429, 0.6105, 0.1248, 0.8265, 0.2117, 0.8574, 0.4282])
        >>> X_u_ = g.v2u(X_v, e_weight=e_weight, aggr="mean")
        >>> # Print the new messages of vertices in set U
        >>> X_u_
        tensor([[1.6537, 1.3607],
                [0.4279, 0.3814],
                [4.1914, 3.6342]])

