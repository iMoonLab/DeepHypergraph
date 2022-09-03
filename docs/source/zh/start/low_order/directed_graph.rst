
有向图上的表示学习
=============================

.. hint:: 

    - 作者:  `丰一帆 <https://fengyifan.site/>`_
    - 翻译:  颜杰龙
    - 校对： `丰一帆 <https://fengyifan.site/>`_ 、张欣炜

定义
-----------------------
`有向图 <https://en.wikipedia.org/wiki/Directed_graph>`_ 可以定义为 :math:`\mathcal{G} = \{\mathcal{V}, \mathcal{E}\}`。

- :math:`\mathcal{V}` 是 **顶点** 集(也可以称为 **节点** 或者 **点**);
- :math:`\mathcal{E} \subseteq \{ (x, y) \mid (x, y) \in \mathcal{V}^2~and~x \neq y \}` 是 **边** 集 (也可以称为 **有向边** 、 **有向连接** 、 **有向线** 、 **箭头** 、 **弧**),
  其为顶点间的 `有序对 <https://en.wikipedia.org/wiki/Ordered_pair>`_ (也就是说，一条边与两个不同的顶点相关联)。

在边 :math:`(x, y)` 中, 顶点 :math:`x` 和 :math:`y` 被称为边的 **端点**，
:math:`x` 为边的 **源** (也被称作 **尾**)， :math:`y` 为边的 **汇** (也被称为 **头** )。
称这条边 **连接** :math:`x` 和 :math:`y` 并且 **关联** :math:`x` 和  :math:`y`。
可能存在孤立顶点。
边 :math:`(y, x)` 称为边 :math:`(x, y)` 的反向边。
两条或更多的边连接同样的源和汇称为 `重边 <https://en.wikipedia.org/wiki/Multiple_edges>`_ 。有向图上不允许出现重边。


结构构建
-------------------------
有向图的关联结构可以通过以下方法构建。详细参考 :ref:`这里 <zh_build_directed_graph>`。

- 边列表 (**默认**) :py:class:`dhg.DiGraph`
- 邻接表 :py:meth:`dhg.DiGraph.from_adj_list`
- 使用特征的k近邻 :py:meth:`dhg.DiGraph.from_feature_kNN`

在如下的例子中，我们随机生成一个有向图关联结构和一个特征矩阵，并对此结构进行一些基本的学习操作。

    .. code:: python

        >>> import torch
        >>> import dhg
        >>> dhg.random.set_seed(2022)
        >>> # Generate a random directed graph with 5 vertices and 8 edges
        >>> g = dhg.random.digraph_Gnm(5, 8) 
        >>> # Generate a vertex feature matrix with size 5x2
        >>> X = torch.rand(5, 2)
        >>> # Print information about the directed graph and feature
        >>> g 
        Directed Graph(num_v=5, num_e=8)
        >>> # Print edges in the graph
        >>> g.e[0]
        [(2, 4), (4, 0), (0, 4), (3, 4), (0, 3), (4, 2), (2, 3), (3, 2)]
        >>> # Print vertex features
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])


.. Structure Visualization
.. ---------------------------------

.. Draw the directed graph structure

..     .. code:: python

..         >>> fig = g.draw(edge_style="line")
..         >>> fig.show()

..     Here is the image.


基于谱域的学习
---------------------------------

我们期待您的贡献！

基于空域的学习
---------------------------------

从源点到汇点的信息传递
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python

        >>> # Print the vertex messages
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])
        >>> X_src2dst = g.v2v(X, aggr="mean", direction="src2dst")
        >>> # Print the new vertex messages
        >>> X_src2dst
        tensor([[0.4643, 0.6329],
                [0.0000, 0.0000],
                [0.6288, 0.7070],
                [0.2110, 0.6407],
                [0.4051, 0.6875]])

从源点到汇点依赖边权的消息传递
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python

        >>> # Print the vertex messages
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])
        >>> g.e_weight
        tensor([1., 1., 1., 1., 1., 1., 1., 1.])
        >>> # Generate random edge weights
        >>> e_weight = torch.rand(len(g.e_weight))
        >>> e_weight
        tensor([0.6689, 0.2302, 0.8003, 0.7353, 0.7477, 0.5585, 0.6226, 0.8429])
        >>> X_ = g.v2v(X, e_weight=e_weight, aggr="softmax_then_sum", direction="src2dst")
        >>> # Print the new vertex messages
        >>> X_
        tensor([[0.4643, 0.6329],
                [0.0000, 0.0000],
                [0.6210, 0.7035],
                [0.1989, 0.6222],
                [0.3809, 0.6432]])


从汇点到源点的消息传递
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python

        >>> # Print the vertex messages
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])
        >>> X_dst2src = g.v2v(X, aggr="mean", direction="dst2src")
        >>> # Print the new vertex messages
        >>> X_dst2src
        tensor([[0.6288, 0.7070],
                [0.0000, 0.0000],
                [0.6288, 0.7070],
                [0.2453, 0.4962],
                [0.2110, 0.6407]])


从汇点到源点依赖边权的消息传递
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python

        >>> # Print the vertex messages
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])
        >>> g.e_weight
        tensor([1., 1., 1., 1., 1., 1., 1., 1.])
        >>> # Generate random edge weights
        >>> e_weight = torch.rand(len(g.e_weight))
        >>> e_weight
        tensor([0.6689, 0.2302, 0.8003, 0.7353, 0.7477, 0.5585, 0.6226, 0.8429])
        >>> X_ = g.v2v(X, e_weight=e_weight, aggr="softmax_then_sum", direction="dst2src")
        >>> # Print the new vertex messages
        >>> X_
        tensor([[0.6644, 0.7230],
                [0.0000, 0.0000],
                [0.6342, 0.7094],
                [0.2246, 0.4832],
                [0.1907, 0.6098]])
