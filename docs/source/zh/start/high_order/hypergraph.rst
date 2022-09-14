.. _zh_start_learning_on_simple_hypergraph:

超图上的表示学习
=================================

.. hint:: 

    - 作者:  `丰一帆 <https://fengyifan.site/>`_
    - 翻译:  颜杰龙
    - 校对： `丰一帆 <https://fengyifan.site/>`_ 、张欣炜

定义
-----------------
`超图 <https://en.wikipedia.org/wiki/Hypergraph>`_ (也可以称为无向超图) 可以表示为 :math:`\mathcal{G} = \{\mathcal{V}, \mathcal{E}\}`。

- :math:`\mathcal{V}` 是 **顶点** 集(也可以称为 **节点** 或者 **点**);
- :math:`\mathcal{E} \subseteq \{ \mathcal{P}(\mathcal{V}) \}` 是 **超边** 集(也可以称为 **边**), 其中 :math:`\mathcal{P}(\mathcal{V})` 是 :math:`\mathcal{V}` 的 `幂集 <https://en.wikipedia.org/wiki/Power_set>`_ 。
  每一条超边 :math:`e \in \mathcal{E}` 可以包含两个或更多顶点。

图内的边仅可以连接2个顶点，而超边可以连接任意数量的顶点。
然而，通常需要研究每条超边均连接相同个数顶点的超图;
k-uniform超图是所有超边连接的顶点数均为k的超图。
(换句话说，这样的超图是一组集合，每个集合包含k个顶点。)
所以2-uniform超图就是图，
3-uniform超图就是无序三元组的集合，依此类推。
无向超图也被称为集合系统或者从全集中提取的集合族。


结构构建
---------------------
超图的关联结构可以通过以下方法构建。详细参考 :ref:`这里 <zh_build_hypergraph>`。

- 超边列表 (**默认**) :py:class:`dhg.Hypergraph`
- 使用特征的k近邻 :py:meth:`dhg.Hypergraph.from_feature_kNN`
- 从低阶关联结构得到

  - 图 :py:meth:`dhg.Hypergraph.from_graph`
  - 图顶点的k阶邻居 :py:meth:`dhg.Hypergraph.from_graph_kHop`
  - 二分图 :py:meth:`dhg.Hypergraph.from_bigraph`


在如下的例子中，我们随机生成一个超图关联结构和一个特征矩阵，并对此结构进行一些基本的学习操作。

    .. code:: python

        >>> import torch
        >>> import dhg
        >>> # Generate a random hypergraph with 5 vertices and 4 hyperedges
        >>> hg = dhg.random.hypergraph_Gnm(5, 4) 
        >>> # Generate a vertex feature matrix with size 5x2
        >>> X = torch.rand(5, 2)
        >>> # Print information about the hypergraph and feature
        >>> hg 
        Hypergraph(num_v=5, num_e=4)
        >>> # Print edges in the hypergraph
        >>> hg.e[0]
        [(2, 3), (0, 2, 4), (2, 3, 4), (1, 2, 3, 4)]
        >>> # Print vertex features
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])

.. Structure Visualization
.. -------------------------------

.. Draw the hypergraph structure

..     .. code:: python

..         >>> fig = hg.draw(edge_style="circle")
..         >>> fig.show()
    
..     This is the image.

基于谱域的学习
-------------------------------

使用HGNN的拉普拉斯进行平滑
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python

        >>> # Print the vertex features befor feautre smoothing
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])
        >>> X_ = hg.smoothing_with_HGNN(X)
        >>> # Print the vertex features after HGNN-based smoothing
        >>> X_
        tensor([[0.2257, 0.4890],
                [0.3745, 0.3443],
                [0.5411, 0.7403],
                [0.4945, 0.5725],
                [0.4888, 0.6728]])

基于空域的学习
-------------------------------

从顶点到超边的消息传递
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python

        >>> # Print the vertex messages
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])
        >>> # Message propagation from vertex to hyperedge
        >>> Y_ = hg.v2e(X, aggr="mean")
        >>> # Print the new hyperedge messages
        >>> Y_
        tensor([[0.4098, 0.5702],
                [0.2955, 0.6381],
                [0.4280, 0.5911],
                [0.5107, 0.5386]])

从顶点到超边依赖边权的消息传递
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python

        >>> # Print the vertex messages
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])
        >>> hg.v2e_weight
        tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        >>> # Generate random edge weights for the first stage
        >>> v2e_weight = torch.rand(len(hg.v2e_weight))
        >>> v2e_weight
        tensor([0.6689, 0.2302, 0.8003, 0.7353, 0.7477, 0.5585, 0.6226, 0.8429, 0.6105,
                0.1248, 0.8265, 0.2117])
        >>> # Message propagation from vertex to hyperedge
        >>> Y_ = hg.v2e(X, v2e_weight=v2e_weight, aggr="mean")
        >>> # Print the new hyperedge messages
        >>> Y_
        tensor([[0.7326, 1.1010],
                [0.5229, 1.4678],
                [2.5914, 3.5052],
                [1.2437, 1.4487]])


从超边到顶点的消息传递
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python

        >>> # Print current hyperedge messages
        >>> Y_
        tensor([[0.4098, 0.5702],
                [0.2955, 0.6381],
                [0.4280, 0.5911],
                [0.5107, 0.5386]])
        >>> # Message propagation from hyperedge to vertex
        >>> X_ = hg.e2v(Y_, aggr="mean")
        >>> # Print the new vertex messages
        >>> X_
        tensor([[0.2955, 0.6381],
                [0.5107, 0.5386],
                [0.4110, 0.5845],
                [0.4495, 0.5667],
                [0.4114, 0.5893]])


从超边到顶点依赖边权的消息传递
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python

        >>> # Print current hyperedge messages
        >>> Y_
        tensor([[0.4098, 0.5702],
                [0.2955, 0.6381],
                [0.4280, 0.5911],
                [0.5107, 0.5386]])
        >>> hg.e2v_weight
        tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        >>> # Generate random edge weights for the second stage
        >>> e2v_weight = torch.rand(len(hg.e2v_weight))
        >>> e2v_weight
        tensor([0.8574, 0.4282, 0.3964, 0.1440, 0.0034, 0.9504, 0.2194, 0.2893, 0.6784,
                0.4997, 0.9144, 0.2833])
        >>> # Message propagation from hyperedge to vertex
        >>> X_ = hg.e2v(Y_, e2v_weight=e2v_weight, aggr="mean")
        >>> # Print the new vertex messages
        >>> X_
        tensor([[0.2172, 0.4691],
                [0.0936, 0.0988],
                [1.0335, 1.2427],
                [0.6650, 0.7853],
                [1.1605, 1.7178]])

从顶点集到顶点集的消息传递
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

每一条超边连接一个顶点集，其为两个顶点集合的信息桥梁。
在超图中，超边连接的源顶点集和汇顶点集是相同的。

    .. code:: python

        >>> # Print the vertex messages
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])
        >>> # Message propagation from vertex set to vertex set
        >>> X_ = hg.v2v(X, aggr="mean")
        >>> # Print the new vertex messages
        >>> X_
        tensor([[0.2955, 0.6381],
                [0.5107, 0.5386],
                [0.4110, 0.5845],
                [0.4495, 0.5667],
                [0.4114, 0.5893]])

从顶点集到顶点集依赖边权的两阶段消息传递
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python

        >>> # Print the vertex messages
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])
        >>> hg.v2e_weight
        tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        >>> # Generate random edge weights for the first stage
        >>> v2e_weight = torch.rand(len(hg.v2e_weight))
        >>> v2e_weight
        tensor([0.5739, 0.2444, 0.2476, 0.1210, 0.6869, 0.6617, 0.5168, 0.9089, 0.8799,
                0.6949, 0.4609, 0.1263])
        >>> hg.e2v_weight
        tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        >>> # Generate random edge weights for the second stage
        >>> e2v_weight = torch.rand(len(hg.e2v_weight))
        >>> e2v_weight
        tensor([0.6332, 0.4839, 0.7779, 0.9180, 0.0768, 0.9693, 0.2956, 0.7251, 0.5438,
                0.7403, 0.3211, 0.5044])
        >>> # Message propagation from vertex set to vertex set
        >>> X_ = hg.v2v(X, v2e_weight=v2e_weight, e2v_weight=e2v_weight, aggr="mean")
        >>> # Print the new vertex messages
        >>> X_
        tensor([[ 0.3082,  0.5642],
                [ 0.4297,  0.4918],
                [ 7.9027, 10.4666],
                [ 3.9316,  4.8732],
                [ 3.3256,  4.5806]])

