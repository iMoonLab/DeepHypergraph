DHG内的关联结构
===================================

.. hint:: 

    - 作者:  `丰一帆 <https://fengyifan.site/>`_
    - 翻译:  颜杰龙
    - 校对： `丰一帆 <https://fengyifan.site/>`_ 、张欣炜

简介
----------------
**DHG** 的核心动机是将基于谱域和基于空域的操作加到每种特定关联结构中。
当关联结构构造后，相关的拉普拉斯矩阵和具有不同聚合函数的信息传递操作可以被调用和组合来操作任意输入特征。
目前， **DHG** 已经完成以下关联结构和操作。
更多关联结构和操作将来会被加入。 **我们期待您的贡献!**


.. csv-table:: 支持的关联结构以及聚合函数总览
    :header: "关联结构", "类", "类型", "基于谱域的操作", "基于空域的操作"
    :widths: 2 2 2 3 3

    "图", ":py:class:`dhg.Graph`", "低阶关联", "| :math:`\mathcal{L}` , :math:`\mathcal{L}_{sym}` , :math:`\mathcal{L}_{rw}`
    | :math:`\mathcal{L}_{GCN}`", ":math:`v \rightarrow v`"
    "有向图", ":py:class:`dhg.DiGraph`", "低阶关联", *未来工作*, "| :math:`v_{src} \rightarrow v_{dst}`
    | :math:`v_{dst} \rightarrow v_{src}`"
    "二分图", ":py:class:`dhg.BiGraph`", "低阶关联", ":math:`\mathcal{L}_{GCN}`", "| :math:`u \rightarrow v`
    | :math:`v \rightarrow u`"
    "超图", ":py:class:`dhg.Hypergraph`", "高阶关联", "| :math:`\mathcal{L}_{sym}` , :math:`\mathcal{L}_{rw}`
    | :math:`\mathcal{L}_{HGNN}`", "| :math:`v \rightarrow e`
    | :math:`v \rightarrow e` (特定超边组内)
    | :math:`e \rightarrow v`
    | :math:`e \rightarrow v` (特定超边组内)"


应用场景
-----------------

.. csv-table:: 不同关联结构的应用总览
    :header: 关联结构, "应用", "示例代码"
    :widths: 2, 6, 3

    "图", "基于引用网络的论文分类等", ":doc:`样例 </zh/examples/vertex_cls/graph>`"
    "有向图", "点云分类等", "\-"
    "二分图", "| 基于<用户-物品>二分图的物品推荐、
    | 基于<蛋白质-药品>二分图的关联预测等", ":doc:`样例 </zh/examples/recommender>`"
    "超图", "| 基于社交网络的顶点分类,
    | 基于多模态视觉对象图的视觉对象分类等", ":doc:`样例 </zh/examples/vertex_cls/hypergraph>`"
    

两个核心操作
----------------------------
在图、超图中的大多数学习可以分为两类：基于谱域的卷积以及基于空域的信息传递。
典型GCN和HGNN等基于谱域的卷积方法，学习给定关联结构的拉普拉斯矩阵，使用生成的拉普拉斯矩阵执行 ``vertex feature smoothing`` ，以将低阶和高阶结构嵌入到顶点特征内。
典型GraphSAGE、GAT、、HGNN :sup:`+` 等基于空域的信息传递方法，
执行 ``vertex to vertex`` 、``vertex to hyperedge``、 ``hyperedge to vertex``、``vertex set to vertex set`` 等信息传递以将低阶和高阶结构嵌入到顶点特征中。
也可以将学习得到的顶点特征池化为统一的结构特征。
最终，学习到的顶点特征或结构特征可以被分类、检索、回归、链路预测等下游任务和论文分类、电影推荐、药物挖掘等应用中。

基于谱域的操作
+++++++++++++++++++++++++++++++
基于谱域卷积的核心在于如拉普拉斯矩阵的平滑矩阵。
每种关联结构中都提供常见的平滑矩阵。
例如, 图和二分图结构中可以调用  `GCN <_blank>`_ 中的拉普拉斯矩阵, 超图结构中可以调用 `HGNN <_blank>`_ 中的拉普拉斯矩阵。

在如下例子中，我们随机生成一个包含5个顶点和8条边的 **图**。
我们可以使用 ``g.L_GCN`` 内部属性获取指定图结构的拉普拉斯矩阵，其生成的拉普拉斯矩阵大小为 :math:`5 \times 5` 。
然后，对于任意输入的顶点特征，您可以使用特定图 ``g`` 中的函数  ``g.smoothing_with_GCN()`` 来对特征平滑处理。

    .. code:: python

        >>> import torch
        >>> import dhg
        >>> g = dhg.random.graph_Gnm(5, 8)
        >>> # Generate a vertex feature matrix with size 5x2
        >>> X = torch.rand(5, 2)
        >>> # Print information about the graph and feature
        >>> g 
        Graph(num_v=5, num_e=8)
        >>> # Print edges in the graph
        >>> g.e[0]
        [(0, 1), (2, 4), (0, 4), (3, 4), (0, 3), (2, 3), (0, 2), (1, 3)]
        >>> # Print vertex features
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])
        >>> # Print the inside Laplacian Matrix by GCN on the graph structure
        >>> g.L_GCN.to_dense()
        tensor([[0.2000, 0.2582, 0.2236, 0.2000, 0.2236],
                [0.2582, 0.3333, 0.0000, 0.2582, 0.0000],
                [0.2236, 0.0000, 0.2500, 0.2236, 0.2500],
                [0.2000, 0.2582, 0.2236, 0.2000, 0.2236],
                [0.2236, 0.0000, 0.2500, 0.2236, 0.2500]])
        >>> X_ = g.smoothing_with_GCN(X)
        >>> # Print the vertex features after GCN-based smoothing
        >>> X_
        tensor([[0.5434, 0.6609],
                [0.5600, 0.5668],
                [0.3885, 0.6289],
                [0.5434, 0.6609],
                [0.3885, 0.6289]])

在如下例子中，我们随机生成集合 :math:`\mathcal{U}` 含有3个顶点、集合 :math:`\mathcal{V}` 含有5个顶点总共8条边的 **二分图**。
我们可以使用 ``g.L_GCN`` 内部属性获取指定二分图结构的拉普拉斯矩阵，其生成的拉普拉斯矩阵大小为 :math:`8 \times 8` 。
然后，对于任意输入的顶点特征，您可以使用特定二分图 ``g`` 中的函数  ``g.smoothing_with_GCN()`` 来对特征平滑处理。
更多细节可以参考自 :ref:`此链接 <zh_start_learning_on_bipartite_graph>` 。

    .. note:: 

        GCN的二分图拉普拉斯矩阵是通过扩展大小为 :math:`|\mathcal{U}| \times |\mathcal{V}|` 的二分图邻接矩阵 :math:`\mathbf{B}` 到
        大小为 :math:`||\mathcal{U}| + |\mathcal{V}|| \times ||\mathcal{U}| + |\mathcal{V}||` 的大邻接矩阵 :math:`\mathbf{A}` 实现的。

    .. code:: python

        >>> import torch
        >>> import dhg
        >>> g = dhg.random.bigraph_Gnm(3, 5, 8)
        >>> # Print edges in the bipartite graph structure 
        >>> g.e[0]
        [(2, 4), (0, 4), (0, 3), (2, 0), (1, 4), (2, 3), (2, 2), (1, 3)]
        >>> # Print the inside Laplacian Matrix by GCN on the bipartite graph structure
        >>> g.L_GCN.to_dense()
        tensor([[0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2887, 0.2887],
                [0.0000, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.2887, 0.2887],
                [0.0000, 0.0000, 0.2000, 0.3162, 0.0000, 0.3162, 0.2236, 0.2236],
                [0.0000, 0.0000, 0.3162, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.3162, 0.0000, 0.0000, 0.5000, 0.0000, 0.0000],
                [0.2887, 0.2887, 0.2236, 0.0000, 0.0000, 0.0000, 0.2500, 0.0000],
                [0.2887, 0.2887, 0.2236, 0.0000, 0.0000, 0.0000, 0.0000, 0.2500]])

在如下例子中，我们随机生成一个包含5个顶点和4条超边的 **超图**。
我们可以使用 ``hg.L_HGNN`` 内部属性获取指定超图结构的拉普拉斯矩阵，其生成的拉普拉斯矩阵大小为 :math:`5 \times 5` 。
然后，对于任意输入的顶点特征，您可以使用 特定超图 ``hg`` 中的函数  ``hg.smoothing_with_HGNN()`` 来对特征平滑处理。
更多细节可以参考自 :ref:`此链接 <zh_start_learning_on_simple_hypergraph>`。

    .. code:: python

        >>> import torch
        >>> import dhg
        >>> hg = dhg.random.hypergraph_Gnm(5, 4)
        >>> # Print hyperedges in the hypergraph structure 
        >>> hg.e[0]
        [(2, 3), (0, 2, 4), (2, 3, 4), (1, 2, 3, 4)]
        >>> # Print the inside Laplacian Matrix by HGNN on the hypergraph structure
        >>> hg.L_HGNN.to_dense()
        tensor([[0.3333, 0.0000, 0.1667, 0.0000, 0.1925],
                [0.0000, 0.2500, 0.1250, 0.1443, 0.1443],
                [0.1667, 0.1250, 0.3542, 0.3127, 0.2646],
                [0.0000, 0.1443, 0.3127, 0.3611, 0.1944],
                [0.1925, 0.1443, 0.2646, 0.1944, 0.3056]])

基于空域的操作
+++++++++++++++++++++++++++++++
基于空域信息传递的核心在于从 ``source domain`` 到 ``target domain`` 的信息传递以及使用不同聚合函数的信息聚合。
在 **DHG** 中， ``source domain`` 和 ``target domain`` 可以是 ``a vertex`` 、 ``a vertex in specified vertex set`` 、 ``a hyperedge`` 、 ``a vertex set`` 的其中之一，
信息聚合函数可以是 ``mean``、 ``softmax``、  ``softmax_then_sum``。
因此，与 `PyG <https://www.pyg.org/>`_ 和 `DGL <https://www.dgl.ai/>`_ 中只能将信息从 ``a vertex`` 传输到 ``another vertex or edge`` 不同，
**DHG** 为低阶和高阶关联结构提供更多种类型的信息传递操作。

在如下例子中，我们随机生成一个包含5个顶点和8条边的 **图**。
图结构提供从 ``a vertex`` 到 ``another vertex`` 的信息传递，以及支持 ``mean`` 、 ``softmax`` 、 ``softmax_then_sum`` 信息聚合函数。

    .. code:: python

        >>> import torch
        >>> import dhg
        >>> g = dhg.random.graph_Gnm(5, 8)
        >>> # Generate a vertex feature matrix with size 5x2
        >>> X = torch.rand(5, 2)
        >>> # Print information about the graph and feature
        >>> g 
        Graph(num_v=5, num_e=8)
        >>> # Print edges in the graph
        >>> g.e[0]
        [(0, 1), (2, 4), (0, 4), (3, 4), (0, 3), (2, 3), (0, 2), (1, 3)]
        >>> # Print vertex messages
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])
        >>> # Propagate messages from a vertex to another vertex with mean aggregation function
        >>> X_ = g.v2v(X, aggr="mean")
        >>> # Print new vertex messages
        >>> X_
        tensor([[0.5107, 0.5386],
                [0.5946, 0.8515],
                [0.5512, 0.7786],
                [0.4113, 0.5738],
                [0.4051, 0.6875]])
        >>> # Propagate messages from a vertex to another vertex with sum aggregation function
        >>> X_ = g.v2v(X, aggr="sum")
        >>> # Print new vertex messages
        >>> X_
        tensor([[2.0427, 2.1545],
                [1.1892, 1.7030],
                [1.6535, 2.3359],
                [1.6452, 2.2954],
                [1.2154, 2.0624]])
        >>> # Set the weight of each edge for softmax in neighbor aggregation
        >>> e_weight = g.e_weight
        >>> # Propagate messages from a vertex to another vertex with softmax_then_sum aggregation function
        >>> X_ = g.v2v(X, e_weight=e_weight, aggr="softmax_then_sum")
        >>> # Print new vertex messages
        >>> X_
        tensor([[0.5107, 0.5386],
                [0.5946, 0.8515],
                [0.5512, 0.7786],
                [0.4113, 0.5738],
                [0.4051, 0.6875]])

在如下例子中，我们随机生成集合 :math:`\mathcal{U}` 含有3个顶点、集合 :math:`\mathcal{V}` 含有5个顶点总共8条边的 **二分图**。
二分图关联结构中，提供从 ``a vertex in a specified vertex set`` 到 ``another vertex in another specified vertex set`` 信息传递
以及支持 ``mean`` 、 ``softmax`` 、 ``softmax_then_sum`` 信息聚合函数。
二分图中基于空域的操作细节可以参考 :ref:`此链接 <zh_start_learning_on_bipartite_graph>` 。


    .. code:: python

        >>> import torch
        >>> import dhg
        >>> # Generate a random bipartite graph with 3 vertices in set U, 5 vertices in set V, and 8 edges
        >>> g = dhg.random.bigraph_Gnm(3, 5, 8)
        >>> # Generate feature matrix for vertices in set U and set V, respectively.
        >>> X_u, X_v = torch.rand(3, 2), torch.rand(5, 2)
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
        >>> # Propagate messages from vertices in set V to vertices in set U with mean aggregation
        >>> X_u_ = g.v2u(X_v, aggr="mean")
        >>> X_u_
        tensor([[0.7740, 0.6469],
                [0.7740, 0.6469],
                [0.7526, 0.5763]])
        >>> # Propagate messages from vertices in set U to vertices in set V with mean aggregation
        >>> X_v_ = g.u2v(X_u, aggr="mean")
        >>> X_v_
        tensor([[0.0262, 0.3594],
                [0.0000, 0.0000],
                [0.0262, 0.3594],
                [0.3936, 0.5542],
                [0.3936, 0.5542]])

在如下例子中，我们随机生成一个包含5个顶点和4条超边的 **超图**。
超图关联结构中，提供从 ``a vertex`` 到 ``another vertex`` 、 从 ``a vertex set`` 到 ``a hyperedge`` 、
从 ``a hyperedge`` 到 ``a vertex set`` 、  从 ``a vertex set`` 到 ``another vertex set`` 四种信息传递
以及支持 ``mean`` 、 ``softmax`` 、 ``softmax_then_sum`` 信息聚合函数。
超图中基于空域的操作细节可以参考 :ref:`此链接 <zh_start_learning_on_simple_hypergraph>`。
 
    .. code:: python
    
        >>> import torch
        >>> import dhg
        >>> g = dhg.random.hypergraph_Gnm(5, 4)
        >>> # Generate a vertex feature matrix with size 5x2
        >>> X = torch.rand(5, 2)
        >>> # Print information about the hypergraph and feature
        >>> g 
        Hypergraph(num_v=5, num_e=4)
        >>> # Print edges in the graph
        >>> g.e[0]
        [(2, 3), (0, 2, 4), (2, 3, 4), (1, 2, 3, 4)]
        >>> # Print vertex messages
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])
        >>> # Propagate messages from vertex sets to hyperedges with mean aggregation function
        >>> Y_ = g.v2e(X, aggr="mean")
        >>> # Print new hyperedge messages
        >>> Y_
        tensor([[0.4098, 0.5702],
                [0.2955, 0.6381],
                [0.4280, 0.5911],
                [0.5107, 0.5386]])
        >>> # Propagate messages from hyperedges to vertex sets with mean aggregation function
        >>> X_ = g.e2v(Y_, aggr="mean")
        >>> # Print new vertex messages
        >>> X_
        tensor([[0.2955, 0.6381],
                [0.5107, 0.5386],
                [0.4110, 0.5845],
                [0.4495, 0.5667],
                [0.4114, 0.5893]])


基于两种操作可以实现什么?
-------------------------------------------


增加先自环以及后自环
++++++++++++++++++++++++++++++++++++++++++

自环是特征学习特别是图关联结构中的重要结构。
在如下的例子中，我们介绍如何在图关联结构中为基于空域的学习增加先自环和后自环。
我们使用 :math:`\mathbf{A} \in \mathbb{R}^{N \times N}` 来表示一个给定图的邻接矩阵，:math:`\mathbf{X} \in \mathbb{R}^{N \times C}` 来表示一个给定图的节点特征.


    .. code:: python

        >>> import torch
        >>> import dhg
        >>> g = dhg.random.graph_Gnm(5, 8)
        >>> # Generate a vertex feature matrix with size 5x2
        >>> X = torch.rand(5, 2)
        >>> # Print information about the graph and feature
        >>> g 
        Graph(num_v=5, num_e=8)
        >>> # Print edges in the graph
        >>> g.e[0]
        [(0, 1), (2, 4), (0, 4), (3, 4), (0, 3), (2, 3), (0, 2), (1, 3)]
        >>> # Print vertex features
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])

使用先自环的信息传递
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. math::

        \left\{  
        \begin{aligned}
        \mathbf{X}^\prime &= \hat{\mathbf{A}} \mathbf{X}\\
        \hat{\mathbf{A}} &= \mathbf{A} + \mathbf{I}   
        \end{aligned}
        \right.  


    .. code:: python
        
        >>> # Print edges in the graph
        >>> g.e[0]
        [(0, 1), (2, 4), (0, 4), (3, 4), (0, 3), (2, 3), (0, 2), (1, 3)]
        >>> # Print vertex features
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])
        >>> # Add self-loop before message passing
        >>> g.add_extra_selfloop()
        >>> g.e[0]
        [(0, 1), (2, 4), (0, 4), (3, 4), (0, 3), (2, 3), (0, 2), (1, 3), (0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        >>> X_ = g.v2v(X, aggr="mean")
        >>> X_
        tensor([[0.4877, 0.6153],
                [0.6493, 0.6947],
                [0.4199, 0.6738],
                [0.4877, 0.6153],
                [0.4199, 0.6738]])


使用后自环的信息传递
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. math::

        \mathbf{X}^\prime = \mathbf{A} \mathbf{X} + \mathbf{X}


    .. code:: python
        
        >>> # Print edges in the graph
        >>> g.e[0]
        [(0, 1), (2, 4), (0, 4), (3, 4), (0, 3), (2, 3), (0, 2), (1, 3)]
        >>> # Print vertex features
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])
        >>> # Add self-loop after message passing
        >>> X_ = X + g.v2v(X, aggr="mean")
        >>> X_
        tensor([[0.9065, 1.4606],
                [1.3534, 1.2326],
                [0.5774, 1.1380],
                [1.2046, 1.3549],
                [0.8695, 1.3204]])


融合从谱域和空域中学习到的特征
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

在如下例子中，我们随机生成一个包含5个顶点和8条边的 **图**。
然后，我们尝试融合从相同关联结构 ``g`` 使用不同方法学习的特征。

    .. code:: python

        >>> import torch
        >>> import dhg
        >>> g = dhg.random.graph_Gnm(5, 8)
        >>> # Generate a vertex feature matrix with size 5x2
        >>> X = torch.rand(5, 2)
        >>> # Print information about the graph and feature
        >>> g 
        Graph(num_v=5, num_e=8)
        >>> # Print edges in the graph
        >>> g.e[0]
        [(0, 1), (2, 4), (0, 4), (3, 4), (0, 3), (2, 3), (0, 2), (1, 3)]
        >>> # Print vertex features
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])
        >>> # Fuse features learned from different domains
        >>> X_ = (g.smoothing_with_GCN(X) + g.v2v(X, aggr="mean"))/2
        >>> X_
        tensor([[0.5271, 0.5998],
                [0.5773, 0.7091],
                [0.4699, 0.7038],
                [0.4774, 0.6174],
                [0.3968, 0.6582]])


融合从不同关联结构中学习到的特征
++++++++++++++++++++++++++++++++++++++++++++++++++

在如下例子中，我们在相同顶点集上随机构建一个 **图** 结构和一个 **超图** 结构。
然后，采用两种关联结构中的消息传递函数来生成不同的顶点特征，通过它们的组合连接生成最终的混合顶点特征。

    .. code:: python

        >>> import torch
        >>> import dhg
        >>> # Generate the vertex features
        >>> X = torch.rand(5, 2)
        >>> # Generate the low-order structure on the vertex set
        >>> g = dhg.random.graph_Gnm(5, 8)
        >>> # Generate the high-order structure on the vertex set
        >>> hg = dhg.random.hypergraph_Gnm(5, 4)
        >>> # Print information before message passing
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])
        >>> g.e[0]
        [(0, 1), (2, 4), (0, 4), (3, 4), (0, 3), (2, 3), (0, 2), (1, 3)]
        >>> hg.e[0]
        [(0, 1), (0, 3, 4), (1, 2, 3), (1, 3)]
        >>> X_low = g.v2v(X, aggr="mean")
        >>> X_high = hg.v2v(X, aggr="mean")
        >>> X_ = torch.cat([X_low, X_high], dim=1)
        >>> # Print new vertex features
        >>> X_
        tensor([[0.5107, 0.5386, 0.5642, 0.7151],
                [0.5946, 0.8515, 0.6265, 0.5799],
                [0.5512, 0.7786, 0.5261, 0.5072],
                [0.4113, 0.5738, 0.6178, 0.6223],
                [0.4051, 0.6875, 0.5512, 0.7786]])
