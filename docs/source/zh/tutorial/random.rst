随机结构生成
=======================

.. hint:: 

    - 作者:  `丰一帆 <https://fengyifan.site/>`_
    - 翻译:  颜杰龙
    - 校对： `丰一帆 <https://fengyifan.site/>`_ 、张欣炜

在这一节里，我们提供在DHG中如何生成随机关联结构的模版。

DHG的结构生成器名称可以分为以下两类：

- ``Gnm``: 生成包含 ``n`` 个顶点和 ``m`` 条边/超边的随机关联结构。
- ``Gnp``: 生成包含 ``n`` 个顶点并且以概率 ``p`` 选择边/超边。


随机图生成
--------------------------------

生成包含 ``n`` 个顶点和 ``m`` 条边的图：

.. code-block:: python

    >>> import dhg.random as dr
    >>> g = dr.graph_Gnm(10, 20)
    >>> g
    Graph(num_v=10, num_e=20)

生成包含 ``n`` 个顶点并且以概率 ``p`` 选择边的图：

.. code-block:: python

    >>> import dhg.random as dr
    >>> g = dr.graph_Gnp(10, 0.5)
    >>> g
    Graph(num_v=10, num_e=24)
    >>> g = dr.graph_Gnp_fast(10, 0.5)
    >>> g
    Graph(num_v=10, num_e=22)


随机有向图生成
-------------------------------------

生成包含 ``n`` 个顶点和 ``m`` 条边的有向图：

.. code-block:: python

    >>> import dhg.random as dr
    >>> g = dr.digraph_Gnm(10, 20)
    >>> g
    Directed Graph(num_v=10, num_e=20)

生成包含 ``n`` 个顶点并且以概率 ``p`` 选择边的有向图：

.. code-block:: python

    >>> import dhg.random as dr
    >>> g = dr.digraph_Gnp(10, 0.5)
    >>> g
    Directed Graph(num_v=10, num_e=39)
    >>> g = dr.digraph_Gnp_fast(10, 0.5)
    >>> g
    Directed Graph(num_v=10, num_e=35)

随机二分图生成
-------------------------------------

生成顶点集 :math:`U` 包含 ``num_u`` 个顶点、顶点集 :math:`V` 包含 ``num_v`` 个顶点和 ``m`` 条边的二分图：

.. code-block:: python

    >>> import dhg.random as dr
    >>> g = dr.bigraph_Gnm(5, 6, 8)
    >>> g
    Bipartite Graph(num_u=5, num_v=6, num_e=8)

生成顶点集 :math:`U` 包含 ``num_u`` 个顶点、顶点集 :math:`V` 包含 ``num_v`` 个顶点并且以概率 ``p`` 选择边的二分图：

.. code-block:: python

    >>> import dhg.random as dr
    >>> g = dr.bigraph_Gnp(5, 6, 0.5)
    >>> g
    Bipartite Graph(num_u=5, num_v=6, num_e=19)

随机超图生成
-------------------------------------

超图生成器可以分为以下两类：

- ``k``-均匀超图：每条超边含有相同数量（k）的顶点。
- 一般超图：每条超边含有的顶点数量随机。

生成包含 ``n`` 个顶点和 ``m`` 条超边的 ``k`` -均匀超图：

.. code-block:: python

    >>> import dhg.random as dr
    >>> hg = dr.uniform_hypergraph_Gnm(3, 20, 5)
    >>> hg
    Hypergraph(num_v=20, num_e=5)
    >>> hg.e
    ([(2, 11, 12), (4, 14, 18), (0, 5, 16), (2, 6, 12), (1, 3, 6)], [1.0, 1.0, 1.0, 1.0, 1.0])

生成包含 ``n`` 个顶点并且以概率 ``p`` 选择超边的 ``k`` -均匀超图：

.. code-block:: python

    >>> import dhg.random as dr
    >>> hg = dr.uniform_hypergraph_Gnp(3, 20, 0.01)
    >>> hg
    Hypergraph(num_v=20, num_e=8)
    >>> hg.e
    ([(1, 6, 16), (2, 17, 18), (3, 14, 16), (5, 9, 17), (7, 12, 14), (10, 18, 19), (12, 13, 19), (12, 18, 19)], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

生成包含 ``n`` 个顶点和 ``m`` 条超边的一般超图：

.. code-block:: python

    >>> import dhg.random as dr
    >>> hg = dr.hypergraph_Gnm(8, 4)
    >>> hg
    Hypergraph(num_v=8, num_e=4)
    >>> hg.e
    ([(0, 2, 5, 6, 7), (3, 4), (0, 1, 4, 5, 6, 7), (2, 5, 6)], [1.0, 1.0, 1.0, 1.0])

