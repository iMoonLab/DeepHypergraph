构建关联结构
===================================
Correlation structures are the core of **DHG**. In this section, we introduce the basic construction methods of different structures 
and some structure transformation functions of them like: *reducing the high-order structrue to the low-order structure* 
and *promoting the low-order structure to the high-order structure.*

Low-Order Structures
-----------------------

Build Simple Graph
+++++++++++++++++++++++

A `simple graph <https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)>`_ is a graph with no loops and no multiple edges, where the edge ``(x, y)`` and ``(y, x)`` are the same edge. 
It can be constructed by the following methods:

- Edge list (**default**) :py:class:`dhg.Graph`
- Adjacency list :py:meth:`dhg.Graph.from_adj_list`
- Reduced from the simple hypergraph structure
  
  - Star expansion :py:meth:`dhg.Graph.from_hypergraph_star`
  - Clique expansion :py:meth:`dhg.Graph.from_hypergraph_clique`
  - `HyperGCN <https://arxiv.org/pdf/1809.02589.pdf>`_-based expansion :py:meth:`dhg.Graph.from_hypergraph_hypergcn`

Common Methods
^^^^^^^^^^^^^^^^^^^

**Construct a simple graph from edge list with**, :py:class:`dhg.Graph`

.. code-block:: python

    >>> import dhg
    >>> g = dhg.Graph(5, [(0, 1), (0, 2), (1, 2), (3, 4)])
    >>> g.v
    [0, 1, 2, 3, 4]
    >>> g.e
    ([(0, 1), (0, 2), (1, 2), (3, 4)], [1.0, 1.0, 1.0, 1.0])
    >>> g.A.to_dense()
    tensor([[0., 1., 1., 0., 0.],
            [1., 0., 1., 0., 0.],
            [1., 1., 0., 0., 0.],
            [0., 0., 0., 0., 1.],
            [0., 0., 0., 1., 0.]])

The ``g.e`` attribute will return a tuple of two lists, the first list is the edge list and the second list is a list of weight for each edge.

.. important:: 

    In simple graph the edge is unordered pair, which means ``(0, 1)`` and ``(1, 0)`` are the same edge. Adding edges ``(0, 1)`` and ``(1, 0)`` is equivalent to adding one edge ``(0, 1)`` twice.


.. code-block:: python

    >>> g = dhg.Graph(5, [(0, 1), (0, 2), (2, 0), (3, 4)])
    >>> g.e
    ([(0, 1), (0, 2), (3, 4)], [1.0, 1.0, 1.0])
    >>> g.add_edges([(0, 1), (4, 3)])
    >>> g.e
    ([(0, 1), (0, 2), (3, 4)], [1.0, 1.0, 1.0])


.. note:: 

    If the added edges have duplicate edges, those duplicate edges will be automatically merged with specified ``merge_op``.

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


You can find the weight of the last edge is ``1.0`` and ``2.0``, if you set the ``merge_op`` to ``mean`` and ``sum``, respectively.


**Construct a simple graph from adjacency list with** :py:meth:`dhg.Graph.from_adj_list`

The adjacency list is a list of lists, each list contains two parts. The first part is the first element of the list, which is the vertex index of the source. 
The second part is the remaining elements of the list, which are the vertex indices of the destination vertices.
For example, assuming we have a graph with 5 vertices and a adjacency list as:

.. code-block:: text

    [[0, 1, 2], [0, 3], [1, 2], [3, 4]]

Then, the transformed edge list is:

.. code-block:: text

    [(0, 1), (0, 2), (0, 3), (1, 2), (3, 4)]

We can construct a simple graph from the adjacency list as:

.. code-block:: python

    >>> import dhg
    >>> g = dhg.Graph.from_adj_list(5, [[0, 1, 2], [1, 3], [4, 3, 0, 2, 1]])
    >>> g.e
    ([(0, 1), (0, 2), (1, 3), (3, 4), (0, 4), (2, 4), (1, 4)], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    >>> g.A.to_dense()
    tensor([[0., 1., 1., 0., 1.],
            [1., 0., 0., 1., 1.],
            [1., 0., 0., 0., 1.],
            [0., 1., 0., 0., 1.],
            [1., 1., 1., 1., 0.]])


Reduced from High-Order Structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Comming soon

Build Directed Graph
+++++++++++++++++++++++

A `directed graph <https://en.wikipedia.org/wiki/Directed_graph>`_ is a graph with directed edges, where the edge ``(x, y)`` and edge ``(y, x)`` can exist in the structure simultaneously.
It can be constructed by the following methods:

- Edge list (**default**) :py:class:`dhg.DiGraph`
- Adjacency list :py:meth:`dhg.DiGraph.from_adj_list`
- Features with k-Nearest Neighbors :py:meth:`dhg.DiGraph.from_feature_kNN`


Common Methods
^^^^^^^^^^^^^^^^^^^
Comming soon

Reduced from High-Order Structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Comming soon


Build Bipartite Graph
+++++++++++++++++++++++

A `bipartite graph <https://en.wikipedia.org/wiki/Bipartite_graph>`_ is a graph that contains two types of vertices and edges between them, 
whose partition has the parts vertex set :math:`\mathcal{U}` and vertex set :math:`\mathcal{V}`. 
It can be constructed by the following methods:

- Edge list (**default**) :py:class:`dhg.BiGraph`
- Adjacency list :py:meth:`dhg.BiGraph.from_adj_list`
- Simple hypergraph :py:meth:`dhg.BiGraph.from_hypergraph`

Common Methods
^^^^^^^^^^^^^^^^^^^
Comming soon

Reduced from High-Order Structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Comming soon


High-Order Structures
-----------------------
Comming soon

Build Simple Hypergraph
++++++++++++++++++++++++++
Comming soon

Common Methods
^^^^^^^^^^^^^^^^^^^
Comming soon

Prometed from Low-Order Structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Comming soon


