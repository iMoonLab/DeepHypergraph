Build Structure
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

**Construct a simple graph from edge list with** :py:class:`dhg.Graph`

.. code-block:: python

    >>> import dhg
    >>> g = dhg.Graph(5, [(0, 1), (0, 2), (1, 2), (3, 4)])
    >>> g
    Simple Graph(num_v=5, num_e=4)
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
    >>> g.A.to_dense()
    tensor([[0., 1., 1., 0., 0.],
            [1., 0., 1., 0., 0.],
            [1., 1., 0., 0., 0.],
            [0., 0., 0., 0., 1.],
            [0., 0., 0., 1., 0.]])

The ``g.e`` attribute will return a tuple of two lists, the first list is the edge list and the second list is a list of weight for each edge.
The ``g.e_both_size`` attribute will return the both side of edges in the simple graph.

.. important:: 

    In simple graph the edge is unordered pair, which means ``(0, 1)`` and ``(1, 0)`` are the same edge. Adding edges ``(0, 1)`` and ``(1, 0)`` is equivalent to adding edge ``(0, 1)`` twice.


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

The adjacency list is a list of lists, each list contains two parts. The first part is the **first element** of the list, which is the vertex index of the source vertex. 
The second part is the **remaining elements** of the list, which are the vertex indices of the destination vertices.
For example, assuming we have a graph with 5 vertices and a adjacency list as:

.. code-block:: text

    [[0, 1, 2], [0, 3], [1, 2], [3, 4]]

Then, the transformed edge list is:

.. code-block:: text

    [(0, 1), (0, 2), (0, 3), (1, 2), (3, 4)]

We can construct a simple graph from the adjacency list as:

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


Reduced from High-Order Structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We first define a simple hypergraph as:

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

**Star Expansion** :py:meth:`dhg.Graph.from_hypergraph_star`

The star expansion will treat the hyperedges in the hypergraph as virtual vertices in the simple graph. 
Each virtual vertex will connect to all the vertices in the hyperedge. 
The :py:meth:`dhg.Graph.from_hypergraph_star` function will return two values.
The first value is the reduced simple graph and the second value is a ``vertex mask`` that indicates whether the vertex is a actual vertex.
The ``True`` in the ``vertex mask`` indicates the vertex is a actual vertex and the ``False`` indicates the vertex is a virtual vertex that is transformed from a hyperedge.

.. code-block:: python

    >>> g, v_mask = dhg.Graph.from_hypergraph_star(hg)
    >>> g
    Simple Graph(num_v=9, num_e=11)
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

**Clique Expansion** :py:meth:`dhg.Graph.from_hypergraph_clique`

Unlike the star expansion, the clique expansion will not add any virtual vertex to the simple graph. 
It is designed to reduce the hyperedges in the simple hypergraph to the edges in the simple graph.
For each hyperedge, the clique expansion will add edges to any two vertices in the hyperedge.

.. code-block:: python

    >>> g = dhg.Hypergraph.from_hypergraph_clique(hg)
    >>> g = dhg.Graph.from_hypergraph_clique(hg)
    >>> g
    Simple Graph(num_v=5, num_e=8)
    >>> g.e
    ([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (2, 3), (3, 4)], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    >>> g.A.to_dense()
    tensor([[0., 1., 1., 1., 1.],
            [1., 0., 1., 1., 0.],
            [1., 1., 0., 1., 0.],
            [1., 1., 1., 0., 1.],
            [1., 0., 0., 1., 0.]])

**HyperGCN-based Expansion** :py:meth:`dhg.Graph.from_hypergraph_hypergcn`

In the `HyperGCN <https://arxiv.org/pdf/1809.02589.pdf>`_ paper, the authors also describe 
a method to reduce the hyperedges in the hypergraph to the edges in the simple graph as the following figure.

.. image:: ../_static/img/hypergcn.png
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
    Simple Graph(num_v=5, num_e=4)
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
    Simple Graph(num_v=5, num_e=6)
    >>> g.e
    ([(1, 2), (0, 1), (2, 3), (1, 3), (3, 4), (0, 3)], [0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408, 0.3333333432674408])
    >>> g.A.to_dense()
    tensor([[0.0000, 0.3333, 0.0000, 0.3333, 0.0000],
            [0.3333, 0.0000, 0.3333, 0.3333, 0.0000],
            [0.0000, 0.3333, 0.0000, 0.3333, 0.0000],
            [0.3333, 0.3333, 0.3333, 0.0000, 0.3333],
            [0.0000, 0.0000, 0.0000, 0.3333, 0.0000]])


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

Welcome to contribute!


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


