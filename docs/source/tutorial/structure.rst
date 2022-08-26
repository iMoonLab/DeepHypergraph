Build Structure
===================================
Correlation structures are the core of **DHG**. In this section, we introduce the basic construction methods of different structures 
and some structure transformation functions of them like: *reducing the high-order structrue to the low-order structure* 
and *promoting the low-order structure to the high-order structure.*

Low-Order Structures
-----------------------

Build Simple Graph
+++++++++++++++++++++++

A `simple graph <https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)>`_ is a graph with no loops and no multiple edges, which can be constructed by the following methods:

- Edge list (**default**) :py:class:`dhg.Graph`
- Adjacency list :py:meth:`dhg.Graph.from_adj_list`
- Reduced from the simple hypergraph structure
  
  - Star expansion :py:meth:`dhg.Graph.from_hypergraph_star`
  - Clique expansion :py:meth:`dhg.Graph.from_hypergraph_clique`
  - `HyperGCN <https://arxiv.org/pdf/1809.02589.pdf>`_-based expansion :py:meth:`dhg.Graph.from_hypergraph_hypergcn`

Common Methods
^^^^^^^^^^^^^^^^^^^

Construction from Edge List:

.. code-block:: python

    >>> import dhg
    >>> g = dhg.Graph([(0, 1), (0, 2), (1, 2)])
    >>> g.e
    # [(0, 1), (0, 2), (1, 2)]

Reduced from High-Order Structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Build Directed Graph
+++++++++++++++++++++++

A `directed graph <https://en.wikipedia.org/wiki/Directed_graph>`_ is a graph

Common Methods
^^^^^^^^^^^^^^^^^^^

Reduced from High-Order Structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


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

Reduced from High-Order Structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


High-Order Structures
-----------------------

Build Simple Hypergraph
++++++++++++++++++++++++++

Common Methods
^^^^^^^^^^^^^^^^^^^

Prometed from Low-Order Structures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


