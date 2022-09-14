Structure Generation
=======================

.. hint:: 

    - Author: `Yifan Feng (丰一帆) <https://fengyifan.site/>`_
    - Proof: Xinwei Zhang

In this section, we provide examples of how to generate random correlation structures of DHG.

The name of DHG's structure generator can be divided into types:

- ``Gnm``: Generate a random structure with ``n`` vertices and ``m`` edges/hyperedges.
- ``Gnp``: Generate a random structure with ``n`` vertices and ``p`` probability of choosing an/a edge/hyperedge.


Random Graph Generation
--------------------------------

Generating a graph with ``n`` vertices and ``m`` edges:

.. code-block:: python

    >>> import dhg.random as dr
    >>> g = dr.graph_Gnm(10, 20)
    >>> g
    Graph(num_v=10, num_e=20)

Generating a graph with ``n`` vertices and ``p`` probability of choosing an edge:

.. code-block:: python

    >>> import dhg.random as dr
    >>> g = dr.graph_Gnp(10, 0.5)
    >>> g
    Graph(num_v=10, num_e=24)
    >>> g = dr.graph_Gnp_fast(10, 0.5)
    >>> g
    Graph(num_v=10, num_e=22)


Random Directed Graph Generation
-------------------------------------

Generating a directed graph with ``n`` vertices and ``m`` edges:

.. code-block:: python

    >>> import dhg.random as dr
    >>> g = dr.digraph_Gnm(10, 20)
    >>> g
    Directed Graph(num_v=10, num_e=20)

Generating a directed graph with ``n`` vertices and ``p`` probability of choosing an edge:

.. code-block:: python

    >>> import dhg.random as dr
    >>> g = dr.digraph_Gnp(10, 0.5)
    >>> g
    Directed Graph(num_v=10, num_e=39)
    >>> g = dr.digraph_Gnp_fast(10, 0.5)
    >>> g
    Directed Graph(num_v=10, num_e=35)

Random Bipartite Graph Generation
-------------------------------------

Generating a bipartite graph with ``num_u`` vertices in set :math:`U`, ``num_v`` vertices in set :math:`V`, and ``m`` edges:

.. code-block:: python

    >>> import dhg.random as dr
    >>> g = dr.bigraph_Gnm(5, 6, 8)
    >>> g
    Bipartite Graph(num_u=5, num_v=6, num_e=8)

Generating a bipartite graph with ``num_u`` vertices in set :math:`U`, ``num_v`` vertices in set :math:`V`, and ``p`` probability of choosing an edge:

.. code-block:: python

    >>> import dhg.random as dr
    >>> g = dr.bigraph_Gnp(5, 6, 0.5)
    >>> g
    Bipartite Graph(num_u=5, num_v=6, num_e=19)

Random Hypergraph Generation
-------------------------------------

The hypergraph generator can be divided into two types:

- ``k``-uniform hypergraph: Each hyperedge has the same number (k) of vertices.
- General hypergraph: Each hyperedge has a random number of vertices.

Generating a ``k``-uniform hypergraph with ``n`` vertices and ``m`` hyperedges:

.. code-block:: python

    >>> import dhg.random as dr
    >>> hg = dr.uniform_hypergraph_Gnm(3, 20, 5)
    >>> hg
    Hypergraph(num_v=20, num_e=5)
    >>> hg.e
    ([(2, 11, 12), (4, 14, 18), (0, 5, 16), (2, 6, 12), (1, 3, 6)], [1.0, 1.0, 1.0, 1.0, 1.0])

Generating a ``k``-uniform hypergraph with ``n`` vertices and ``p`` probability of choosing a hyperedge:

.. code-block:: python

    >>> import dhg.random as dr
    >>> hg = dr.uniform_hypergraph_Gnp(3, 20, 0.01)
    >>> hg
    Hypergraph(num_v=20, num_e=8)
    >>> hg.e
    ([(1, 6, 16), (2, 17, 18), (3, 14, 16), (5, 9, 17), (7, 12, 14), (10, 18, 19), (12, 13, 19), (12, 18, 19)], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

Generating a general hypergraph with ``n`` vertices and ``m`` hyperedges:

.. code-block:: python

    >>> import dhg.random as dr
    >>> hg = dr.hypergraph_Gnm(8, 4)
    >>> hg
    Hypergraph(num_v=8, num_e=4)
    >>> hg.e
    ([(0, 2, 5, 6, 7), (3, 4), (0, 1, 4, 5, 6, 7), (2, 5, 6)], [1.0, 1.0, 1.0, 1.0])

