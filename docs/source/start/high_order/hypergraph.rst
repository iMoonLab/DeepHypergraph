.. _start_learning_on_simple_hypergraph:

Learning on Simple Hypergraphs
=================================

Definition
-----------------
`Simple hypergraphs <https://en.wikipedia.org/wiki/Hypergraph>`_ (also called undirected hypergraphs) can be indicated with :math:`\mathcal{G} = \{\mathcal{V}, \mathcal{E}\}`.

- :math:`\mathcal{V}`, is a set of **vertices** (also called **nodes** or **points**);
- :math:`\mathcal{E} \subseteq \{ \mathcal{P}(\mathcal{V}) \}`, a set of **hyperedges** (also called **edges**), where :math:`\mathcal{P}(\mathcal{V})` is the `power set <https://en.wikipedia.org/wiki/Power_set>`_ of :math:`\mathcal{V}`.
  Each hyperedge :math:`e \in \mathcal{E}` can contains two or more vertices.

While graph edges connect only 2 vertices, hyperedges connect an arbitrary number of vertices. 
However, it is often desirable to study hypergraphs where all hyperedges have the same cardinality; 
a k-uniform hypergraph is a hypergraph such that all its hyperedges have size k. 
(In other words, one such hypergraph is a collection of sets, 
each such set a hyperedge connecting k nodes.) So a 2-uniform hypergraph is a graph, 
a 3-uniform hypergraph is a collection of unordered triples, and so on. 
An undirected hypergraph is also called a set system or a family of sets drawn from the universal set.


Construction
---------------------
The simple hypergraph structure can be constructed by the following methods. More details can refer to aaaaaaaaaaaaa.

- Hyperedge list (**default**) :py:class:`dhg.Hypergraph`
- Features with k-Nearest Neighbors :py:meth:`dhg.Hypergraph.from_feature_kNN`
- Promoted from the low-order structures

    - Simple Graph :py:meth:`dhg.Hypergraph.from_graph`
    - k-Hop Neighbors of vertices in a simple graph :py:meth:`dhg.Hypergraph.from_graph_kHop`
    - Bipartite Graph :py:meth:`dhg.Hypergraph.from_bigraph`


In the following example, we randomly generate a simple hypergraph structure and a feature matrix to perform some basic learning operations on this structure.
   
    .. code:: python

        >>> import torch
        >>> import dhg
        >>> # Generate a random simple hypergraph with 5 vertices and 4 hyperedges
        >>> hg = dhg.random.hypergraph_Gnm(5, 4) 
        >>> # Generate a vertex feature matrix with size 5x2
        >>> X = torch.rand(5, 2)
        >>> # Print information about the hypergraph and feature
        >>> hg 
        Simple Hypergraph(num_v=5, num_e=4)
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

Spectral-Based Learning
-------------------------------

Smoothing with HGNN's Laplacian
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

Spatial-Based Learning
-------------------------------

Message Propagation on Simple Hypergraphs
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
        >>> # Message propagation from hyperedge to vertex
        >>> X_ = hg.e2v(Y_, aggr="mean")
        >>> # Print the new hyperedge messages
        >>> Y_
        tensor([[0.4098, 0.5702],
                [0.2955, 0.6381],
                [0.4280, 0.5911],
                [0.5107, 0.5386]])
        >>> # Print the new vertex messages
        >>> X_
        tensor([[0.2955, 0.6381],
                [0.5107, 0.5386],
                [0.4110, 0.5845],
                [0.4495, 0.5667],
                [0.4114, 0.5893]])
        >>> # Or you can use the combination function of v2e and e2v
        >>> X_ = hg.v2v(X, aggr="mean")
        >>> # Print the new vertex messages
        >>> X_
        tensor([[0.2955, 0.6381],
                [0.5107, 0.5386],
                [0.4110, 0.5845],
                [0.4495, 0.5667],
                [0.4114, 0.5893]])


