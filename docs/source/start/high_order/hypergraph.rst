.. _start_learning_on_simple_hypergraph:

Learning on Hypergraph
=================================

.. hint:: 

    - Author: `Yifan Feng (丰一帆) <https://fengyifan.site/>`_
    - Proof: Xinwei Zhang

Definition
-----------------
A `hypergraph <https://en.wikipedia.org/wiki/Hypergraph>`_ (also called undirected hypergraph) can be indicated with :math:`\mathcal{G} = \{\mathcal{V}, \mathcal{E}\}`.

- :math:`\mathcal{V}`, is a set of **vertices** (also called **nodes** or **points**);
- :math:`\mathcal{E} \subseteq \{ \mathcal{P}(\mathcal{V}) \}`, a set of **hyperedges** (also called **edges**), where :math:`\mathcal{P}(\mathcal{V})` is the `power set <https://en.wikipedia.org/wiki/Power_set>`_ of :math:`\mathcal{V}`.
  Each hyperedge :math:`e \in \mathcal{E}` can contains two or more vertices.

While graph edges connect only 2 vertices, hyperedges connect an arbitrary number of vertices. 
However, it is often desirable to study hypergraph where all hyperedges have the same cardinality; 
a k-uniform hypergraph is a hypergraph such that all its hyperedges have size k. 
(In other words, one such hypergraph is a collection of sets, 
each such set a hyperedge connecting k nodes.) So a 2-uniform hypergraph is a graph, 
a 3-uniform hypergraph is a collection of unordered triples, and so on. 
An undirected hypergraph is also called a set system or a family of sets drawn from the universal set.


Construction
---------------------
The hypergraph structure can be constructed by the following methods. More details can refer to :ref:`here <build_hypergraph>`.

- Hyperedge list (**default**) :py:class:`dhg.Hypergraph`
- Features with k-Nearest Neighbors :py:meth:`dhg.Hypergraph.from_feature_kNN`
- Promoted from the low-order structures

  - Graph :py:meth:`dhg.Hypergraph.from_graph`
  - k-Hop Neighbors of vertices in a graph :py:meth:`dhg.Hypergraph.from_graph_kHop`
  - Bipartite Graph :py:meth:`dhg.Hypergraph.from_bigraph`


In the following example, we randomly generate a hypergraph structure and a feature matrix to perform some basic learning operations on this structure.
   
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

Message Propagation from Vertex to Hyperedge
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

Message Propagation from Vertex to Hyperedge with different Edge Weights
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


Message Propagation from Hyperedge to Vertex
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


Message Propagation from Hyperedge to Vertex with different Edge Weights
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

Message Propagation from Vertex Set to Vertex Set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each hyperedge connects a set of vertices, and it is a message bridge between two sets of vertices.
In hypergraph, the source vertex set and the target vertex set that the hyperedge connects are the same.

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

Message Propagation from Vertex Set to Vertex Set with different Edge Weights in Two Stages
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

