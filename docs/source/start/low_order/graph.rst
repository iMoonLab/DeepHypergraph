Learning on Graph
=============================

.. hint:: 

    - Author: `Yifan Feng (丰一帆) <https://fengyifan.site/>`_
    - Proof: Xinwei Zhang

Definition
-------------------------
A `graph <https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)>`_ can be indicated with :math:`\mathcal{G} = \{\mathcal{V}, \mathcal{E}\}`.

- :math:`\mathcal{V}`, is a set of **vertices** (also called **nodes** or **points**);
- :math:`\mathcal{E} \subseteq \{ \{x, y\} \mid x, y \in \mathcal{V}~and~x \neq y \}`, a set of **edges** (also called **links** or **lines**), 
  which are `unordered pairs <https://en.wikipedia.org/wiki/Unordered_pair>`_ of vertices (that is, an edge is associated with two distinct vertices).

In the edge :math:`\{x, y\}`, the vertices :math:`x` and :math:`y` are called the **endpoints** of the edge. 
The edge is said to **join** :math:`x` and :math:`y` and to be **incident** on :math:`x` and on :math:`y`. 
A vertex may exist in a graph and not belong to an edge.
`Multiple edges <https://en.wikipedia.org/wiki/Multiple_edges>`_, not allowed under the definition above, are two or more edges that join the same two vertices.

Construction
-------------------------
The graph structure can be constructed by the following methods. More details can refer to :ref:`here <build_graph>`.

- Edge list (**default**) :py:class:`dhg.Graph`
- Adjacency list :py:meth:`dhg.Graph.from_adj_list`
- Reduced from the hypergraph structure
  
  - Star expansion :py:meth:`dhg.Graph.from_hypergraph_star`
  - Clique expansion :py:meth:`dhg.Graph.from_hypergraph_clique`
  - `HyperGCN <https://arxiv.org/pdf/1809.02589.pdf>`_-based expansion :py:meth:`dhg.Graph.from_hypergraph_hypergcn`

In the following example, we randomly generate a graph structure and a feature matrix to perform some basic learning operations on this structure.

    .. code:: python

        >>> import torch
        >>> import dhg
        >>> # Generate a random graph with 5 vertices and 8 edges
        >>> g = dhg.random.graph_Gnm(5, 8) 
        >>> # Generate a vertex feature matrix with size 5x2
        >>> X = torch.rand(5, 2)
        >>> # Print information about the graph and feature
        >>> g 
        Graph(num_v=5, num_e=8)
        >>> # Print edges in the graph
        >>> g.e[0]
        [(0, 1), (2, 4), (0, 4), (3, 4), (0, 3), (2, 3), (0, 2), (1, 3)]
        >>> # Print the both side of edges in the graph
        >>> g.e_both_side[0]
        [(0, 1), (2, 4), (0, 4), (3, 4), (0, 3), (2, 3), (0, 2), (1, 3), (1, 0), (4, 2), (4, 0), (4, 3), (3, 0), (3, 2), (2, 0), (3, 1)]
        >>> # Print vertex features
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])

.. Structure Visualization
.. -------------------------
.. Draw the graph structure

..     .. code:: python

..         >>> fig = g.draw(edge_style="line")
..         >>> fig.show()

..     Here is the image.


Spectral-Based Learning
-------------------------

Smoothing with GCN's Laplacian
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python

        >>> # Print the Laplacian matrix defined by GCN that associated with the graph structure
        >>> g.L_GCN.to_dense()
        tensor([[0.2000, 0.2582, 0.2236, 0.2000, 0.2236],
                [0.2582, 0.3333, 0.0000, 0.2582, 0.0000],
                [0.2236, 0.0000, 0.2500, 0.2236, 0.2500],
                [0.2000, 0.2582, 0.2236, 0.2000, 0.2236],
                [0.2236, 0.0000, 0.2500, 0.2236, 0.2500]])
        >>> # Print the vertex features befor feautre smoothing
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])
        >>> X_ = g.smoothing_with_GCN(X)
        >>> # Print the vertex features after GCN-based smoothing
        >>> X_
        tensor([[0.5434, 0.6609],
                [0.5600, 0.5668],
                [0.3885, 0.6289],
                [0.5434, 0.6609],
                [0.3885, 0.6289]])

Smoothing with Symmetrically Normalized Laplacian
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python

        >>> # Print the symmetrically normalized Laplacian matrix associated with the graph structure
        >>> g.L_sym.to_dense()
        tensor([[ 1.0000, -0.3536, -0.2887, -0.2500, -0.2887],
                [-0.3536,  1.0000,  0.0000, -0.3536,  0.0000],
                [-0.2887,  0.0000,  1.0000, -0.2887, -0.3333],
                [-0.2500, -0.3536, -0.2887,  1.0000, -0.2887],
                [-0.2887,  0.0000, -0.3333, -0.2887,  1.0000]])
        >>> # Print the vertex features
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])
        >>> X_ = g.smoothing(X, g.L_sym, 0.1)
        >>> # print the new vertex features
        >>> X_
        tensor([[ 0.3746,  0.9525],
                [ 0.7926,  0.3590],
                [-0.0210,  0.3251],
                [ 0.8218,  0.7940],
                [ 0.4756,  0.6351]])

Smoothing with Left (random-walk) Normalized Laplacian
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python

        >>> # Print the left(random-walk) normalized Laplacian matrix associated with the graph structure
        >>> g.L_rw.to_dense()
        tensor([[ 1.0000, -0.2500, -0.2500, -0.2500, -0.2500],
                [-0.5000,  1.0000,  0.0000, -0.5000,  0.0000],
                [-0.3333,  0.0000,  1.0000, -0.3333, -0.3333],
                [-0.2500, -0.2500, -0.2500,  1.0000, -0.2500],
                [-0.3333,  0.0000, -0.3333, -0.3333,  1.0000]])
        >>> # Print the vertex features
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])
        >>> X_ = g.smoothing(X, g.L_rw, 0.1)
        >>> # Print the new vertex features
        >>> X_
        tensor([[ 0.3843,  0.9603],
                [ 0.7752,  0.3341],
                [-0.0263,  0.3174],
                [ 0.8316,  0.8018],
                [ 0.4703,  0.6275]])


Spatial-Based Learning
----------------------------

Message Propagation from Vertex to Vertex
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python

        >>> # Print the vertex messages
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])
        >>> X_ = g.v2v(X, aggr="mean") 
        >>> # Print the new vertex messages
        >>> X_
        tensor([[0.5107, 0.5386],
                [0.5946, 0.8515],
                [0.5512, 0.7786],
                [0.4113, 0.5738],
                [0.4051, 0.6875]])

Message Propagation from Vertex to Vertex with different Edge Weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python

        >>> # Print the vertex messages
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])
        >>> g.e_weight
        tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        >>> # Generate random edge weights
        >>> e_weight = torch.rand(len(g.e_weight))
        >>> e_weight
        tensor([0.6689, 0.2302, 0.8003, 0.7353, 0.7477, 0.5585, 0.6226, 0.8429, 0.6105,
                0.1248, 0.8265, 0.2117, 0.8574, 0.4282, 0.3964, 0.1440])
        >>> X_ = g.v2v(X, e_weight=e_weight, aggr="softmax_then_sum")
        >>> # Print the new vertex messages
        >>> X_
        tensor([[0.5648, 0.5657],
                [0.5758, 0.8582],
                [0.5699, 0.7794],
                [0.4720, 0.5493],
                [0.3742, 0.6827]])

