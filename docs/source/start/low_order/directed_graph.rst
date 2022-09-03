
Learning on Directed Graph
=============================

.. hint:: 

    - Author: `Yifan Feng (丰一帆) <https://fengyifan.site/>`_
    - Proof: Xinwei Zhang

Definition
-----------------------

A `directed graph <https://en.wikipedia.org/wiki/Directed_graph>`_ can be indicated with :math:`\mathcal{G} = \{\mathcal{V}, \mathcal{E}\}`.

- :math:`\mathcal{V}`, is a set of **vertices** (also called **nodes** or **points**);
- :math:`\mathcal{E} \subseteq \{ (x, y) \mid (x, y) \in \mathcal{V}^2~and~x \neq y \}`, a set of **edges** (also called **directed edges**, **directed links**, **directed lines**, **arrow**, or **arcs**), 
  which are `ordered pairs <https://en.wikipedia.org/wiki/Ordered_pair>`_ of vertices (that is, an edge is associated with two distinct vertices).

In the edge :math:`(x, y)`, the vertices :math:`x` and :math:`y` are called the **endpoints** of the edge,
:math:`x` is the **source** (also called **tail**) of the edge and :math:`y` is the **target** (also called **head**) of the edge.
The edge is said to **join** :math:`x` and :math:`y` and to be **incident** on :math:`x` and on :math:`y`. 
A vertex may exist in a directed graph and not belong to an edge. The edge :math:`(y, x)` is called the inverted edge of :math:`(x, y)`. 
`Multiple edges <https://en.wikipedia.org/wiki/Multiple_edges>`_, not allowed under the definition above, are two or more edges with both the same tail and the same head.


Construction
-------------------------

The directed graph structure can be constructed by the following methods. More details can refer to :ref:`here <build_directed_graph>`.

- Edge list (**default**) :py:class:`dhg.DiGraph`
- Adjacency list :py:meth:`dhg.DiGraph.from_adj_list`
- Features with k-Nearest Neighbors :py:meth:`dhg.DiGraph.from_feature_kNN`

In the following example, we randomly general a directed graph structure and a feature matrix to perform some basic learning operations on this structure.

    .. code:: python

        >>> import torch
        >>> import dhg
        >>> dhg.random.set_seed(2022)
        >>> # Generate a random directed graph with 5 vertices and 8 edges
        >>> g = dhg.random.digraph_Gnm(5, 8) 
        >>> # Generate a vertex feature matrix with size 5x2
        >>> X = torch.rand(5, 2)
        >>> # Print information about the directed graph and feature
        >>> g 
        Directed Graph(num_v=5, num_e=8)
        >>> # Print edges in the graph
        >>> g.e[0]
        [(2, 4), (4, 0), (0, 4), (3, 4), (0, 3), (4, 2), (2, 3), (3, 2)]
        >>> # Print vertex features
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])


.. Structure Visualization
.. ---------------------------------

.. Draw the directed graph structure

..     .. code:: python

..         >>> fig = g.draw(edge_style="line")
..         >>> fig.show()

..     Here is the image.


Spectral-Based Learning
---------------------------------

Welcome to contribute!


Spatial-Based Learning
---------------------------------


Message Propagation from Source Vertex to Target Vertex
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python

        >>> # Print the vertex messages
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])
        >>> X_src2dst = g.v2v(X, aggr="mean", direction="src2dst")
        >>> # Print the new vertex messages
        >>> X_src2dst
        tensor([[0.4643, 0.6329],
                [0.0000, 0.0000],
                [0.6288, 0.7070],
                [0.2110, 0.6407],
                [0.4051, 0.6875]])


Message Propagation from Source Vertex to Target Vertex with different Edge Weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python

        >>> # Print the vertex messages
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])
        >>> g.e_weight
        tensor([1., 1., 1., 1., 1., 1., 1., 1.])
        >>> # Generate random edge weights
        >>> e_weight = torch.rand(len(g.e_weight))
        >>> e_weight
        tensor([0.6689, 0.2302, 0.8003, 0.7353, 0.7477, 0.5585, 0.6226, 0.8429])
        >>> X_ = g.v2v(X, e_weight=e_weight, aggr="softmax_then_sum", direction="src2dst")
        >>> # Print the new vertex messages
        >>> X_
        tensor([[0.4643, 0.6329],
                [0.0000, 0.0000],
                [0.6210, 0.7035],
                [0.1989, 0.6222],
                [0.3809, 0.6432]])


Message Propagation from Target Vertex to Source Vertex
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python

        >>> # Print the vertex messages
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])
        >>> X_dst2src = g.v2v(X, aggr="mean", direction="dst2src")
        >>> # Print the new vertex messages
        >>> X_dst2src
        tensor([[0.6288, 0.7070],
                [0.0000, 0.0000],
                [0.6288, 0.7070],
                [0.2453, 0.4962],
                [0.2110, 0.6407]])


Message Propagation from Target Vertex to Source Vertex with different Edge Weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python

        >>> # Print the vertex messages
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])
        >>> g.e_weight
        tensor([1., 1., 1., 1., 1., 1., 1., 1.])
        >>> # Generate random edge weights
        >>> e_weight = torch.rand(len(g.e_weight))
        >>> e_weight
        tensor([0.6689, 0.2302, 0.8003, 0.7353, 0.7477, 0.5585, 0.6226, 0.8429])
        >>> X_ = g.v2v(X, e_weight=e_weight, aggr="softmax_then_sum", direction="dst2src")
        >>> # Print the new vertex messages
        >>> X_
        tensor([[0.6644, 0.7230],
                [0.0000, 0.0000],
                [0.6342, 0.7094],
                [0.2246, 0.4832],
                [0.1907, 0.6098]])
