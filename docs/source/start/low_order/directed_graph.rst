
Learning on Directed Graphs
=============================

Definition
-----------------------
Directed graphs can be 

Construction
-------------------------

Generate a directed graph structure and a feature matrix.

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

Propagate messages on the directed graph structure


**Spectral-Based Feature Smoothing on Directed Graphs**

Welcome to contribute!

Spatial-Based Learning
---------------------------------


Spatial-Based Message Propagation on Directed Graphs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python

        >>> # Print the vertex messages
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329]])
        >>> X_dst2src = g.v2v(X, aggr="mean", direction="dst2src")
        >>> X_src2dst = g.v2v(X, aggr="mean", direction="src2dst")
        >>> # Print the new vertex messages
        >>> X_dst2src
        tensor([[0.6288, 0.7070],
                [0.0000, 0.0000],
                [0.6288, 0.7070],
                [0.2453, 0.4962],
                [0.2110, 0.6407]])
        >>> X_src2dst
        tensor([[0.4643, 0.6329],
                [0.0000, 0.0000],
                [0.6288, 0.7070],
                [0.2110, 0.6407],
                [0.4051, 0.6875]])

