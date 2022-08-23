
.. _start_learning_on_bipartite_graph:

Learning on Bipartite Graphs
==============================

Definition
-------------------------
Bipartite graphs can be indicated with



Construction
-------------------------

Generate a bipartite graph structure and a feature matrix.

    .. code:: python

        >>> import torch
        >>> import dhg
        >>> dhg.random.set_seed(2022)
        >>> # Generate a random bipartite graph with 3 vertices in set U, 5 vertices in set V, and 8 edges
        >>> g = dhg.random.bigraph_Gnm(3, 5, 8)
        >>> # Generate feature matrix for vertices in set U and set V, respectively.
        >>> X_u, X_v = torch.rand(3, 2), torch.rand(5, 2)
        >>> # Print information about the bipartite graph and feature
        >>> g 
        Bipartite Graph(num_u=3, num_v=5, num_e=8)
        >>> # Print edges in the graph
        >>> g.e[0]
        [(2, 4), (0, 4), (0, 3), (2, 0), (1, 4), (2, 3), (2, 2), (1, 3)]
        >>> # Print vertex features
        >>> X_u
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594]])
        >>> X_v
        tensor([[0.7933, 0.7811],
                [0.4643, 0.6329],
                [0.6689, 0.2302],
                [0.8003, 0.7353],
                [0.7477, 0.5585]])

.. Structure Visualization
.. -------------------------------

.. Draw the bipartite graph structure

..     .. code:: python

..         >>> fig = g.draw(edge_style="line")
..         >>> fig.show()

..     Here is the image.

Spectral-Based Learning
-----------------------------

Propagate messages on the bipartite graph structure


Spectral-Based Feature Smoothing on Bipartite Graphs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python
        
        >>> # Print the Laplacian matrix defined by GCN that associated with the bipartite graph structure
        >>> g.L_GCN.to_dense()
        tensor([[0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2887, 0.2887],
                [0.0000, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.2887, 0.2887],
                [0.0000, 0.0000, 0.2000, 0.3162, 0.0000, 0.3162, 0.2236, 0.2236],
                [0.0000, 0.0000, 0.3162, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.3162, 0.0000, 0.0000, 0.5000, 0.0000, 0.0000],
                [0.2887, 0.2887, 0.2236, 0.0000, 0.0000, 0.0000, 0.2500, 0.0000],
                [0.2887, 0.2887, 0.2236, 0.0000, 0.0000, 0.0000, 0.0000, 0.2500]])
        >>> # Concate the vertex features
        >>> X = torch.cat([X_u, X_v], dim=0)
        >>> # Print the vertex features
        >>> X
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594],
                [0.7933, 0.7811],
                [0.4643, 0.6329],
                [0.6689, 0.2302],
                [0.8003, 0.7353],
                [0.7477, 0.5585]])
        >>> X_ = g.smoothing_with_GCN(X, aggr="mean")
        >>> # Print the new vertex features
        >>> X_
        tensor([[0.5788, 0.6808],
                [0.6998, 0.5005],
                [0.8138, 0.6810],
                [0.4050, 0.5042],
                [0.4643, 0.6329],
                [0.3428, 0.2288],
                [0.5392, 0.6403],
                [0.5261, 0.5961]])

Spatial-Based Learning
----------------------------

Spatial-Based Message Propagation on Bipartite Graphs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python

        >>> # Print the messages of vertices in set U and set V, respectively
        >>> X_u
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594]])
        >>> X_v
        tensor([[0.7933, 0.7811],
                [0.4643, 0.6329],
                [0.6689, 0.2302],
                [0.8003, 0.7353],
                [0.7477, 0.5585]])
        >>> X_u_ = g.v2u(X_v, aggr="mean")
        >>> X_v_ = g.u2v(X_u, aggr="mean")
        >>> # Print the new messages of vertices in set U and set V, respectively
        >>> X_u_
        tensor([[0.7740, 0.6469],
                [0.7740, 0.6469],
                [0.7526, 0.5763]])
        >>> X_v_
        tensor([[0.0262, 0.3594],
                [0.0000, 0.0000],
                [0.0262, 0.3594],
                [0.3936, 0.5542],
                [0.3936, 0.5542]])
