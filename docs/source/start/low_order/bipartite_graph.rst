
.. _start_learning_on_bipartite_graph:

Learning on Bipartite Graph
==============================

.. hint:: 

    - Author: `Yifan Feng (丰一帆) <https://fengyifan.site/>`_
    - Proof: Xinwei Zhang

Definition
-------------------------
A `bipartite graph <https://en.wikipedia.org/wiki/Bipartite_graph>`_ can be indicated with :math:`\mathcal{G} = (\mathcal{U}, \mathcal{V}, \mathcal{E})` 
whose partition has the parts :math:`\mathcal{U}` and :math:`\mathcal{V}`, with :math:`\mathcal{E}` denoting the edges of the bipartite graph. 

- :math:`\mathcal{U}`, is one set of **vertices** (also called **nodes** or **points**);
- :math:`\mathcal{V}`, is another set of **vertices** (also called **nodes** or **points**);
- :math:`\mathcal{E} \subseteq \{ (x, y) \mid x \in \mathcal{U}~and~y \in \mathcal{V} \}`, a set of **edges** (also called **links** or **lines**).

When modelling relations between two different classes of objects, bipartite graph very often arise naturally. 
For instance, a bipartite graph of football players and clubs, with an edge between a player and a club if the player has played for that club, 
is a natural example of an affiliation network, a type of bipartite graph used in social network analysis. 
Another example is a bipartite graph of user-item interactions, where the user watches/clicks/views/likes/budges an item can model 
different scenarios like: movie/news/goods/hotels/products/services recommender. 


Construction
-------------------------
The bipartite graph structure can be constructed by the following methods. More details can refer to :ref:`here <build_bipartite_graph>`.

- Edge list (**default**) :py:class:`dhg.BiGraph`
- Adjacency list :py:meth:`dhg.BiGraph.from_adj_list`
- Hypergraph :py:meth:`dhg.BiGraph.from_hypergraph`

In the following example, we randomly generate a bipartite graph structure and two feature matrices to perform some basic learning operations on this structure.

    .. code:: python

        >>> import torch
        >>> import dhg
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

Smoothing with GCN's Laplacian
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
        >>> X_ = g.smoothing_with_GCN(X)
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
        >>> # Print the new vertex feautres in set U and set V, respectively
        >>> X_u_, X_v_ = torch.split(X_, [g.num_u, g.num_v], dim=0)
        >>> X_u_
        tensor([[0.5788, 0.6808],
                [0.6998, 0.5005],
                [0.8138, 0.6810]])
        >>> X_v_
        tensor([[0.4050, 0.5042],
                [0.4643, 0.6329],
                [0.3428, 0.2288],
                [0.5392, 0.6403],
                [0.5261, 0.5961]])


Spatial-Based Learning
----------------------------

Message Propagation from Vertices in Set :math:`U` to Vertices in Set :math:`V`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python

        >>> # Print the messages of vertices in set U
        >>> X_u
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594]])
        >>> X_v_ = g.u2v(X_u, aggr="mean")
        >>> # Print the new messages of vertices in set V
        >>> X_v_
        tensor([[0.0262, 0.3594],
                [0.0000, 0.0000],
                [0.0262, 0.3594],
                [0.3936, 0.5542],
                [0.3936, 0.5542]])

Message Propagation from Vertices in Set :math:`U` to Vertices in Set :math:`V` with different Edge Weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python

        >>> # Print the messages of vertices in set U
        >>> X_u
        tensor([[0.3958, 0.9219],
                [0.7588, 0.3811],
                [0.0262, 0.3594]])
        >>> g.e_weight
        tensor([1., 1., 1., 1., 1., 1., 1., 1.])
        >>> # Generate random edge weights
        >>> e_weight = torch.rand(len(g.e_weight))
        >>> e_weight
        tensor([0.6226, 0.8429, 0.6105, 0.1248, 0.8265, 0.2117, 0.8574, 0.4282])
        >>> X_v_ = g.u2v(X_u, e_weight=e_weight, aggr="mean")
        >>> # Print the new messages of vertices in set V
        >>> X_v_
        tensor([[1.7913e-02, 2.4547e-01],
                [0.0000e+00, 0.0000e+00],
                [1.1753e-03, 1.6106e-02],
                [1.5306e+00, 2.3305e+00],
                [6.1360e-01, 1.3660e+00]])

Message Propagation from Vertices in Set :math:`V` to Vertices in Set :math:`U`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python

        >>> # Print the messages of vertices in set V
        >>> X_v
        tensor([[0.7933, 0.7811],
                [0.4643, 0.6329],
                [0.6689, 0.2302],
                [0.8003, 0.7353],
                [0.7477, 0.5585]])
        >>> X_u_ = g.v2u(X_v, aggr="mean")
        >>> # Print the new messages of vertices in set U
        >>> X_u_
        tensor([[0.7740, 0.6469],
                [0.7740, 0.6469],
                [0.7526, 0.5763]])

Message Propagation from Vertices in Set :math:`V` to Vertices in Set :math:`U` with different Edge Weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. code:: python

        >>> # Print the messages of vertices in set V
        >>> X_v
        tensor([[0.7933, 0.7811],
                [0.4643, 0.6329],
                [0.6689, 0.2302],
                [0.8003, 0.7353],
                [0.7477, 0.5585]])
        >>> g.e_weight
        tensor([1., 1., 1., 1., 1., 1., 1., 1.])
        >>> # Generate random edge weights
        >>> e_weight = torch.rand(len(g.e_weight))
        >>> e_weight
        tensor([0.6226, 0.8429, 0.6105, 0.1248, 0.8265, 0.2117, 0.8574, 0.4282])
        >>> X_u_ = g.v2u(X_v, e_weight=e_weight, aggr="mean")
        >>> # Print the new messages of vertices in set U
        >>> X_u_
        tensor([[1.6537, 1.3607],
                [0.4279, 0.3814],
                [4.1914, 3.6342]])

