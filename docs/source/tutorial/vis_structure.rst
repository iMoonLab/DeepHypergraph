Structure Visualization
=============================


Basic Usage
--------------
DHG provides a simple interface to visualize the correlation structures:

1. Create a Structure object (*i.e.*, :py:class:`dhg.Graph`, :py:class:`dhg.BiGraph`, :py:class:`dhg.DiGraph`, and :py:class:`dhg.Hypergraph`);
2. Call the ``draw()`` method of the object;
3. Call ``plt.show()`` to show the figure or ``plt.savefig()`` to save the figure. 
   
.. note:: The ``plt`` is short of ``matplotlib.pyplot`` module.


Visualization of Undirected Graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../_static/img/vis_graph.png
    :align: center
    :alt: Visualization of Undirected Graph
    :height: 400px


.. code-block:: python

    import matplotlib.pyplot as plt
    from dhg.random import graph_Gnm

    g = graph_Gnm(10, 12)
    g.draw()
    plt.show()


Visualization of Directed Graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../_static/img/vis_digraph.png
    :align: center
    :alt: Visualization of Directed Graph 
    :height: 400px

.. code-block:: python

    import matplotlib.pyplot as plt
    from dhg.random import digraph_Gnm

    g = digraph_Gnm(12, 18)
    g.draw()
    plt.show()


Visualization of Bipartite Graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. image:: ../_static/img/vis_bigraph.png
    :align: center
    :alt: Visualization of Bipartite Graph
    :height: 400px

.. code-block:: python

    import matplotlib.pyplot as plt
    from dhg.random import bigraph_Gnm

    g = bigraph_Gnm(30, 40, 20)
    g.draw()
    plt.show()


Visualization of Undirected Hypergraph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../_static/img/vis_hypergraph.png
    :align: center
    :alt: Visualization of Undirected Hypergraph
    :height: 400px

.. code-block:: python

    import matplotlib.pyplot as plt
    from dhg.random import hypergraph_Gnm

    h = hypergraph_Gnm(10, 8, method='low_order_first')
    h.draw()
    plt.show()



Advanced Usage
---------------------

Coming soon...

.. different style, change size, change color, change opacity


.. Mathamatical Principles
.. -----------------------

.. Simple Graph
.. ~~~~~~~~~~~~~~

.. Directed Graph
.. ~~~~~~~~~~~~~~~

.. Bipartite Graph
.. ~~~~~~~~~~~~~~~~

.. Simple Hypergraph
.. ~~~~~~~~~~~~~~~~~~
