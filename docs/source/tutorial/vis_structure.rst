Structure Visualization
=============================

Coming soon!

Basic Usage
--------------
DHG provides a simple interface to visualize the structures:

1. Create a Structure object (*i.e.*, ``Graph``, ``BiGraph``, ``DiGraph``, and ``Hypergraph``);

2. Call the ``draw`` method of the object;

3. Call ``plt.show()`` to show the figure or ``plt.savefig()`` to save the figure.

Visualization for ``Graph``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import matplotlib.pyplot as plt
    from dhg.random import graph_Gnm

    g = graph_Gnm(100, 50)
    g.draw()
    plt.show()


.. image:: ../_static/img/vis_graph.png
    :align: center
    :alt: vis_graph
    :height: 200px


Visualization for ``DiGraph``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import matplotlib.pyplot as plt
    from dhg.random import digraph_Gnm

    g = digraph_Gnm(100, 50)
    g.draw()
    plt.show()


.. image:: ../_static/img/vis_digraph.png
    :align: center
    :alt: vis_digraph
    :height: 200px


Visualization for ``BiGraph``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: python

    import matplotlib.pyplot as plt
    from dhg.random import bigraph_Gnm

    g = bigraph_Gnm(100, 80, 20)
    g.draw()
    plt.show()


.. image:: ../_static/img/vis_bigraph.png
    :align: center
    :alt: vis_bigraph
    :height: 200px



Visualization for ``Hypergraph``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: python

    import matplotlib.pyplot as plt
    from dhg.random import hypergraph_Gnm

    h = hypergraph_Gnm(100, 10, method='low_order_first')
    h.draw()
    plt.show()


.. image:: ../_static/img/vis_hypergraph.png
    :align: center
    :alt: vis_hypergraph
    :height: 200px




.. Advanced Usage
.. ---------------------

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
