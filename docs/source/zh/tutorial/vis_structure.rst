关联结构可视化
=============================

.. hint::

    - 作者: 张欣炜
    - 翻译：颜杰龙
    - 校对: `丰一帆 <https://fengyifan.site/>`_

基本用法
--------------
DHG提供了一种简单的接口来可视化关联结构：

1. 构造关联结构对象 (*也就是*, :py:class:`dhg.Graph`、 :py:class:`dhg.BiGraph`、 :py:class:`dhg.DiGraph` 和 :py:class:`dhg.Hypergraph`);
2. 调用对象的 ``draw()`` 方法;
3. 调用 ``plt.show()`` 显示图片或者 ``plt.savefig()`` 保存图片。

.. note:: ``plt`` 为 ``matplotlib.pyplot`` 模块的缩写。


图的可视化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../../_static/img/vis_graph.png
    :align: center
    :alt: Visualization of Graph
    :height: 400px


.. code-block:: python

    >>> import matplotlib.pyplot as plt
    >>> from dhg.random import graph_Gnm
    >>> g = graph_Gnm(10, 12)
    >>> g.draw()
    >>> plt.show()


有向图的可视化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../../_static/img/vis_digraph.png
    :align: center
    :alt: Visualization of Directed Graph
    :height: 400px

.. code-block:: python

    >>> import matplotlib.pyplot as plt
    >>> from dhg.random import digraph_Gnm
    >>> g = digraph_Gnm(12, 18)
    >>> g.draw()
    >>> plt.show()


二分图的可视化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. image:: ../../_static/img/vis_bigraph.png
    :align: center
    :alt: Visualization of Bipartite Graph
    :height: 400px

.. code-block:: python

    >>> import matplotlib.pyplot as plt
    >>> from dhg.random import bigraph_Gnm
    >>> g = bigraph_Gnm(30, 40, 20)
    >>> g.draw()
    >>> plt.show()


超图的可视化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../../_static/img/vis_hypergraph.png
    :align: center
    :alt: Visualization of Hypergraph
    :height: 400px

.. code-block:: python

    >>> import matplotlib.pyplot as plt
    >>> from dhg.random import hypergraph_Gnm
    >>> hg = hypergraph_Gnm(10, 8, method='low_order_first')
    >>> hg.draw()
    >>> plt.show()



高级用法
---------------------

自定义标签
^^^^^^^^^^^^^^^^^^^^^^^^^
顶点的标签可以通过 ``v_label`` 参数自定义。
``v_label`` 可以为字符串列表。
顶点的标签为列表中的字符串。
例如，以下代码显示如何自定义图关联结构中顶点的标签。
如果没有指定 ``v_label`` ， 那么图中不会显示任何标签。
``dhg.Graph``、 ``dhg.DiGraph`` 和 ``dhg.Hypergraph`` 中的 ``font_size`` 参数以及 ``dhg.BiGraph`` 中的 ``u_font_size`` 和 ``v_font_size`` 参数用于指定标签字体的相对大小，
其默认值为 ``1.0`` 。
``font_family`` 参数用于指定标签的字体，其默认值为 ``'sans-serif'`` 。

.. code-block:: python

    >>> import matplotlib.pyplot as plt
    >>> from dhg.random import graph_Gnm
    >>> g = graph_Gnm(10, 12)
    >>> labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    >>> g.draw(v_label=labels, font_size=1.5, font_family='serif')
    >>> plt.show()

.. image:: ../../_static/img/custom_label.png
    :align: center
    :alt: Customize label
    :height: 400px


自定义颜色
^^^^^^^^^^^^^^^^^^^^^^^^^
对于 ``dhg.Graph`` 、 ``dhg.DiGraph`` 和 ``dhg.Hypergraph`` ，
顶点的颜色可以由 ``v_color`` 参数指定，边的颜色可以由 ``e_color`` 指定。
对于 ``dhg.BiGraph`` ， 集合 :math:`\mathcal{U}` 内顶点的颜色可以由 ``u_color`` 参数指定，
集合 :math:`\mathcal{V}` 内顶点的颜色可以由 ``v_color`` 参数指定。
``v_color`` 、 ``u_color`` 和 ``e_color`` 参数为单个字符串或者字符串列表。
若为单个字符串，那么所有的顶点或边将根据该字符串着色。
若为字符串列表，顶点或者边的颜色为该列表中的字符串。
例如，以下代码显示如何自定义超图的顶点和边的颜色。

.. code-block:: python

    >>> import matplotlib.pyplot as plt
    >>> from dhg.random import hypergraph_Gnm
    >>> hg = hypergraph_Gnm(10, 8, method='low_order_first')
    >>> hg.draw(v_color='cyan', e_color='grey')
    >>> plt.show()

.. image:: ../../_static/img/custom_color.png
    :align: center
    :alt: Customize color
    :height: 400px


自定义大小
^^^^^^^^^^^^^^^^^^^^^^^^^
对于 ``dhg.Graph`` 、 ``dhg.DiGraph`` 和 ``dhg.Hypergraph`` ，
顶点的大小可以由 ``v_size`` 参数指定，边的大小可以由 ``e_size`` 参数指定。
对于 ``dhg.BiGraph`` ， 集合 :math:`\mathcal{U}` 内顶点的大小可以由 ``u_size`` 参数指定，
集合 :math:`\mathcal{V}` 内顶点的大小可以由 ``v_size`` 参数指定。
``v_size`` 、 ``u_size`` 和 ``e_size`` 参数为单个浮点数或者浮点数列表。
若为单个浮点数，那么所有的顶点或边将根据该浮点数调整大小。
若为浮点数列表，顶点或者边的大小为该列表中的浮点数。
``v_line_width`` 表示顶点周围线的宽度。
``e_line_width`` 表示边周围线的宽度。
以上所有的大小为相对大小，默认值为 ``1.0`` 。
例如，以下代码显示如何自定义超图的顶点和边的大小。

.. code-block:: python

    >>> import matplotlib.pyplot as plt
    >>> from dhg.random import graph_Gnm
    >>> g = graph_Gnm(10, 12)
    >>> g.draw(v_size=1.5, v_line_width=1.5, e_line_width=1.5)
    >>> plt.show()

.. image:: ../../_static/img/custom_size.png
    :align: center
    :alt: Customize size
    :height: 400px



自定义布局
^^^^^^^^^^^^^^^^^^^^^^^^^
关联结构的布局基于改进后的定向力布局算法。
顶点的位置由四种力确定，也就是，顶点吸引力 :math:`f_{na}` 、顶点斥力 :math:`f_{nr}`、 边斥力 :math:`f_{er}` 和 中心力 :math:`f_c`。
:math:`f_{na}` 是吸引相邻顶点的弹力。
:math:`f_{nr}` 用于将顶点相互排斥。
:math:`f_{er}` 用于将超边相互排斥，其只用于超图可视化。
:math:`f_c` 用于将顶点吸引到中心（二分图的两个中心）。
各种力的强度可以通过 ``forces`` 参数自定义，该参数是包含
``Simulator.NODE_ATTRACTION`` 、 ``Simulator.NODE_REPULSION`` 、 ``Simulator.EDGE_REPULSION`` 和 ``Simulator.CENTER_GRAVITY`` 键值的字典。
力的默认值为 ``1.0`` 。


.. different style, change size, change color, change opacity


.. Mathamatical Principles
.. -----------------------

.. Graph
.. ~~~~~~~~~~~~~~

.. Directed Graph
.. ~~~~~~~~~~~~~~~

.. Bipartite Graph
.. ~~~~~~~~~~~~~~~~

.. Hypergraph
.. ~~~~~~~~~~~~~~~~~~
