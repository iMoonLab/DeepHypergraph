特征可视化
=========================

.. hint:: 

    - 作者: 唐青梅
    - 翻译：颜杰龙
    - 校对: `丰一帆 <https://fengyifan.site/>`_

基本用法
---------------
DHG提供了一种简单的接口来可视化特征分布：

1. 输入特征及标签（可选的）；
2. 指定参数 (*例如*, `可视化的维度`, `顶点大小`, `颜色` 和 `降维方法`);
3. 调用 ``plt.show()`` 函数显示图片/动画。

   
.. note:: ``plt`` 为 ``matplotlib.pyplot`` 模块的缩写。


在欧几里得空间中进行特征可视化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../../_static/img/vis_ft_euclidean.png
    :align: center
    :alt: Visualization of Features in Euclidean Space
    :height: 400px


.. code-block:: python

    >>> import dhg
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import dhg.visualization as vis
    >>> lbl = (np.random.rand(200)*10).astype(int)
    >>> ft = dhg.random.normal_features(lbl)
    >>> vis.draw_in_euclidean_space(ft, lbl)
    >>> plt.show()


在庞加莱空间中进行特征可视化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../../_static/img/vis_ft_poincare.png
    :align: center
    :alt: Visualization of Features in Poincare Space
    :height: 400px


.. code-block:: python

    >>> import dhg
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import dhg.visualization as vis
    >>> lbl = (np.random.rand(200)*10).astype(int)
    >>> ft = dhg.random.normal_features(lbl)
    >>> vis.draw_in_poincare_space(ft, lbl)
    >>> plt.show()


制作动画
-------------------------
我们提供了制作 3D 旋转动画来可视化特征的函数。

欧几里得空间中特征的旋转可视化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../../_static/img/vis_ft_euclidean_ani.png
    :align: center
    :alt: Rotating Visualization of Features in Euclidean Space
    :height: 400px


.. code-block:: python

    >>> import dhg
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import dhg.visualization as vis
    >>> lbl = (np.random.rand(200)*10).astype(int)
    >>> ft = dhg.random.normal_features(lbl)
    >>> vis.animation_of_3d_euclidean_space(ft, lbl)
    >>> plt.show()

.. 
    >>> import numpy as np
    >>> from dhg.visualization.feature import animation_of_3d_euclidean_ball
    >>> ile_dir = "data/modelnet40/test_img_feat_4.npy"
    >>> save_dir = None  # None for show now or file name to save
    >>> label = np.load("data/modelnet40/test_label.npy")
    >>> ft = np.load(file_dir)
    >>> d = 3
    >>> low_demen_method = "tsne"  # vis for poincare_ball: pca or tsne
    >>> show_method = "Rotation"  # None for 2d or Rotation and Drag for 3d
    >>> animation_of_3d_euclidean_ball(
            ft, save_dir, d, label, reduce_method=low_demen_method, auto_play=show_method
        )

庞加莱空间中特征的旋转可视化
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: ../../_static/img/vis_ft_poincare_ani.png
    :align: center
    :alt: Rotating Visualization of Features in Poincare Space
    :height: 400px


.. code-block:: python

    >>> import dhg
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import dhg.visualization as vis
    >>> lbl = (np.random.rand(200)*10).astype(int)
    >>> ft = dhg.random.normal_features(lbl)
    >>> vis.animation_of_3d_poincare_ball(ft, lbl)
    >>> plt.show()

..
    >>> import numpy as np
    >>> from dhg.visualization.feature import animation_of_3d_poincare_ball
    >>> file_dir = "data/modelnet40/test_img_feat_4.npy" #This varies depending on the situation
    >>> save_dir = None  # None for show now or file name to save
    >>> label = np.load("data/modelnet40/test_label.npy")
    >>> ft = np.load(file_dir)
    >>> d = 3
    >>> low_demen_method = "tsne"  # vis for poincare_ball, pca or tsne
    >>> show_method = "Rotation"  # None for 2d or Rotation and Drag for 3d
    >>> animation_of_3d_poincare_ball(
            ft, save_dir, d, label, reduce_method=low_demen_method, auto_play=show_method
        )



双曲空间的数学原理
--------------------------------------------------

双曲空间是一个处处具有恒定高斯常数负曲率的流形，其有几种不同的模型并且非常适合基于梯度的优化。
我们以下工作基于庞加莱球模型展开。

具有恒定负曲率 :math:`-1 / k(k>0)` 的庞加莱球模型对应于黎曼流形 :math:`\left(\mathbb{P}^{n,k},  g_{\mathbf{x}}^{\mathbb{P}}\right)` 。
:math:`\mathbb{P}^{n,k} = \left\{\mathbf{x} \in \mathbb{R}^{n}: \| \mathbf{x}\|<1 \right\}` 是一个 :math:`n` 维单位开球，
其中 :math:`\|. \|` 代表欧几里得范数。
它的度量张量是 :math:`g_{\mathbf{x}}^{\mathbb{P}} = \lambda_{\mathbf{x}}^{2} g^{E}` ，
其中 :math:`\lambda_{\mathbf{x}} = \frac{2} {1- k\|\mathbf{x}\|^{2} }` 为保形因子、 :math:`g^{E}=\mathbf{I}_{n}` 为欧几里得度量张量。
对于两点 :math:`\mathbf{x}, \mathbf{y} \in \mathbb{P}^{n,k}` ，我们使用莫比乌斯加法 :math:`\oplus` 将陀螺空间框架与黎曼几何连接来进行加法运算：

.. math::

    \mathbf{x} \oplus_{k} \mathbf{y} =\frac{\left(1+2k\langle\mathbf{x}, \mathbf{y}\rangle+k\|\mathbf{y}\|^{2}\right) \mathbf{x}+\left(1-k\|\mathbf{x}\|^{2}\right) \mathbf{y}}{1+2k\langle\mathbf{x}, \mathbf{y}\rangle+k^{2}\|\mathbf{x}\|^{2}\|\mathbf{y}\|^{2}} .

:math:`\mathbf{x}, \mathbf{y} \in \mathbb{P}^{n,k}` 两点之间的距离是通过度量张量的集成来计算的，如下：

.. math::

    d_{\mathbb{P}}^{k} (\mathbf{x}, \mathbf{y}) = (2 / \sqrt{K}) \tanh ^{-1}\left(\sqrt{k}\left\|-x \oplus_{k} y\right\|\right) .


将点 :math:`\mathbf{z} \in \mathcal{T}_{\mathrm{x}} \mathbb{P}^{n,k}` 表示为以双曲空间中任意点 :math:`\mathbf{x}` 为中心的切线（欧几里得）空间。
对于切向量 :math:`\mathbf{z} \neq \mathbf{0}` 和点 :math:`\mathbf{y} \neq \mathbf{0}` ，
满足 :math:`\mathbf{y} \neq \mathbf{x}` 的
指数映射 :math:`\exp _{\mathbf{x}}: \mathcal{T}_{\mathbf{x}} \mathbb{P}^{n,k} \rightarrow \mathbb{P}^{n,k}`
和对数映射 :math:`\log_{\mathbf{x}}: \mathbb{P}^{n,k} \rightarrow \mathcal{T}_{\mathbf{x}} \mathbb{P}^{n,k}` 由下式给出：

.. math::

    \exp _{\mathbf{x}}^{k}(\mathbf{z})=\mathbf{x} \oplus_{k}\left(\tanh \left(\sqrt{k} \frac{\lambda_{\mathbf{x}}^{k}\|\mathbf{z}\|}{2}\right) \frac{\mathbf{z}}{\sqrt{k}\|\mathbf{z}\|}\right), 

以及

.. math::

    \log _{\mathbf{x}}^{k}(\mathbf{y})=\frac{2}{\sqrt{k} \lambda_{\mathbf{x}}^{k}} \tanh ^{-1}\left(\sqrt{k}\left\|-\mathbf{x} \oplus_{k} \mathbf{y}\right\|\right) \frac{-\mathbf{x} \oplus_{k} \mathbf{y}}{\left\|-\mathbf{x} \oplus_{k} \mathbf{y}\right\|} .

需要注意的是，我们的初始数据是在欧几里得空间上，需要转换成双曲空间上的嵌入。
所以首先把之前得到的欧几里得空间上的数据投影到双曲流形空间上，
以便使用基于谱域的超图双曲卷积网络来学习从而更新节点嵌入。
以 :math:`t:=\{\sqrt{K}, 0, 0, \dots, 0\}\in\mathbb{P}^{d, K}` 为参考点进行切线空间运算，
其中 :math:`-1/K` 为双曲线模型的负曲率。
上述前提使 :math:`\langle(0, \mathbf{x}^{0, E}), t\rangle=0` 成立，
所以 :math:`(0, \mathbf{x}^{0, E})` 可以看成是超图结构在双曲流形空间 :math:`\mathcal{T}_t\mathbb{P}^{d, K}` 切面上的初始嵌入表示。
然后使用以下等式将初始超图结构嵌入映射到双曲流形空间 :math:`\mathbb{P}` ：

.. math::

    \mathbf{x}^{0, \mathbb{P}} &=\exp _{t}^{K}\left(\left(0, \mathbf{x}^{0, \mathrm{E}}\right)\right) \\
    &=\left(\sqrt{K} \cosh \left(\frac{\left\|\mathbf{x}^{0, \mathbb{E}}\right\|_{2}}{\sqrt{K}}\right), 
    \sqrt{K} \sinh \left(\frac{\left\|\mathbf{x}^{0, \mathbb{E}}\right\|_{2}}{\sqrt{K}}\right) \frac{\mathbf{x}^{0, \mathbb{E}}}{\left\|\mathbf{x}^{0, \mathbb{E}}\right\|_{2}}\right).

双曲运算是通过欧几里得空间和双曲空间之间的特征映射来完成的。

