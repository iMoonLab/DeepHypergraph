安装
===========

.. hint:: 

    - 作者:  `丰一帆 <https://fengyifan.site/>`_
    - 翻译:  颜杰龙
    - 校对： `丰一帆 <https://fengyifan.site/>`_ 、张欣炜

- Python >= 3.8
- Pytorch >= 1.12


目前， **DHG** 的最新稳定版本 **0.9.4** 已经发布，可以使用 ``pip`` 指令直接安装：

.. code-block:: bash

    pip install dhg

如果您想尝试最新的日构建版本(nightly version) **0.9.5** ，可以使用以下指令安装：

.. code-block:: bash

    pip install git+https://github.com/iMoonLab/DeepHypergraph.git

.. note:: 
    
    Nightly version 通常会比稳定版本更新，因为它包含了最新的功能和SOTA方法、数据集。但是，nightly version 也可能会有一些bug，因此不建议在生产环境中使用。
    如果您发现了bug，请在 `GitHub <https://github.com/iMoonLab/DeepHypergraph/issues>`_ 上提交issue。
    