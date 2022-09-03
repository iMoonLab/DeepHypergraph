如何加入DHG贡献团队
======================

.. hint:: 

    - 作者:  `丰一帆 <https://fengyifan.site/>`_
    - 翻译:  颜杰龙
    - 校对： `丰一帆 <https://fengyifan.site/>`_ 、张欣炜

DHG是免费软件，您可以基于Apache License 2.0协议自由分发或修改。我们期待您的贡献。

您可以选择以下方式贡献DHG

1. 修复bugs。
2. 实现新功能和增强功能。
3. 实现或提升DHG的低阶或高阶关联结构。
4. 实现指定关联结构内基于谱域的拉普拉斯矩阵。
5. 实现基于空域的信息传递或聚合操作。
6. 实现低阶或高阶关联结构内的卷积层。
7. 实现最先进模型。
8. 实现全新损失函数。
9. 实现指定任务内的全新评测指标。
10. 标注或上传新的数据集。
11. 提升文档质量。
12. 提升Auto-ML模块。

一旦您选择了，我们建议您首先在Github上提出一个问题或讨论。
开发之前，请先阅读以下关于编程风格和测试的章节。
审阅者会审查代码并提供必要的更改建议，一旦审阅者批准更改，PR就可以合并了。

编程风格
----------------
对于python代码，我们通常遵循 `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ 风格指导。
文档遵循 `Google <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google>`_ 风格。

DHG使用 `black <https://black.readthedocs.io/en/stable/>`_ 包格式代码。
格式代码的配置在 ``pyproject.toml`` 文件中。

代码测试
-------------
DHG的测试位于 ``tests/`` 文件夹下，您可以在 ``tests/`` 内的子文件夹中为您实现的功能补充新的测试文件。
使用如下命令运行所有测试

.. code-block:: bash

    pytest .


使用如下命令运行单个文件的测试

.. code-block:: bash

    pytest tests/xxx/xxx.py

``tests/xxx/xxx.py`` 为示例文件名。


构建文档
------------------------------
1. 克隆DHG仓库。

    .. code-block:: bash

        git clone https://github.com/iMoonLab/DeepHypergraph

2. 在  ``docs/`` 文件夹内根据 ``requirements.txt`` 安装依赖。

    .. code-block:: bash
    
        pip install -r docs/requirements.txt

3. 使用以下命令构建文档

    .. code-block:: bash
    
        cd docs
        make clean && make html

4. 使用浏览器打开html文件(``docs/build/html/index.html``)。


