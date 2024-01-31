DHG简介
==============

**DHG** (*DeepHypergraph*) 是基于 `PyTorch <https://pytorch.org/>`_ 的深度学习包，
可用于图神经网络以及超图神经网络。
其支持 **从顶点到顶点**、**从一个域的顶点到另一个域的顶点**、 **从顶点到超边**,、**从超边到顶点**、**从顶点集到顶点集** 等低阶或高阶信息传递的通用框架。

其支持大量低阶关联结构（图、有向图、二分图等）以及高阶关联结构（超图等）。
大量基于谱域的操作（例如基于拉普拉斯的平滑）和基于空域的操作（例如从域到域的信息传递）集成在不同的关联结构中。
其为不同任务的性能评测提供多种通用评测指标，并且覆盖多种当前最先进的模型以便简单使用。
我们同样为低阶或高阶关联结构提供多种可视化工具。

除此之外，DHG的 :doc:`/api/experiments` 模块基于 `Optuna <https://optuna.org/>`_ 实现了 **Auto-ML**，
可以帮您调整模型训练超参数以便轻松超过最先进的的模型。

**新闻**

- 2024-01-31  ->   **v0.9.4** 正式发布！ 修复了若干bug，包含更多数据集！
- 2022-12-28  ->   **v0.9.3** 正式发布！ 包含更多数据集和超图操作！
- 2022-09-25  ->   **v0.9.2** 正式发布！ 包含更多数据集、最新模型和可视化功能！
- 2022-08-25  ->   DHG的第一个版本 **v0.9.1** 正式发布！


**引用**

如果您觉得我们的包对您的研究有用，请引用我们的论文：


.. code-block:: text

   @article{gao2022hgnn,
      title={HGNN $\^{}+ $: General Hypergraph Neural Networks},
      author={Gao, Yue and Feng, Yifan and Ji, Shuyi and Ji, Rongrong},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
      year={2022},
      publisher={IEEE}
   }


.. code-block:: text

   @inproceedings{feng2019hypergraph,
      title={Hypergraph neural networks},
      author={Feng, Yifan and You, Haoxuan and Zhang, Zizhao and Ji, Rongrong and Gao, Yue},
      booktitle={Proceedings of the AAAI conference on artificial intelligence},
      volume={33},
      number={01},
      pages={3558--3565},
      year={2019}
   }