.. DeepHypergraph documentation master file, created by
   sphinx-quickstart on Mon Jun 20 17:17:46 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DHG Documentation
=======================

**DHG** (*DeepHypergraph*) is a deep learning library built upon `PyTorch <https://pytorch.org/>`_ for learning with both Graph Neural Networks and Hypergraph Neural Networks. 
It is a general framework that supports both low-order and high-order message passing like 
**from vertex to vertex**, **from vertex in one domain to vertex in another domain**, **from vertex to hyperedge**, **from hyperedge to vertex**, **from vertex set to vertex set**.

It supports a wide variety of structures like low-order structures (simple graphs, directed graphs, bipartite graphs, etc.), 
high-order structures (simple hypergraphs, etc.). Various spectral-based operations (like Laplacian-based smoothing) 
and spatial-based operations (like message psssing from domain to domain) are integrated inside different structures. 
It provides multiple common metrics for performance evaluation on different tasks. Many state-of-the-art models are 
implemented and can be easily used for research. We also provide various visualization tools for both low-order 
structures and high-order structures. 

In addition, DHG's ``experiments`` module (that implements **Auto-ML** upon `Optuna <https://optuna.org/>`_) 
can help you automatically tune the hyper-parameters of your models in training and easily outperforms the state-of-the-art models.

**News**

- The **v0.9.1 release** is now available!

**Citing**

If you find our library useful for your research, please cite our papers:


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

.. toctree::
   :maxdepth: 2
   :caption: Get Started

   start/install
   start/structure
   start/low_order
   start/high_order
   start/exp_cls
   start/exp_recommender
   start/exp_autoML


.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   Overview <tutorial/overview>
   Build Structure <tutorial/structure>
   Build Dataset <tutorial/dataset>
   Build Model <tutorial/model>
   Build Evaluator <tutorial/metric>
   Model Training <tutorial/train>
   Auto-ML <tutorial/auto_ml>
   Random Structure Generation <tutorial/random>
   tutorial/structure/index

.. toctree:: 
   :maxdepth: 2
   :caption: 中文文档

   zh/start/index
   zh/tutorial/index


.. toctree::
   :maxdepth: 2
   :caption: API Reference

   dhg <api/dhg>
   dhg.nn <api/nn>
   dhg.models <api/models>
   dhg.data <api/data>
   dhg.datapipe <api/datapipe>
   dhg.metrics <api/metrics>
   dhg.experiments <api/experiments>
   dhg.random <api/random>
   dhg.utils <api/utils>


Indices and Tables
===================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
