.. DeepHypergraph documentation master file, created by
   sphinx-quickstart on Mon Jun 20 17:17:46 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=======================
DHG Overview
=======================

.. raw:: latex

   \part{English}
   \chapter{DHG Overview}


**DHG** (*DeepHypergraph*) is a deep learning library built upon `PyTorch <https://pytorch.org/>`_ for learning with both Graph Neural Networks and Hypergraph Neural Networks. 
It is a general framework that supports both low-order and high-order message passing like 
**from vertex to vertex**, **from vertex in one domain to vertex in another domain**, **from vertex to hyperedge**, **from hyperedge to vertex**, **from vertex set to vertex set**.

It supports a wide variety of structures like low-order structures (graph, directed graph, bipartite graph, etc.), 
high-order structures (hypergraph, etc.). Various spectral-based operations (like Laplacian-based smoothing) 
and spatial-based operations (like message psssing from domain to domain) are integrated inside different structures. 
It provides multiple common metrics for performance evaluation on different tasks. Many state-of-the-art models are 
implemented and can be easily used for research. We also provide various visualization tools for both low-order 
structures and high-order structures. 

In addition, DHG's :doc:`/api/experiments` module (that implements **Auto-ML** upon `Optuna <https://optuna.org/>`_) 
can help you automatically tune the hyper-parameters of your models in training and easily outperforms the state-of-the-art models.

**News**

- 2024-01-31 -> The **v0.9.4** is now available! Fix some bugs and more datasets are included!
- 2022-12-28 -> The **v0.9.3** is now available! More datasets and operations of hypergraph are included!
- 2022-09-25 -> The **v0.9.2** is now available! More datasets, SOTA models, and visualizations are included!
- 2022-08-25 -> The **v0.9.1** is now available!

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

.. ======================================================

.. decomment for html

.. toctree::
   :maxdepth: 2
   :caption: Get Started
   :hidden:

   start/install
   start/structure
   start/low_order/index
   start/high_order/index
   start/contribution


.. toctree:: 
   :maxdepth: 2
   :caption: Examples
   :hidden:
   
   examples/vertex_cls/index
   examples/recommender
   examples/auto_ml/index

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   tutorial/overview
   tutorial/structure
   tutorial/dataset
   tutorial/model
   tutorial/metric
   tutorial/train
   tutorial/auto_ml
   tutorial/random
   tutorial/vis_structure
   tutorial/vis_feature

.. decomment for latex

   .. toctree:: 
      :maxdepth: 3
      :caption: English
      :hidden:

      start/index
      examples/index
      tutorial/index

.. ======================================================

.. raw:: latex

   \part{中文}


.. toctree:: 
   :maxdepth: 3
   :caption: 中文文档
   :hidden:

   zh/overview
   zh/start/index
   zh/examples/index
   zh/tutorial/index


.. raw:: latex

   \part{API Reference}


.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   dhg <api/dhg>
   dhg.nn <api/nn>
   dhg.models <api/models>
   dhg.data <api/data>
   dhg.datapipe <api/datapipe>
   dhg.metrics <api/metrics>
   dhg.experiments <api/experiments>
   dhg.visualization <api/visualization>
   dhg.random <api/random>
   dhg.utils <api/utils>


.. Indices 
.. ===================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
