.. DeepHypergraph documentation master file, created by
   sphinx-quickstart on Mon Jun 20 17:17:46 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DHG Documentation
=======================

**DHG** is short for **DeepHypergraph**, which is a pytorch-based toobox for learning on both graph structure and hypergraph structure. 

**News:**

- The **v0.9.0 release** is now available!

**Citation:**

Please cite our `paper <https://ieeexplore.ieee.org/abstract/document/9795251/>`_, if you find our toolbox useful for your research. 


::

   @article{gao2022hgnn,
      title={HGNN $\^{}+ $: General Hypergraph Neural Networks},
      author={Gao, Yue and Feng, Yifan and Ji, Shuyi and Ji, Rongrong},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
      year={2022},
      publisher={IEEE}
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
   Visualization <tutorial/visualization>
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
   dhg.visualization <api/vis/index>
   dhg.random <api/random>
   dhg.utils <api/utils>


Indices and Tables
===================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
