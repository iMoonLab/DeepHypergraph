dhg.metrics
=============

Basic Metrics
----------------

Classification 
+++++++++++++++++

.. autofunction:: dhg.metrics.available_classification_metrics

.. autofunction:: dhg.metrics.classification.accuracy

.. autofunction:: dhg.metrics.classification.f1_score

.. autofunction:: dhg.metrics.classification.confusion_matrix


Recommender
+++++++++++++++++

.. autofunction:: dhg.metrics.available_recommender_metrics


.. autofunction:: dhg.metrics.recommender.precision
    
.. autofunction:: dhg.metrics.recommender.recall

.. autofunction:: dhg.metrics.recommender.ndcg


Retrieval
+++++++++++++++++

.. autofunction:: dhg.metrics.available_retrieval_metrics


.. autofunction:: dhg.metrics.retrieval.precision
    
.. autofunction:: dhg.metrics.retrieval.recall
    
.. autofunction:: dhg.metrics.retrieval.ap

.. autofunction:: dhg.metrics.retrieval.map

.. autofunction:: dhg.metrics.retrieval.ndcg

.. autofunction:: dhg.metrics.retrieval.rr 

.. autofunction:: dhg.metrics.retrieval.mrr 

.. autofunction:: dhg.metrics.retrieval.pr_curve


Evaluators for Different Tasks
--------------------------------------------------------

.. autofunction:: dhg.metrics.build_evaluator


Base Class
++++++++++++++++++++++++++++++++++++
.. autoclass:: dhg.metrics.BaseEvaluator
    :members:

Vertex Classification Task
++++++++++++++++++++++++++++++++++++

On Graph
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: dhg.metrics.GraphVertexClassificationEvaluator
    :members:
    :show-inheritance:


On Hypergraph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dhg.metrics.HypergraphVertexClassificationEvaluator
    :members:
    :show-inheritance:

Recommender Task
++++++++++++++++++

On User-Item Bipartite Graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dhg.metrics.UserItemRecommenderEvaluator
    :members:
    :show-inheritance:

Retrieval Task
+++++++++++++++++++

.. autoclass:: dhg.metrics.RetrievalEvaluator
    :members:
    :show-inheritance:
