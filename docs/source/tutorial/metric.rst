Building Evaluator
===================

.. hint:: 

    - Author: `Yifan Feng (丰一帆) <https://fengyifan.site/>`_
    - Proof: Xinwei Zhang

Currently, DHG supports classification, recommender, and retrieval tasks. The detailed metrics are:

- **Classification** ->
  :py:meth:`Accuracy <dhg.metrics.classification.accuracy>` 
  :py:meth:`F1 Score <dhg.metrics.classification.f1_score>` 
  :py:meth:`Confusion Matrix <dhg.metrics.classification.confusion_matrix>`
  
- **Recommender** -> 
  :py:meth:`Precision <dhg.metrics.recommender.precision>`
  :py:meth:`Recall <dhg.metrics.recommender.recall>`
  :py:meth:`NDCG <dhg.metrics.recommender.ndcg>`

- **Retrieval** -> 
  :py:meth:`Precision <dhg.metrics.retrieval.precision>`
  :py:meth:`Recall <dhg.metrics.retrieval.recall>`
  :py:meth:`NDCG <dhg.metrics.retrieval.ndcg>`
  :py:meth:`mAP <dhg.metrics.retrieval.map>`
  :py:meth:`MRR <dhg.metrics.retrieval.mrr>`
  :py:meth:`Precision-Recall Curve <dhg.metrics.retrieval.pr_curve>`


Generally speaking, the evaluation strategy can be divided into two categories:

- :ref:`Epoch Evaluation <tutorial_evaluation_ee>`
  
  like vertex classification in graph, the evaluation is performed on the whole graph at each epoch.


- :ref:`Add Batches Then Do Epoch Evaluation <tutorial_evaluation_abe>`
  
  like recommender systems, one epoch consists of multiple batches, and the evaluation is performed on each batch, then those batch results are aggregated to get the epoch result.

Initialization
---------------

All evaluators in DHG can be created with the same parameters as the following code:

.. code-block:: python

    >>> import dhg.metrics as dm
    >>> evaluator = dm.GraphVertexClassificationEvaluator(
            metric_configs = [
                "accuracy",
                {"f1_score": {"average": "macro"}},
            ],
            validate_index = 0
        )
    >>> evaluator = dm.UserItemRecommenderEvaluator(
            metric_configs = [
                {"precision": {"k": 20}},
                {"recall": {"k": 20}},
                {"ndcg": {"k": 20}},
            ],
            validate_index = 2
        )

The first parameter ``metric_configs`` is the metric configuration, which is a list of metric names or metric configurations. 
The second parameter ``validate_index`` is the index of the metric that is used to validate the model, which is used to compute the results in the validation set.

.. _tutorial_evaluation_ee:

Epoch Evaluation
-----------------------------------

Currently, DHG implements two <Epoch Evaluation> tasks: vertex classification on graph and hypergraph. 
As for validation and testing, you can directly call the :py:meth:`validate(y_true, y_pred) <dhg.metrics.BaseEvaluator.validate>` method and 
:py:meth:`test(y_true, y_pred) <dhg.metrics.BaseEvaluator.test>` method as follows:

.. note:: 

    The ``evaluator.validate(y_true, y_pred)`` will only return ``i``-th metric value, where ``i`` is specified by ``validate_index``. 
    The ``evaluator.test(y_true, y_pred)`` will return a result dictionary of all metrics specified in ``metric_configs``.

The following example shows a graph with ``5`` vertices and each vertex belongs to one of ``3`` classes.

.. code-block:: python

    >>> evaluator = dm.GraphVertexClassificationEvaluator(
            metric_configs = [
                "accuracy",
                {"f1_score": {"average": "micro"}},
                {"f1_score": {"average": "macro"}},
                "confusion_matrix",
            ],
            validate_index = 0
        )
    >>> y_true = torch.tensor([0, 2, 1, 0, 1])
    >>> y_pred = torch.tensor([0, 1, 0, 0, 1])
    >>> evaluator.validate(y_true, y_pred)
    0.6000000238418579
    >>> evaluator.test(y_true, y_pred)
    {
        'accuracy': 0.6000000238418579, 
        'f1_score -> average@micro': 0.6, 
        'f1_score -> average@macro': 0.43333333333333335, 
        'confusion_matrix': array([
            [2, 0, 0],
            [1, 1, 0],
            [0, 1, 0]
        ])
    }
    >>> y_pred = torch.tensor([[0.7, 0.1, 0.2],
                                [0.1, 0.8, 0.1],
                                [0.7, 0.1, 0.2],
                                [0.6, 0.2, 0.2],
                                [0.2, 0.7, 0.1],])
    >>> evaluator.validate(y_true, y_pred)
    0.6000000238418579
    >>> evaluator.test(y_true, y_pred)
    {
        'accuracy': 0.6000000238418579, 
        'f1_score -> average@micro': 0.6, 
        'f1_score -> average@macro': 0.43333333333333335, 
        'confusion_matrix': array([
            [2, 0, 0],
            [1, 1, 0],
            [0, 1, 0]
        ])
    }


.. _tutorial_evaluation_abe:

Add Batches Then Do Epoch Evaluation
--------------------------------------------------

Currently, DHG implements only one <Add Batches Then Do Epoch Evaluation> task: recommender systems. 
As for validation, you can call the :py:meth:`validate_add_batch(y_true, y_pred) <dhg.metrics.BaseEvaluator.validate_add_batch>` method to add batch data
and then call the :py:meth:`validate_epoch_res() <dhg.metrics.BaseEvaluator.validate_epoch_res>` method to get the epoch result in the validation set.
As for testing, you can call the :py:meth:`test_add_batch(y_true, y_pred) <dhg.metrics.BaseEvaluator.test_add_batch>` method to add batch data
and then call the :py:meth:`test_epoch_res() <dhg.metrics.BaseEvaluator.test_epoch_res>` method to get the epoch result in the testing set.

.. note:: 

    The ``evaluator.validate_epoch_res()`` will only return ``i``-th metric value, where ``i`` is specified by ``validate_index``. 
    The ``evaluator.test_epoch_res()`` will return a result dictionary of all metrics specified in ``metric_configs``.

The following example shows a User-Item bipartite graph with ``4`` users and ``6`` items, and the batch size is ``2``.

.. code-block:: python

    >>> evaluator = dm.UserItemRecommenderEvaluator(
            metric_configs = [
                {"precision": {"k": 20}},
                {"recall": {"k": 20}},
                {"ndcg": {"k": 20}},
            ],
            validate_index = 2
        )
    >>> batch_y_true = torch.tensor([[0, 1, 0, 1, 0, 0],
                                    [0, 0, 1, 1, 0, 0]])
    >>> batch_y_pred = torch.tensor([[0.7, 0.9, 0.1, 0.1, 0.2, 0.0],
                                     [0.1, 0.2, 0.5, 0.3, 0.6, 0.0]])
    >>> evaluator.validate_add_batch(batch_y_true, batch_y_pred)
    >>> batch_y_true = torch.tensor([[0, 1, 0, 1, 1, 0],
                                    [0, 0, 1, 0, 1, 1]])
    >>> batch_y_pred = torch.tensor([[0.3, 0.2, 0.1, 0.5, 0.2, 0.3],
                                     [0.3, 0.5, 0.7, 0.2, 0.1, 0.5]])
    >>> evaluator.validate_add_batch(batch_y_true, batch_y_pred)
    >>> evaluator.validate_epoch_res()
    0.816944420337677
    >>> batch_y_true = torch.tensor([[0, 1, 0, 1, 0, 0],
                                    [0, 0, 1, 1, 0, 0]])
    >>> batch_y_pred = torch.tensor([[0.7, 0.9, 0.1, 0.1, 0.2, 0.0],
                                     [0.1, 0.2, 0.5, 0.3, 0.6, 0.0]])
    >>> evaluator.test_add_batch(batch_y_true, batch_y_pred)
    >>> batch_y_true = torch.tensor([[0, 1, 0, 1, 1, 0],
                                    [0, 0, 1, 0, 1, 1]])
    >>> batch_y_pred = torch.tensor([[0.3, 0.2, 0.1, 0.5, 0.2, 0.3],
                                     [0.3, 0.5, 0.7, 0.2, 0.1, 0.5]])
    >>> evaluator.test_add_batch(batch_y_true, batch_y_pred)
    >>> evaluator.test_epoch_res()
    {
        'precision -> k@20': 0.4166666716337204, 
        'recall -> k@20': 1.0, 
        'ndcg -> k@20': 0.816944420337677
    }

