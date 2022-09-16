构建指标评测器
=================

.. hint:: 

    - 作者:  `丰一帆 <https://fengyifan.site/>`_
    - 翻译:  颜杰龙
    - 校对： `丰一帆 <https://fengyifan.site/>`_ 、张欣炜

DHG当前支持分类、推荐以及回归任务。各任务支持的指标如下：

- **分类** ->
  :py:meth:`Accuracy <dhg.metrics.classification.accuracy>` 
  :py:meth:`F1 Score <dhg.metrics.classification.f1_score>` 
  :py:meth:`Confusion Matrix <dhg.metrics.classification.confusion_matrix>`
  
- **推荐** ->
  :py:meth:`Precision <dhg.metrics.recommender.precision>`
  :py:meth:`Recall <dhg.metrics.recommender.recall>`
  :py:meth:`NDCG <dhg.metrics.recommender.ndcg>`

- **检索** ->
  :py:meth:`Precision <dhg.metrics.retrieval.precision>`
  :py:meth:`Recall <dhg.metrics.retrieval.recall>`
  :py:meth:`NDCG <dhg.metrics.retrieval.ndcg>`
  :py:meth:`mAP <dhg.metrics.retrieval.map>`
  :py:meth:`MRR <dhg.metrics.retrieval.mrr>`
  :py:meth:`Precision-Recall Curve <dhg.metrics.retrieval.pr_curve>`

一般来说，评测策略分为两类：

- :ref:`整轮评测 <zh_tutorial_evaluation_ee>`

  就像图神经网络上的顶点分类任务一样，评测是在每一轮结束之后对整个图进行的。


- :ref:`添加批数据后整轮评测 <zh_tutorial_evaluation_abe>`

  就像推荐系统一样，一轮包含多批训练，评测是在每一批结束之后进行的，然后整轮的结果由各批的结果聚合得到。

初始化
---------------

DHG中所有的评测器都可以使用与如下代码一样的参数构建：

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

第一个参数 ``metric_configs`` 是指标的配置，其为指标名字或指标配置的列表。
第二个参数 ``validate_index`` 是用于验证模型的指标索引，用于在验证集中计算和比较结果。

.. _zh_tutorial_evaluation_ee:

整轮评测
-----------------------------------

目前，DHG实现了两个<整轮评测>任务：图上和超图上的顶点分类。
对于验证和测试，您可以按照如下方式直接调用 :py:meth:`validate(y_true, y_pred) <dhg.metrics.BaseEvaluator.validate>` 方法和
:py:meth:`test(y_true, y_pred) <dhg.metrics.BaseEvaluator.test>` 方法：

.. note:: 

    ``evaluator.validate(y_true, y_pred)`` 只会返回第i个指标的值， 其中 ``i`` 为指定的 ``validate_index`` 。
    ``evaluator.test(y_true, y_pred)`` 会返回一个包含在 ``metric_configs`` 中所有指标的结果字典。

如下的例子展示了一个包含 ``5`` 个顶点、每个顶点属于 ``3`` 类之一的图。

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


.. _zh_tutorial_evaluation_abe:

添加批数据后整轮评测
--------------------------------------------------

目前，DHG实现了一个<添加批数据后整轮评测>任务：推荐系统。
对于验证，您可以调用 :py:meth:`validate_add_batch(y_true, y_pred) <dhg.metrics.BaseEvaluator.validate_add_batch>` 方法添加批数据，
然后调用 :py:meth:`validate_epoch_res() <dhg.metrics.BaseEvaluator.validate_epoch_res>` 方法得到验证集中的整轮结果。
对于测试，您可以调用 :py:meth:`test_add_batch(y_true, y_pred) <dhg.metrics.BaseEvaluator.test_add_batch>` 方法添加批数据，
然后调用 :py:meth:`test_epoch_res() <dhg.metrics.BaseEvaluator.test_epoch_res>` 方法得到测试集中的整轮结果。

.. note:: 

    ``evaluator.validate_epoch_res()`` 只会返回第i个指标的值， 其中 ``i`` 为指定的 ``validate_index`` 。
    ``evaluator.test_epoch_res()`` 会返回一个包含在 ``metric_configs`` 中所有指标的结果字典。

如下的例子展示了一个包含 ``4`` 个用户、 ``6`` 个物品的<用户-物品>二分图，每一轮含有 ``2`` 批。

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
