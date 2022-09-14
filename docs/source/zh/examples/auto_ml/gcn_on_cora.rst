在Cora上使用GCN
=================

.. hint:: 

    - 作者:  `丰一帆 <https://fengyifan.site/>`_
    - 翻译:  颜杰龙
    - 校对： `丰一帆 <https://fengyifan.site/>`_ 、张欣炜

在如下的例子中，我们介绍在图节点分类任务中使用Auto-ML。
关于如何使用 :doc:`/api/experiments` 进行自动模型调优的细节可以参考自 <:doc:`/tutorial/auto_ml`>。

配置
--------------

- 模型: GCN (:py:class:`dhg.models.GCN`): `Semi-Supervised Classification with Graph Convolutional Networks <https://arxiv.org/pdf/1609.02907>`_ 论文 (ICLR 2017)。
- 数据集: Cora (:py:class:`dhg.data.Cora`): 节点分类任务使用的引用网络数据集。

导入依赖包
---------------------

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.optim as optim

    from dhg import Graph
    from dhg.data import Cora
    from dhg.models import GCN
    from dhg.random import set_seed
    from dhg.experiments import GraphVertexClassificationTask as Task
    from dhg.metrics import GraphVertexClassificationEvaluator as Evaluator



定义函数
-------------------

.. code-block:: python


    def model_builder(trial):
        return GCN(ft_dim, trial.suggest_int("hidden_dim", 8, 32), num_classes)


    def train_builder(trial, model):
        optimizer = optim.Adam(model.parameters(), lr=trial.suggest_loguniform("lr", 1e-4, 1e-2), weight_decay=5e-4,)
        criterion = nn.CrossEntropyLoss()
        return {
            "optimizer": optimizer,
            "criterion": criterion,
        }

主函数
--------

.. important:: 

    您需要修改 ``work_root`` 变量为您的工作目录。


.. code-block:: python

    if __name__ == "__main__":
        work_root = "/home/fengyifan/OS3D/toolbox/exp_cache/tmp"
        set_seed(2022)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        data = Cora()
        num_v, ft_dim = data["features"].shape
        num_classes = data["labels"].max().item() + 1
        input_data = {
            "features": data["features"],
            "structure": Graph(num_v, data["edge_list"]),
            "labels": data["labels"],
            "train_mask": data["train_mask"],
            "val_mask": data["val_mask"],
            "test_mask": data["test_mask"],
        }
        evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
        task = Task(work_root, input_data, model_builder, train_builder, evaluator, device,)
        task.run(200, 50, "maximize")

.. only:: not latex

    输出
    -------------

    .. code-block:: 

        [I 2022-08-25 17:40:25,670] Logs will be saved to /home/fengyifan/OS3D/toolbox/exp_cache/tmp/2022-08-25--17-40-25/log.txt
        [I 2022-08-25 17:40:25,670] Files in training will be saved in /home/fengyifan/OS3D/toolbox/exp_cache/tmp/2022-08-25--17-40-25
        [I 2022-08-25 17:40:27,112] Random seed is 2022
        [I 2022-08-25 17:40:27,112] A new study created in memory with name: no-name-0c8f8a97-d029-4bc6-81dd-b2dbaeae38ef
        [I 2022-08-25 17:40:28,107] Trial 0 finished with value: 0.2639999985694885 and parameters: {'hidden_dim': 8, 'lr': 0.0009956704582324435}. Best is trial 0 with value: 0.2639999985694885.
        [I 2022-08-25 17:40:28,540] Trial 1 finished with value: 0.3160000145435333 and parameters: {'hidden_dim': 10, 'lr': 0.00012587747894812976}. Best is trial 1 with value: 0.3160000145435333.
        [I 2022-08-25 17:40:29,018] Trial 2 finished with value: 0.578000009059906 and parameters: {'hidden_dim': 25, 'lr': 0.0009418378430920174}. Best is trial 2 with value: 0.578000009059906.
        [I 2022-08-25 17:40:29,487] Trial 3 finished with value: 0.7919999957084656 and parameters: {'hidden_dim': 30, 'lr': 0.0019719874263090698}. Best is trial 3 with value: 0.7919999957084656.
        [I 2022-08-25 17:40:29,948] Trial 4 finished with value: 0.7900000214576721 and parameters: {'hidden_dim': 30, 'lr': 0.002768661479102045}. Best is trial 3 with value: 0.7919999957084656.
        [I 2022-08-25 17:40:30,421] Trial 5 finished with value: 0.8019999861717224 and parameters: {'hidden_dim': 28, 'lr': 0.0045199760918655345}. Best is trial 5 with value: 0.8019999861717224.
        [I 2022-08-25 17:40:30,425] Trial 6 pruned. 
        [I 2022-08-25 17:40:30,428] Trial 7 pruned. 
        [I 2022-08-25 17:40:30,431] Trial 8 pruned. 
        [I 2022-08-25 17:40:30,435] Trial 9 pruned. 
        [I 2022-08-25 17:40:30,925] Trial 10 finished with value: 0.800000011920929 and parameters: {'hidden_dim': 23, 'lr': 0.009037693209516048}. Best is trial 5 with value: 0.8019999861717224.
        [I 2022-08-25 17:40:30,933] Trial 11 pruned. 
        [I 2022-08-25 17:40:30,940] Trial 12 pruned. 
        [I 2022-08-25 17:40:31,431] Trial 13 finished with value: 0.7979999780654907 and parameters: {'hidden_dim': 26, 'lr': 0.0042888086003282895}. Best is trial 5 with value: 0.8019999861717224.
        [I 2022-08-25 17:40:31,929] Trial 14 finished with value: 0.7919999957084656 and parameters: {'hidden_dim': 18, 'lr': 0.004496088097060599}. Best is trial 5 with value: 0.8019999861717224.
        [I 2022-08-25 17:40:31,937] Trial 15 pruned. 
        [I 2022-08-25 17:40:31,945] Trial 16 pruned. 
        [I 2022-08-25 17:40:32,066] Trial 17 pruned. 
        [I 2022-08-25 17:40:32,073] Trial 18 pruned. 
        [I 2022-08-25 17:40:32,081] Trial 19 pruned. 
        [I 2022-08-25 17:40:32,089] Trial 20 pruned. 
        [I 2022-08-25 17:40:32,097] Trial 21 pruned. 
        [I 2022-08-25 17:40:32,121] Trial 22 pruned. 
        [I 2022-08-25 17:40:32,129] Trial 23 pruned. 
        [I 2022-08-25 17:40:32,138] Trial 24 pruned. 
        [I 2022-08-25 17:40:32,147] Trial 25 pruned. 
        [I 2022-08-25 17:40:32,155] Trial 26 pruned. 
        [I 2022-08-25 17:40:32,164] Trial 27 pruned. 
        [I 2022-08-25 17:40:32,173] Trial 28 pruned. 
        [I 2022-08-25 17:40:32,199] Trial 29 pruned. 
        [I 2022-08-25 17:40:32,208] Trial 30 pruned. 
        [I 2022-08-25 17:40:32,216] Trial 31 pruned. 
        [I 2022-08-25 17:40:32,712] Trial 32 finished with value: 0.8019999861717224 and parameters: {'hidden_dim': 30, 'lr': 0.004347108689545798}. Best is trial 5 with value: 0.8019999861717224.
        [I 2022-08-25 17:40:32,720] Trial 33 pruned. 
        [I 2022-08-25 17:40:32,728] Trial 34 pruned. 
        [I 2022-08-25 17:40:32,738] Trial 35 pruned. 
        [I 2022-08-25 17:40:33,239] Trial 36 finished with value: 0.7979999780654907 and parameters: {'hidden_dim': 29, 'lr': 0.00753212665126261}. Best is trial 5 with value: 0.8019999861717224.
        [I 2022-08-25 17:40:33,247] Trial 37 pruned. 
        [I 2022-08-25 17:40:33,255] Trial 38 pruned. 
        [I 2022-08-25 17:40:33,264] Trial 39 pruned. 
        [I 2022-08-25 17:40:33,272] Trial 40 pruned. 
        [I 2022-08-25 17:40:33,282] Trial 41 pruned. 
        [I 2022-08-25 17:40:33,293] Trial 42 pruned. 
        [I 2022-08-25 17:40:33,305] Trial 43 pruned. 
        [I 2022-08-25 17:40:33,317] Trial 44 pruned. 
        [I 2022-08-25 17:40:33,327] Trial 45 pruned. 
        [I 2022-08-25 17:40:33,336] Trial 46 pruned. 
        [I 2022-08-25 17:40:33,344] Trial 47 pruned. 
        [I 2022-08-25 17:40:33,355] Trial 48 pruned. 
        [I 2022-08-25 17:40:33,364] Trial 49 pruned. 
        [I 2022-08-25 17:40:33,381] Best trial:
        [I 2022-08-25 17:40:33,382]     Value: 0.802
        [I 2022-08-25 17:40:33,382]     Params:
        [I 2022-08-25 17:40:33,382]             hidden_dim |-> 28
        [I 2022-08-25 17:40:33,382]             lr |-> 0.0045199760918655345
        [I 2022-08-25 17:40:33,413] Final test results:
        [I 2022-08-25 17:40:33,413]     accuracy |-> 0.821
        [I 2022-08-25 17:40:33,413]     f1_score |-> 0.811
        [I 2022-08-25 17:40:33,413]     f1_score -> average@micro |-> 0.821
