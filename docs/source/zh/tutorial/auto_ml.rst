自动化超参调优
========================

.. hint:: 

    - 作者:  `丰一帆 <https://fengyifan.site/>`_
    - 翻译:  颜杰龙
    - 校对： `丰一帆 <https://fengyifan.site/>`_ 、张欣炜

Auto-ML是一种自动化超参搜索及调优技术，可以帮助您挖掘模型潜力并跑出最高性能。在DHG中我们基于 `Optuna <https://optuna.org/>`_ 库实现
**自动化搜索最优高阶结构** 、 **自动化搜索最优模型架构** 、 **自动化搜索最优训练超参**。

.. important::

    您可以查看 `Optuna上手指南 <https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/001_first.html>`_ 来了解Auto-ML的基本概念。


自动调优的构造函数
------------------------------

在Auto-ML中， `trial <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial>`_ 是代表实验单次运行的重要概念。
``trial`` 参数应作为第一个参数传递给每一个 ``builder`` 函数。

- :ref:`定义结构调优构造函数 <zh_tutorial_structure_builder>`
- :ref:`定义模型调优构造函数 <zh_tutorial_model_builder>`
- :ref:`定义训练调优构造函数 <zh_tutorial_train_builder>`

在每一个构造函数中， ``trial`` 可以在每次实验运行中被调用来 **建议** 当前参数。例如， ``trial.suggest_int`` 可以建议一个整数参数， ``trial.suggest_categorical`` 可以建议一个离散参数， ``trial.suggest_float`` 可以建议一个浮点数参数。

所有可调用的 **建议** 函数如下：

- `trial.suggest_categorical(name, choices) <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_categorical>`_ : 从给定列表中建议出某一项。
- `trial.suggest_discrete_uniform(name, low, high, q) <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_discrete_uniform>`_ : 从离散均匀分布中采样出一个值。
- `trial.suggest_float(name, low, high, step=None, log=False) <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_float>`_ : 采样出一个浮点数。
- `trial.suggest_int(name, low, high, step=1, log=False) <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_int>`_ : 采样出一个整数。
- `trial.suggest_loguniform(name, low, high) <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_loguniform>`_ : 从对数均匀分布中采样出参数值。
- `trial.suggest_uniform(name, low, high) <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_uniform>`_ : 从均匀分布中采样的参数值。


.. _zh_tutorial_structure_builder:

定义结构调优构造函数
++++++++++++++++++++++++++++++++++++

结构调优构造函数一般来说是针对超图之类的高阶关联结构自动搜索出最有效的高阶关联结构。
低阶关联结构在模型使用时一般来说是固定的。
但高阶关联结构的构造是灵活多变的。
不同的高阶关联结构可能会影响最终模型的性能，详情可参考自论文 `HGNN+ <https://ieeexplore.ieee.org/document/9795251>`_ 。

在如下的例子中，我们将展示如何定义结构调优构造函数，每次实验运行从低阶关联结构构建可能的高阶关联结构。

.. code-block:: python

    def structure_builder(trial):
        # ``g`` is the graph, ``X`` is the vertex feature matrix
        global g, X

        hg = dhg.Hypergraph.from_graph(g)
        if trial.suggest_categorical("use_hop1", [True, False]):
            hg.add_hyperedges_from_graph_kHop(g, 1, only_kHop=True)
        if trial.suggest_categorical("use_hop2", [True, False]):
            hg.add_hyperedges_from_graph_kHop(g, 2, only_kHop=True)
        if trial.suggest_categorical("use_feature_knn", [True, False]):
            k = trial.suggest_int("k", 1, 10)
            hg.add_hyperedges_from_feature_kNN(X, k)
        
        return hg

.. _zh_tutorial_model_builder:

定义模型调优构造函数
++++++++++++++++++++++++++++++++++++

模型调优构造函数是定义层数、隐藏层维度、激活函数等模型架构的函数。
模型调优构造函数的返回值是一个模型对象，其为 ``torch.nn.Module`` 的一个实例。

在如下的例子中，我们将展示如何定义模型调优构造函数，每次实验运行构建不同模型架构。

.. code-block:: python

    from dhg.models import HGNNP

    def model_builder(trial):
        global feature_dim, num_classes

        hidden_dim = trial.suggest_int("hidden_dim", 8, 128)
        use_bn = trial.suggest_categorical("use_bn", [True, False])
        model = HGNNP(feature_dim, hidden_dim, num_classes, use_bn=use_bn)

        return model

.. _zh_tutorial_train_builder:

定义训练调优构造函数
+++++++++++++++++++++++++++++++

训练调优构造函数是定义优化器、损失函数等训练过程中所需的对象。
训练调优构造函数的输入参数为 ``trial`` 和 ``model`` 对象。
训练调优构造函数的返回值是一个至少包含优化器和损失函数的字典。
学习率调整器 ``scheduler`` 是可选的。

.. code-block:: python

    import torch.nn as nn
    import torch.optim as optim

    def train_builder(trial, model):
        optimizer = optim.Adam(
            model.parameters(),
            lr=trial.suggest_loguniform("lr", 1e-4, 1e-2),
            weight_decay=trial.suggest_loguniform("weight_decay", 1e-4, 1e-2),
        )
        criterion = nn.CrossEntropyLoss()
        return {
            "optimizer": optimizer,
            "criterion": criterion,
        }


自动化调优的任务类
------------------------

我们需要定义一个任务特定的类，来使用Auto-ML实验。
目前，DHG支持以下任务：

- :py:class:`dhg.experiments.GraphVertexClassificationTask`: 图上的顶点分类任务。
- :py:class:`dhg.experiments.HypergraphVertexClassificationTask`: 超图上的顶点分类任务。
- :py:class:`dhg.experiments.UserItemRecommenderTask`: <用户-物品>二分图上的物品推荐任务。

更多的Auto-ML任务将会在以后添加。期待您的贡献以及在 `GitHub <https://github.com/iMoonLab/DeepHypergraph>`_ 上提出问题。


自动化节点分类任务
---------------------------------------

在如下的例子中，我们将分别在图和超图的顶点分类任务中介绍如何使用DHG的Auto-ML进行实验。

自动化图节点分类任务
++++++++++++++++++++++++

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


    def model_builder(trial):
        return GCN(ft_dim, trial.suggest_int("hidden_dim", 8, 32), num_classes)


    def train_builder(trial, model):
        optimizer = optim.Adam(model.parameters(), lr=trial.suggest_loguniform("lr", 1e-4, 1e-2), weight_decay=5e-4,)
        criterion = nn.CrossEntropyLoss()
        return {
            "optimizer": optimizer,
            "criterion": criterion,
        }
    

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

自动化超图节点分类任务
+++++++++++++++++++++++++++

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.optim as optim

    from dhg import Hypergraph
    from dhg.data import Cooking200
    from dhg.models import HGNNP
    from dhg.random import set_seed
    from dhg.experiments import HypergraphVertexClassificationTask as Task
    from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator

    def structure_builder(trial):
        global hg_base, g
        cur_hg: Hypergraph = hg_base.clone()
        return cur_hg


    def model_builder(trial):
        return HGNNP(dim_features, trial.suggest_int("hidden_dim", 10, 20), num_classes, use_bn=True)


    def train_builder(trial, model):
        optimizer = optim.Adam(
            model.parameters(),
            lr=trial.suggest_loguniform("lr", 1e-4, 1e-2),
            weight_decay=trial.suggest_loguniform("weight_decay", 1e-4, 1e-2),
        )
        criterion = nn.CrossEntropyLoss()
        return {
            "optimizer": optimizer,
            "criterion": criterion,
        }


    if __name__ == "__main__":
        work_root = "/home/fengyifan/OS3D/toolbox/exp_cache/tmp"
        set_seed(2022)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        data = Cooking200()
        dim_features = data["num_vertices"]
        num_classes = data["num_classes"]
        hg_base = Hypergraph(data["num_vertices"], data["edge_list"])
        input_data = {
            "features": torch.eye(data["num_vertices"]),
            "labels": data["labels"],
            "train_mask": data["train_mask"],
            "val_mask": data["val_mask"],
            "test_mask": data["test_mask"],
        }
        evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
        task = Task(
            work_root, input_data, model_builder, train_builder, evaluator, device, structure_builder=structure_builder,
        )
        task.run(200, 50, "maximize")


自动化物品推荐任务
---------------------------------------

在如下的例子中，我们将在<用户-物品>二分图的物品推荐任务中介绍如何使用DHG的Auto-ML进行实验。

.. code-block:: python

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from dhg import BiGraph
    from dhg.data import Gowalla
    from dhg.models import LightGCN
    from dhg.nn import BPRLoss, EmbeddingRegularization
    from dhg.experiments import UserItemRecommenderTask as Task
    from dhg.metrics import UserItemRecommenderEvaluator as Evaluator
    from dhg.random import set_seed
    from dhg.utils import UserItemDataset, adj_list_to_edge_list


    class BPR_Reg(nn.Module):
        def __init__(self, weight_decay):
            super().__init__()
            self.reg = EmbeddingRegularization(p=2, weight_decay=weight_decay)
            self.bpr = BPRLoss(activation="softplus")

        def forward(self, emb_users, emb_items, users, pos_items, neg_items, model):
            cur_u = emb_users[users]
            cur_pos_i, cur_neg_i = emb_items[pos_items], emb_items[neg_items]
            pos_scores, neg_scores = (cur_u * cur_pos_i).sum(dim=1), (cur_u * cur_neg_i).sum(dim=1)
            loss_bpr = self.bpr(pos_scores, neg_scores)
            raw_emb_users, raw_emb_items = model.u_embedding.weight, model.i_embedding.weight
            raw_u = raw_emb_users[users]
            raw_pos_i, raw_neg_i = raw_emb_items[pos_items], raw_emb_items[neg_items]
            loss_l2 = self.reg(raw_u, raw_pos_i, raw_neg_i)
            loss = loss_bpr + loss_l2

            return loss


    def model_builder(trial):
        return LightGCN(num_u, num_i, trial.suggest_int("hidden_dim", 20, 80))


    def train_builder(trial, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=trial.suggest_loguniform("lr", 1e-4, 1e-2))
        criterion = BPR_Reg(weight_decay=trial.suggest_loguniform("weight_decay", 1e-5, 1e-3))
        return {
            "optimizer": optimizer,
            "criterion": criterion,
        }


    if __name__ == "__main__":
        work_root = "/home/fengyifan/OS3D/toolbox/exp_cache/tmp"
        dim_emb = 64
        lr = 0.001
        num_workers = 0
        batch_sz = 2048
        val_freq = 20
        epoch_max = 500
        weight_decay = 1e-4
        set_seed(2022)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        evaluator = Evaluator([{"ndcg": {"k": 20}}, {"recall": {"k": 20}}])
        # data = MovieLens1M()
        data = Gowalla()
        num_u, num_i = data["num_users"], data["num_items"]
        train_adj_list = data["train_adj_list"]
        test_adj_list = data["test_adj_list"]
        ui_bigraph = BiGraph.from_adj_list(num_u, num_i, train_adj_list)
        ui_bigraph = ui_bigraph.to(device)
        train_edge_list = adj_list_to_edge_list(train_adj_list)
        test_edge_list = adj_list_to_edge_list(test_adj_list)
        train_dataset = UserItemDataset(num_u, num_i, train_edge_list)
        test_dataset = UserItemDataset(num_u, num_i, test_edge_list, train_user_item_list=train_edge_list, phase="test")
        train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=False, num_workers=num_workers)

        input_data = {
            "train_loader": train_loader,
            "test_loader": test_loader,
            "structure": ui_bigraph,
        }
        task = Task(work_root, input_data, model_builder, train_builder, evaluator, device)
        task.run(10, 300, "maximize")
