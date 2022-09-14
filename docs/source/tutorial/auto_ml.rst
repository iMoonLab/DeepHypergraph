Training with Auto ML 
========================

.. hint:: 

    - Author: `Yifan Feng (丰一帆) <https://fengyifan.site/>`_
    - Proof: Xinwei Zhang

Auto-ML is a technique that automates the process of searching/selecting the hyper-parameters 
for building a structure, building a model, and training the model. 
DHG's Auto-ML is based on the `Optuna <https://optuna.org/>`_ library.

.. important::

    As for the basic concepts of Auto-ML you should first have a look at the `Get Started with Optuna <https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/001_first.html>`_ .


Builder Functions for Auto-ML
------------------------------

In Auto-ML, `trial <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial>`_ is an important concept that represents a single run of the experiment.
The ``trial`` parameter should be passed to every ``builder`` function as the first parameter.

- :ref:`Structure Builder <tutorial_structure_builder>`
- :ref:`Model Builder <tutorial_model_builder>`
- :ref:`Train Builder <tutorial_train_builder>`

In each builder function, the ``trial`` parameter can be called to suggest hyper-parameters in every single run of the experiment. The following suggestion functions are available:

- `trial.suggest_categorical(name, choices) <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_categorical>`_ : Suggest a value for the categorical parameter.
- `trial.suggest_discrete_uniform(name, low, high, q) <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_discrete_uniform>`_ : Suggest a value for the discrete parameter.
- `trial.suggest_float(name, low, high, step=None, log=False) <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_float>`_ : Suggest a value for the floating point parameter.
- `trial.suggest_int(name, low, high, step=1, log=False) <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_int>`_ : Suggest a value for the integer parameter.
- `trial.suggest_loguniform(name, low, high) <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_loguniform>`_ : Suggest a value for the log-uniform parameter.
- `trial.suggest_uniform(name, low, high) <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial.suggest_uniform>`_ : Suggest a value for the uniform parameter.


.. _tutorial_structure_builder:

Defining the Structure Builder
++++++++++++++++++++++++++++++++++++

The structure builder is a function that defines the input correlation structure, especially for high-order structures like hypergraph. 
Generally, the low-order structure is usually fixed when fed to the model. But the construction of high-order structures is flexible.
Different high-order structures may lead to different performances refer to the `HGNN+ <https://ieeexplore.ieee.org/document/9795251>`_ paper for more details.

In the following examples, we show how to define the structure builder to construct different high-order structures from low-order structures for every single run of the experiment.

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

.. _tutorial_model_builder:

Defining the Model Builder
++++++++++++++++++++++++++++++++++++

The model builder is a function that defines the model architecture like the number of layers, the number of hidden units, and the activation functions.
The model builder should return a model object that is an instance of ``torch.nn.Module``.

In the following examples, we show how to define the model builder to construct different model architectures for every single run of the experiment.

.. code-block:: python

    from dhg.models import HGNNP

    def model_builder(trial):
        global feature_dim, num_classes

        hidden_dim = trial.suggest_int("hidden_dim", 8, 128)
        use_bn = trial.suggest_categorical("use_bn", [True, False])
        model = HGNNP(feature_dim, hidden_dim, num_classes, use_bn=use_bn)

        return model

.. _tutorial_train_builder:

Defining the Train Builder
+++++++++++++++++++++++++++++++

The train builder is a function that defines the training process like the optimizer, and the loss function.
The input parameters of the train builder are the ``trial`` and the ``model`` object.
The return value of the train builder is a dictionary that at least contains the optimizer and the loss function. 
The learn rate ``scheduler`` is optional.

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


Task Class for Auto-ML
------------------------

To run experiments with Auto-ML, we need to define a task-specific class. 
Currently, DHG supports the following tasks:

- :py:class:`dhg.experiments.GraphVertexClassificationTask`: Vertex classification task on graph.
- :py:class:`dhg.experiments.HypergraphVertexClassificationTask`: Vertex classification task on hypergraph.
- :py:class:`dhg.experiments.UserItemRecommenderTask`: Item recommendation task on User-Item bipartite graph.

More Auto-ML tasks will be added in the future. Welcome to contribute and propose issues on `GitHub <https://github.com/iMoonLab/DeepHypergraph>`_.


Auto-ML for Vertex Classification Task
---------------------------------------

In the following examples, we show how to use DHG to run Auto-ML experiments for vertex classification tasks on graph and hypergraph, respectively.

On Graph
++++++++++++++++++++

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

On Hypergraph
++++++++++++++++++++++++

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


Auto-ML for Item Recommender Task
---------------------------------------

In the following example, we show how to use DHG to run Auto-ML experiments for item recommendation tasks on User-Item bipartite graph.

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
