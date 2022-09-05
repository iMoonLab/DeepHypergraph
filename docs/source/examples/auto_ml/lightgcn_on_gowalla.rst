LightGCN on Gowalla
=======================

.. hint:: 

    - Author: `Yifan Feng (丰一帆) <https://fengyifan.site/>`_
    - Proof: Xinwei Zhang

In the following example, we present a simple example of Auto-ML for recommender task on User-Item bipartite graph.
More details for how to use the :doc:`/api/experiments` to auto tuning your own model can be found in the tutorial <:doc:`/tutorial/auto_ml`>.

Configuration
--------------

- Model: LightGCN (:py:class:`dhg.models.LightGCN`): `LightGCN: Lightweight Graph Convolutional Networks <https://arxiv.org/pdf/2002.02126>`_ paper (SIGIR 2020).
- Dataset: Gowalla (:py:class:`dhg.data.Gowalla`): The Gowalla dataset is collected for user-item recommendation task. Locations are viewed as items.


Import Libraries
---------------------

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


Define Functions
-------------------

.. code-block:: python

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

Main
-----

.. important:: 

    You must change the ``work_root`` to your own work directory.

.. code-block:: python

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



.. only:: not latex

    Outputs
    -------------

    .. code-block:: 

        [I 2022-08-25 17:52:02,601] Logs will be saved to /home/fengyifan/OS3D/toolbox/exp_cache/tmp/2022-08-25--17-52-02/log.txt
        [I 2022-08-25 17:52:02,601] Files in training will be saved in /home/fengyifan/OS3D/toolbox/exp_cache/tmp/2022-08-25--17-52-02
        [I 2022-08-25 17:52:02,601] Random seed is 2022
        [I 2022-08-25 17:52:02,601] A new study created in memory with name: no-name-a1095326-8011-47c1-8a71-1d8051016f21

