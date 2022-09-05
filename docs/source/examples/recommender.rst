User-Item Recommender
====================================

.. hint:: 

    - Author: `Yifan Feng (丰一帆) <https://fengyifan.site/>`_
    - Proof: Xinwei Zhang

In the following examples, two typical collaborative filtering neural networks are used to perform item recommender on the User-Item bipartite graph.

Models
---------------------------

- NGCF (:py:class:`dhg.models.GCN`), `Neural Graph Collaborative Filtering <https://arxiv.org/pdf/1905.08108>`_ paper (SIGIR 2019).
- LightGCN (:py:class:`dhg.models.LightGCN`), `LightGCN: Lightweight Graph Convolutional Networks <https://arxiv.org/pdf/2002.02126>`_ paper (SIGIR 2020).

Dataset
---------------------------

The Gowalla dataset (:py:class:`dhg.data.Gowalla`) is collected for user-item recommendation task. Locations are viewed as items.
The full dataset can be found in this `website <https://snap.stanford.edu/data/loc-gowalla.html>`_.

Results
----------------

========    ======================  ==========================
Model       `NDCG@20` on Testing    `Recall@20` on Testing
========    ======================  ==========================
NGCF        0.1307                  0.1547
LightGCN    0.1550                  0.1830
========    ======================  ==========================


NGCF on Gowalla
-----------------

Import Libraries
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import time
    from copy import deepcopy

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from dhg import BiGraph
    from dhg.data import Gowalla
    from dhg.models import NGCF
    from dhg.nn import BPRLoss, EmbeddingRegularization
    from dhg.metrics import UserItemRecommenderEvaluator as Evaluator
    from dhg.random import set_seed
    from dhg.utils import UserItemDataset, adj_list_to_edge_list


Define Functions
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class BPR_Reg(nn.Module):
        def __init__(self, weight_decay):
            super().__init__()
            self.reg = EmbeddingRegularization(p=2, weight_decay=weight_decay)
            self.bpr = BPRLoss(activation="softplus")

        def forward(self, emb_users, emb_items, users, pos_items, neg_items, raw_emb_users, raw_emb_items):
            cur_u = emb_users[users]
            cur_pos_i, cur_neg_i = emb_items[pos_items], emb_items[neg_items]
            pos_scores, neg_scores = (cur_u * cur_pos_i).sum(dim=1), (cur_u * cur_neg_i).sum(dim=1)
            loss_bpr = self.bpr(pos_scores, neg_scores)
            raw_u = raw_emb_users[users]
            raw_pos_i, raw_neg_i = raw_emb_items[pos_items], raw_emb_items[neg_items]
            loss_l2 = self.reg(raw_u, raw_pos_i, raw_neg_i)
            loss = loss_bpr + loss_l2

            return loss


    def train(net, data_loader, optimizer, criterion, epoch):
        net.train()

        loss_mean, st = 0, time.time()
        for users, pos_items, neg_items in data_loader:
            users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)
            optimizer.zero_grad()
            emb_users, emb_items = net(ui_bigraph)
            loss = criterion(
                emb_users, emb_items, users, pos_items, neg_items, net.u_embedding.weight, net.i_embedding.weight,
            )
            loss.backward()
            optimizer.step()
            loss_mean += loss.item() * users.shape[0]
        loss_mean /= len(data_loader.dataset)
        print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss_mean:.5f}")


    @torch.no_grad()
    def validate(net, data_loader):
        net.eval()

        for users, train_mask, true_rating in data_loader:
            users, train_mask, true_rating = (
                users.to(device),
                train_mask.to(device),
                true_rating.to(device),
            )

            emb_users, emb_items = net(ui_bigraph)
            pred_rating = emb_users[users] @ emb_items.t()
            pred_rating += train_mask
            evaluator.validate_add_batch(true_rating, pred_rating)
        return evaluator.validate_epoch_res()


    @torch.no_grad()
    def test(net, data_loader):
        net.eval()

        for users, train_mask, true_rating in data_loader:
            users, train_mask, true_rating = (
                users.to(device),
                train_mask.to(device),
                true_rating.to(device),
            )
            emb_users, emb_items = net(ui_bigraph)
            pred_rating = emb_users[users] @ emb_items.t()
            pred_rating += train_mask
            evaluator.test_add_batch(true_rating, pred_rating)
        return evaluator.test_epoch_res()


Main
^^^^^^^^^^^

.. note::

    More details about the metric ``Evaluator`` can be found in the :doc:`Building Evaluator </tutorial/metric>` section.

.. code-block:: python

    if __name__ == "__main__":
        dim_emb = 64
        lr = 0.001
        num_workers = 0
        batch_sz = 2048
        val_freq = 20
        epoch_max = 1000
        weight_decay = 1e-4
        set_seed(2022)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        evaluator = Evaluator([{"ndcg": {"k": 20}}, {"recall": {"k": 20}}])

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

        net = NGCF(num_u, num_i, dim_emb)
        net = net.to(device)
        criterion = BPR_Reg(weight_decay)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        best_state, best_val, best_epoch = None, 0, -1
        for epoch in range(epoch_max):
            train(net, train_loader, optimizer, criterion, epoch)
            if epoch % val_freq == 0:
                val_res = validate(net, test_loader)
                print(f"Validation: NDCG@20 -> {val_res}")
                if val_res > best_val:
                    best_epoch = epoch
                    best_val = val_res
                    best_state = deepcopy(net.state_dict())
        print("train finished")
        print(f"best val: {best_val}")
        print(f"best epoch: {best_epoch}")
        print("testing...")
        net.load_state_dict(best_state)
        test_res = test(net, test_loader)
        print(f"test res: {test_res}")

.. only:: not latex

    Outputs
    ^^^^^^^^^^^

    .. code-block::

        Epoch: 0, Time: 17.58267s, Loss: 0.57975
        Validation: NDCG@20 -> 0.059597898178090525
        Epoch: 1, Time: 17.35420s, Loss: 0.53948
        Epoch: 2, Time: 16.67790s, Loss: 0.49900
        Epoch: 3, Time: 16.58108s, Loss: 0.45402
        Epoch: 4, Time: 16.49324s, Loss: 0.41055
        Epoch: 5, Time: 16.47659s, Loss: 0.37169
        Epoch: 6, Time: 16.50014s, Loss: 0.33842
        Epoch: 7, Time: 16.53070s, Loss: 0.31066
        Epoch: 8, Time: 16.50077s, Loss: 0.28642
        Epoch: 9, Time: 16.47511s, Loss: 0.26709
        Epoch: 10, Time: 16.48582s, Loss: 0.25038
        Epoch: 11, Time: 16.48268s, Loss: 0.23593
        Epoch: 12, Time: 16.55598s, Loss: 0.22323
        Epoch: 13, Time: 16.59955s, Loss: 0.21264
        Epoch: 14, Time: 16.57648s, Loss: 0.20171
        Epoch: 15, Time: 16.61875s, Loss: 0.19399
        Epoch: 16, Time: 16.60113s, Loss: 0.18529
        Epoch: 17, Time: 16.67147s, Loss: 0.17826
        Epoch: 18, Time: 16.39420s, Loss: 0.17196
        Epoch: 19, Time: 16.43819s, Loss: 0.16577
        Epoch: 20, Time: 16.39095s, Loss: 0.16056
        Validation: NDCG@20 -> 0.0796889192830519
        Epoch: 21, Time: 16.38881s, Loss: 0.15489
        Epoch: 22, Time: 16.38985s, Loss: 0.15104
        Epoch: 23, Time: 16.34736s, Loss: 0.14618
        Epoch: 24, Time: 16.45004s, Loss: 0.14248
        Epoch: 25, Time: 16.47847s, Loss: 0.13837
        Epoch: 26, Time: 16.37476s, Loss: 0.13460
        Epoch: 27, Time: 16.39726s, Loss: 0.13152
        Epoch: 28, Time: 16.46616s, Loss: 0.12831
        Epoch: 29, Time: 16.42933s, Loss: 0.12524
        Epoch: 30, Time: 16.42688s, Loss: 0.12196
        Epoch: 31, Time: 16.39388s, Loss: 0.11980
        Epoch: 32, Time: 16.45209s, Loss: 0.11667
        Epoch: 33, Time: 16.38611s, Loss: 0.11485
        Epoch: 34, Time: 16.38676s, Loss: 0.11186
        Epoch: 35, Time: 16.43171s, Loss: 0.11052
        Epoch: 36, Time: 16.42350s, Loss: 0.10853
        Epoch: 37, Time: 16.46971s, Loss: 0.10643
        Epoch: 38, Time: 16.41361s, Loss: 0.10481
        Epoch: 39, Time: 16.40113s, Loss: 0.10274
        Epoch: 40, Time: 16.45297s, Loss: 0.10065
        Validation: NDCG@20 -> 0.09484630939006403
        Epoch: 41, Time: 16.20096s, Loss: 0.09951
        Epoch: 42, Time: 16.33159s, Loss: 0.09786
        Epoch: 43, Time: 16.41295s, Loss: 0.09629
        Epoch: 44, Time: 16.29521s, Loss: 0.09473
        Epoch: 45, Time: 16.31462s, Loss: 0.09310
        Epoch: 46, Time: 16.30070s, Loss: 0.09155
        Epoch: 47, Time: 16.42125s, Loss: 0.09050
        Epoch: 48, Time: 16.34268s, Loss: 0.08982
        Epoch: 49, Time: 16.38188s, Loss: 0.08804
        Epoch: 50, Time: 16.35001s, Loss: 0.08735
        Epoch: 51, Time: 16.30478s, Loss: 0.08568
        Epoch: 52, Time: 16.26564s, Loss: 0.08473
        Epoch: 53, Time: 16.38207s, Loss: 0.08340
        Epoch: 54, Time: 16.34952s, Loss: 0.08275
        Epoch: 55, Time: 16.41525s, Loss: 0.08166
        Epoch: 56, Time: 16.34333s, Loss: 0.08030
        Epoch: 57, Time: 16.43872s, Loss: 0.07994
        Epoch: 58, Time: 16.37634s, Loss: 0.07892
        Epoch: 59, Time: 16.37193s, Loss: 0.07846
        Epoch: 60, Time: 16.36561s, Loss: 0.07732
        Validation: NDCG@20 -> 0.10073506573468528
        Epoch: 61, Time: 16.31512s, Loss: 0.07683
        Epoch: 62, Time: 16.48562s, Loss: 0.07560
        Epoch: 63, Time: 16.38161s, Loss: 0.07542
        Epoch: 64, Time: 16.38181s, Loss: 0.07415
        Epoch: 65, Time: 16.37734s, Loss: 0.07392
        Epoch: 66, Time: 16.35093s, Loss: 0.07365
        Epoch: 67, Time: 16.42241s, Loss: 0.07198
        Epoch: 68, Time: 16.39753s, Loss: 0.07206
        Epoch: 69, Time: 16.43910s, Loss: 0.07088
        Epoch: 70, Time: 16.40806s, Loss: 0.07004
        Epoch: 71, Time: 16.38006s, Loss: 0.07041
        Epoch: 72, Time: 16.42882s, Loss: 0.06922
        Epoch: 73, Time: 16.41414s, Loss: 0.06855
        Epoch: 74, Time: 16.34444s, Loss: 0.06793
        Epoch: 75, Time: 16.40675s, Loss: 0.06769
        Epoch: 76, Time: 16.41324s, Loss: 0.06697
        Epoch: 77, Time: 16.38147s, Loss: 0.06661
        Epoch: 78, Time: 16.42382s, Loss: 0.06648
        Epoch: 79, Time: 16.41072s, Loss: 0.06594
        Epoch: 80, Time: 16.38907s, Loss: 0.06481
        Validation: NDCG@20 -> 0.10532317576637099
        Epoch: 81, Time: 16.42970s, Loss: 0.06468
        Epoch: 82, Time: 16.45658s, Loss: 0.06442
        Epoch: 83, Time: 16.38556s, Loss: 0.06388
        Epoch: 84, Time: 16.32818s, Loss: 0.06370
        Epoch: 85, Time: 16.36058s, Loss: 0.06294
        Epoch: 86, Time: 16.34388s, Loss: 0.06260
        Epoch: 87, Time: 16.33080s, Loss: 0.06234
        Epoch: 88, Time: 16.36727s, Loss: 0.06197
        Epoch: 89, Time: 16.32790s, Loss: 0.06154
        Epoch: 90, Time: 16.43729s, Loss: 0.06101
        Epoch: 91, Time: 16.38772s, Loss: 0.06070
        Epoch: 92, Time: 16.42943s, Loss: 0.06037
        Epoch: 93, Time: 16.36849s, Loss: 0.06043
        Epoch: 94, Time: 16.39440s, Loss: 0.05969
        Epoch: 95, Time: 16.33486s, Loss: 0.05954
        Epoch: 96, Time: 16.34549s, Loss: 0.05876
        Epoch: 97, Time: 16.37610s, Loss: 0.05866
        Epoch: 98, Time: 16.39110s, Loss: 0.05857
        Epoch: 99, Time: 16.38359s, Loss: 0.05788
        Epoch: 100, Time: 16.42878s, Loss: 0.05773
        Validation: NDCG@20 -> 0.10774315184649631
        Epoch: 101, Time: 16.37178s, Loss: 0.05742
        Epoch: 102, Time: 16.50821s, Loss: 0.05743
        Epoch: 103, Time: 16.38737s, Loss: 0.05706
        Epoch: 104, Time: 16.38123s, Loss: 0.05672
        Epoch: 105, Time: 16.38323s, Loss: 0.05625
        Epoch: 106, Time: 16.39332s, Loss: 0.05609
        Epoch: 107, Time: 16.38817s, Loss: 0.05554
        Epoch: 108, Time: 16.39039s, Loss: 0.05561
        Epoch: 109, Time: 16.40110s, Loss: 0.05534
        Epoch: 110, Time: 16.42629s, Loss: 0.05496
        Epoch: 111, Time: 16.40456s, Loss: 0.05436
        Epoch: 112, Time: 16.42960s, Loss: 0.05448
        Epoch: 113, Time: 16.41036s, Loss: 0.05448
        Epoch: 114, Time: 16.38433s, Loss: 0.05405
        Epoch: 115, Time: 16.38922s, Loss: 0.05338
        Epoch: 116, Time: 16.37122s, Loss: 0.05375
        Epoch: 117, Time: 16.39454s, Loss: 0.05359
        Epoch: 118, Time: 16.37232s, Loss: 0.05301
        Epoch: 119, Time: 16.38497s, Loss: 0.05317
        Epoch: 120, Time: 16.44990s, Loss: 0.05326
        Validation: NDCG@20 -> 0.11050138281284864
        Epoch: 121, Time: 16.42819s, Loss: 0.05270
        Epoch: 122, Time: 16.43767s, Loss: 0.05240
        Epoch: 123, Time: 16.33994s, Loss: 0.05205
        Epoch: 124, Time: 16.37961s, Loss: 0.05193
        Epoch: 125, Time: 16.40023s, Loss: 0.05187
        Epoch: 126, Time: 16.44434s, Loss: 0.05143
        Epoch: 127, Time: 16.44631s, Loss: 0.05155
        Epoch: 128, Time: 16.42970s, Loss: 0.05141
        Epoch: 129, Time: 16.43539s, Loss: 0.05119
        Epoch: 130, Time: 16.41379s, Loss: 0.05097
        Epoch: 131, Time: 16.43115s, Loss: 0.05080
        Epoch: 132, Time: 16.41100s, Loss: 0.05077
        Epoch: 133, Time: 16.42312s, Loss: 0.05043
        Epoch: 134, Time: 16.39068s, Loss: 0.05028
        Epoch: 135, Time: 16.37832s, Loss: 0.05016
        Epoch: 136, Time: 16.39196s, Loss: 0.04994
        Epoch: 137, Time: 16.38732s, Loss: 0.04976
        Epoch: 138, Time: 16.41807s, Loss: 0.04935
        Epoch: 139, Time: 16.37651s, Loss: 0.04916
        Epoch: 140, Time: 16.39615s, Loss: 0.04923
        Validation: NDCG@20 -> 0.11280099123452347
        Epoch: 141, Time: 16.41225s, Loss: 0.04903
        Epoch: 142, Time: 16.46800s, Loss: 0.04892
        Epoch: 143, Time: 16.39678s, Loss: 0.04835
        Epoch: 144, Time: 16.38563s, Loss: 0.04838
        Epoch: 145, Time: 16.37892s, Loss: 0.04874
        Epoch: 146, Time: 16.46196s, Loss: 0.04824
        Epoch: 147, Time: 16.39248s, Loss: 0.04801
        Epoch: 148, Time: 16.37935s, Loss: 0.04801
        Epoch: 149, Time: 16.44855s, Loss: 0.04773
        Epoch: 150, Time: 16.94777s, Loss: 0.04736
        Epoch: 151, Time: 17.25382s, Loss: 0.04770
        Epoch: 152, Time: 17.55223s, Loss: 0.04734
        Epoch: 153, Time: 17.03791s, Loss: 0.04729
        Epoch: 154, Time: 17.59021s, Loss: 0.04759
        Epoch: 155, Time: 17.50267s, Loss: 0.04705
        Epoch: 156, Time: 17.43284s, Loss: 0.04690
        Epoch: 157, Time: 16.67660s, Loss: 0.04659
        Epoch: 158, Time: 17.15853s, Loss: 0.04668
        Epoch: 159, Time: 16.93252s, Loss: 0.04653
        Epoch: 160, Time: 16.66944s, Loss: 0.04636
        Validation: NDCG@20 -> 0.11396838930066855
        Epoch: 161, Time: 16.75059s, Loss: 0.04627
        Epoch: 162, Time: 16.80186s, Loss: 0.04613
        Epoch: 163, Time: 16.75320s, Loss: 0.04616
        Epoch: 164, Time: 16.79349s, Loss: 0.04604
        Epoch: 165, Time: 16.82817s, Loss: 0.04579
        Epoch: 166, Time: 16.78084s, Loss: 0.04599
        Epoch: 167, Time: 16.83057s, Loss: 0.04553
        Epoch: 168, Time: 16.83778s, Loss: 0.04554
        Epoch: 169, Time: 16.83636s, Loss: 0.04548
        Epoch: 170, Time: 16.76483s, Loss: 0.04547
        Epoch: 171, Time: 16.85442s, Loss: 0.04487
        Epoch: 172, Time: 16.83118s, Loss: 0.04475
        Epoch: 173, Time: 16.80676s, Loss: 0.04518
        Epoch: 174, Time: 16.82507s, Loss: 0.04470
        Epoch: 175, Time: 16.87042s, Loss: 0.04485
        Epoch: 176, Time: 17.00146s, Loss: 0.04471
        Epoch: 177, Time: 17.02007s, Loss: 0.04455
        Epoch: 178, Time: 16.63682s, Loss: 0.04445
        Epoch: 179, Time: 17.08953s, Loss: 0.04450
        Epoch: 180, Time: 16.89926s, Loss: 0.04419
        Validation: NDCG@20 -> 0.11516925413130324


LightGCN on Gowalla
-----------------------------------


Import Libraries
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import time
    from copy import deepcopy

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from dhg import BiGraph
    from dhg.data import Gowalla
    from dhg.models import LightGCN
    from dhg.nn import BPRLoss, EmbeddingRegularization
    from dhg.metrics import UserItemRecommenderEvaluator as Evaluator
    from dhg.random import set_seed
    from dhg.utils import UserItemDataset, adj_list_to_edge_list


Define Functions
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    class BPR_Reg(nn.Module):
        def __init__(self, weight_decay):
            super().__init__()
            self.reg = EmbeddingRegularization(p=2, weight_decay=weight_decay)
            self.bpr = BPRLoss(activation="softplus")

        def forward(self, emb_users, emb_items, users, pos_items, neg_items, raw_emb_users, raw_emb_items):
            cur_u = emb_users[users]
            cur_pos_i, cur_neg_i = emb_items[pos_items], emb_items[neg_items]
            pos_scores, neg_scores = (cur_u * cur_pos_i).sum(dim=1), (cur_u * cur_neg_i).sum(dim=1)
            loss_bpr = self.bpr(pos_scores, neg_scores)
            raw_u = raw_emb_users[users]
            raw_pos_i, raw_neg_i = raw_emb_items[pos_items], raw_emb_items[neg_items]
            loss_l2 = self.reg(raw_u, raw_pos_i, raw_neg_i)
            loss = loss_bpr + loss_l2

            return loss


    def train(net, data_loader, optimizer, criterion, epoch):
        net.train()

        loss_mean, st = 0, time.time()
        for users, pos_items, neg_items in data_loader:
            users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)
            optimizer.zero_grad()
            emb_users, emb_items = net(ui_bigraph)
            loss = criterion(
                emb_users, emb_items, users, pos_items, neg_items, net.u_embedding.weight, net.i_embedding.weight,
            )
            loss.backward()
            optimizer.step()
            loss_mean += loss.item() * users.shape[0]
        loss_mean /= len(data_loader.dataset)
        print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss_mean:.5f}")


    @torch.no_grad()
    def validate(net, data_loader):
        net.eval()

        for users, train_mask, true_rating in data_loader:
            users, train_mask, true_rating = (
                users.to(device),
                train_mask.to(device),
                true_rating.to(device),
            )

            emb_users, emb_items = net(ui_bigraph)
            pred_rating = emb_users[users] @ emb_items.t()
            pred_rating += train_mask
            evaluator.validate_add_batch(true_rating, pred_rating)
        return evaluator.validate_epoch_res()


    @torch.no_grad()
    def test(net, data_loader):
        net.eval()

        for users, train_mask, true_rating in data_loader:
            users, train_mask, true_rating = (
                users.to(device),
                train_mask.to(device),
                true_rating.to(device),
            )
            emb_users, emb_items = net(ui_bigraph)
            pred_rating = emb_users[users] @ emb_items.t()
            pred_rating += train_mask
            evaluator.test_add_batch(true_rating, pred_rating)
        return evaluator.test_epoch_res()


Main
^^^^^^^^^^^

.. note::

    More details about the metric ``Evaluator`` can be found in the :doc:`Building Evaluator </zh/tutorial/metric>` section.

.. code-block:: python

    if __name__ == "__main__":
        dim_emb = 64
        lr = 0.001
        num_workers = 0
        batch_sz = 2048
        val_freq = 20
        epoch_max = 1000
        weight_decay = 1e-4
        set_seed(2022)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        evaluator = Evaluator([{"ndcg": {"k": 20}}, {"recall": {"k": 20}}])

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

        net = LightGCN(num_u, num_i, dim_emb)
        net = net.to(device)
        criterion = BPR_Reg(weight_decay)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        best_state, best_val, best_epoch = None, 0, -1
        for epoch in range(epoch_max):
            train(net, train_loader, optimizer, criterion, epoch)
            if epoch % val_freq == 0:
                val_res = validate(net, test_loader)
                print(f"Validation: NDCG@20 -> {val_res}")
                if val_res > best_val:
                    best_epoch = epoch
                    best_val = val_res
                    best_state = deepcopy(net.state_dict())
        print("train finished")
        print(f"best val: {best_val}")
        print(f"best epoch: {best_epoch}")
        print("testing...")
        net.load_state_dict(best_state)
        test_res = test(net, test_loader)
        print(f"test res: {test_res}")

.. only:: not latex

    Outputs
    ^^^^^^^^^^^

    .. code-block::

        Epoch: 0, Time: 13.99548s, Loss: 0.50885
        Validation: 0.06785113136172075
        Epoch: 1, Time: 13.64900s, Loss: 0.23104
        Epoch: 2, Time: 14.04344s, Loss: 0.17122
        Epoch: 3, Time: 14.17638s, Loss: 0.14329
        Epoch: 4, Time: 14.00283s, Loss: 0.12590
        Epoch: 5, Time: 13.74365s, Loss: 0.11401
        Epoch: 6, Time: 13.84788s, Loss: 0.10609
        Epoch: 7, Time: 13.74667s, Loss: 0.09946
        Epoch: 8, Time: 13.52109s, Loss: 0.09344
        Epoch: 9, Time: 13.36949s, Loss: 0.08926
        Epoch: 10, Time: 13.38029s, Loss: 0.08551
        Epoch: 11, Time: 13.34837s, Loss: 0.08222
        Epoch: 12, Time: 13.35350s, Loss: 0.08036
        Epoch: 13, Time: 13.40274s, Loss: 0.07683
        Epoch: 14, Time: 13.40829s, Loss: 0.07503
        Epoch: 15, Time: 12.97056s, Loss: 0.07243
        Epoch: 16, Time: 12.99591s, Loss: 0.07044
        Epoch: 17, Time: 13.00338s, Loss: 0.06855
        Epoch: 18, Time: 12.99054s, Loss: 0.06646
        Epoch: 19, Time: 12.99395s, Loss: 0.06438
        Epoch: 20, Time: 12.98960s, Loss: 0.06309
        Validation: 0.11326732434508542
        Epoch: 21, Time: 12.86631s, Loss: 0.06106
        Epoch: 22, Time: 12.91743s, Loss: 0.06020
        Epoch: 23, Time: 12.98517s, Loss: 0.05888
        Epoch: 24, Time: 12.94291s, Loss: 0.05717
        Epoch: 25, Time: 13.02582s, Loss: 0.05655
        Epoch: 26, Time: 13.63684s, Loss: 0.05494
        Epoch: 27, Time: 13.43329s, Loss: 0.05417
        Epoch: 28, Time: 13.40190s, Loss: 0.05334
        Epoch: 29, Time: 13.34597s, Loss: 0.05158
        Epoch: 30, Time: 13.39115s, Loss: 0.05144
        Epoch: 31, Time: 13.36618s, Loss: 0.05128
        Epoch: 32, Time: 13.39683s, Loss: 0.04959
        Epoch: 33, Time: 13.35201s, Loss: 0.04861
        Epoch: 34, Time: 13.36965s, Loss: 0.04781
        Epoch: 35, Time: 13.66474s, Loss: 0.04725
        Epoch: 36, Time: 14.06111s, Loss: 0.04681
        Epoch: 37, Time: 13.79196s, Loss: 0.04575
        Epoch: 38, Time: 13.82344s, Loss: 0.04595
        Epoch: 39, Time: 13.98216s, Loss: 0.04427
        Epoch: 40, Time: 14.05523s, Loss: 0.04426
        Validation: 0.12357260732699984
        Epoch: 41, Time: 13.36727s, Loss: 0.04292
        Epoch: 42, Time: 13.37445s, Loss: 0.04322
        Epoch: 43, Time: 13.38032s, Loss: 0.04226
        Epoch: 44, Time: 13.40528s, Loss: 0.04133
        Epoch: 45, Time: 14.11705s, Loss: 0.04139
        Epoch: 46, Time: 13.51289s, Loss: 0.04059
        Epoch: 47, Time: 13.63507s, Loss: 0.03985
        Epoch: 48, Time: 13.82129s, Loss: 0.03967
        Epoch: 49, Time: 13.38149s, Loss: 0.03917
        Epoch: 50, Time: 13.61731s, Loss: 0.03890
        Epoch: 51, Time: 13.77848s, Loss: 0.03834
        Epoch: 52, Time: 13.78244s, Loss: 0.03772
        Epoch: 53, Time: 13.53519s, Loss: 0.03744
        Epoch: 54, Time: 13.56650s, Loss: 0.03690
        Epoch: 55, Time: 13.77765s, Loss: 0.03633
        Epoch: 56, Time: 13.55891s, Loss: 0.03594
        Epoch: 57, Time: 13.82406s, Loss: 0.03581
        Epoch: 58, Time: 13.62316s, Loss: 0.03546
        Epoch: 59, Time: 13.86439s, Loss: 0.03511
        Epoch: 60, Time: 13.75384s, Loss: 0.03478
        Validation: 0.13109645468633707
        Epoch: 61, Time: 14.04090s, Loss: 0.03443
        Epoch: 62, Time: 13.59308s, Loss: 0.03342
        Epoch: 63, Time: 13.47868s, Loss: 0.03315
        Epoch: 64, Time: 13.58020s, Loss: 0.03313
        Epoch: 65, Time: 13.78613s, Loss: 0.03299
        Epoch: 66, Time: 14.13540s, Loss: 0.03287
        Epoch: 67, Time: 13.88064s, Loss: 0.03239
        Epoch: 68, Time: 14.19946s, Loss: 0.03220
        Epoch: 69, Time: 13.85164s, Loss: 0.03172
        Epoch: 70, Time: 13.80321s, Loss: 0.03161
        Epoch: 71, Time: 13.59180s, Loss: 0.03125
        Epoch: 72, Time: 13.57149s, Loss: 0.03068
        Epoch: 73, Time: 13.87281s, Loss: 0.03073
        Epoch: 74, Time: 13.98456s, Loss: 0.03003
        Epoch: 75, Time: 13.83081s, Loss: 0.03033
        Epoch: 76, Time: 13.60854s, Loss: 0.02954
        Epoch: 77, Time: 13.74393s, Loss: 0.02925
        Epoch: 78, Time: 13.82418s, Loss: 0.02909
        Epoch: 79, Time: 13.55567s, Loss: 0.02887
        Epoch: 80, Time: 13.39723s, Loss: 0.02884
        Validation: 0.13620756897343958
        Epoch: 81, Time: 13.87684s, Loss: 0.02881
        Epoch: 82, Time: 13.72004s, Loss: 0.02830
        Epoch: 83, Time: 13.52762s, Loss: 0.02796
        Epoch: 84, Time: 13.50852s, Loss: 0.02777
        Epoch: 85, Time: 13.65227s, Loss: 0.02762
        Epoch: 86, Time: 13.84981s, Loss: 0.02752
        Epoch: 87, Time: 14.03578s, Loss: 0.02743
        Epoch: 88, Time: 13.86019s, Loss: 0.02709
        Epoch: 89, Time: 14.47703s, Loss: 0.02670
        Epoch: 90, Time: 13.90316s, Loss: 0.02669
        Epoch: 91, Time: 13.85412s, Loss: 0.02622
        Epoch: 92, Time: 14.55231s, Loss: 0.02636
        Epoch: 93, Time: 14.12314s, Loss: 0.02616
        Epoch: 94, Time: 14.14073s, Loss: 0.02643
        Epoch: 95, Time: 14.76731s, Loss: 0.02528
        Epoch: 96, Time: 13.95123s, Loss: 0.02558
        Epoch: 97, Time: 13.58211s, Loss: 0.02548
        Epoch: 98, Time: 14.17444s, Loss: 0.02538
        Epoch: 99, Time: 14.03820s, Loss: 0.02530
        Epoch: 100, Time: 13.79881s, Loss: 0.02477
        Validation: 0.14007331335739823
        Epoch: 101, Time: 14.41267s, Loss: 0.02501
        Epoch: 102, Time: 13.95937s, Loss: 0.02485
        Epoch: 103, Time: 14.02000s, Loss: 0.02445
        Epoch: 104, Time: 13.91621s, Loss: 0.02418
        Epoch: 105, Time: 13.97738s, Loss: 0.02410
        Epoch: 106, Time: 13.94001s, Loss: 0.02383
        Epoch: 107, Time: 13.96132s, Loss: 0.02386
        Epoch: 108, Time: 13.96773s, Loss: 0.02362
        Epoch: 109, Time: 14.00794s, Loss: 0.02350
        Epoch: 110, Time: 13.80064s, Loss: 0.02343
        Epoch: 111, Time: 14.28152s, Loss: 0.02332
        Epoch: 112, Time: 14.38398s, Loss: 0.02308
        Epoch: 113, Time: 14.34458s, Loss: 0.02345
        Epoch: 114, Time: 14.18515s, Loss: 0.02276
        Epoch: 115, Time: 13.56739s, Loss: 0.02268
        Epoch: 116, Time: 14.22387s, Loss: 0.02314
        Epoch: 117, Time: 14.02960s, Loss: 0.02266
        Epoch: 118, Time: 13.98667s, Loss: 0.02241
        Epoch: 119, Time: 13.81673s, Loss: 0.02238
        Epoch: 120, Time: 13.91288s, Loss: 0.02207
        Validation: 0.14275566576589846
        Epoch: 121, Time: 14.15440s, Loss: 0.02199
        Epoch: 122, Time: 14.28269s, Loss: 0.02178
        Epoch: 123, Time: 14.10793s, Loss: 0.02202
        Epoch: 124, Time: 14.46924s, Loss: 0.02160
        Epoch: 125, Time: 14.01888s, Loss: 0.02190
        Epoch: 126, Time: 14.50532s, Loss: 0.02163
        Epoch: 127, Time: 13.96982s, Loss: 0.02135
        Epoch: 128, Time: 13.80776s, Loss: 0.02115
        Epoch: 129, Time: 13.81826s, Loss: 0.02132
        Epoch: 130, Time: 13.64502s, Loss: 0.02090
        Epoch: 131, Time: 14.08872s, Loss: 0.02094
        Epoch: 132, Time: 13.89601s, Loss: 0.02117
        Epoch: 133, Time: 13.81755s, Loss: 0.02088
        Epoch: 134, Time: 14.06675s, Loss: 0.02075
        Epoch: 135, Time: 14.07287s, Loss: 0.02068
        Epoch: 136, Time: 14.07303s, Loss: 0.02062
        Epoch: 137, Time: 14.07205s, Loss: 0.02035
        Epoch: 138, Time: 13.73393s, Loss: 0.02037
        Epoch: 139, Time: 14.10216s, Loss: 0.02026
        Epoch: 140, Time: 13.71037s, Loss: 0.02014
        Validation: 0.14488457332453364
