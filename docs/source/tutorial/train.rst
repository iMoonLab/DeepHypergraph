Training Model
========================

.. hint:: 

    - Author: `Yifan Feng (丰一帆) <https://fengyifan.site/>`_
    - Proof: Xinwei Zhang

Here, we provide two templates for training models.

- :ref:`Training without Batch Data <turorial_training_without_batch_data>`
- :ref:`Training with Batch Data <tutorial_training_with_batch_data>`

.. _turorial_training_without_batch_data:

Training without Batch Data
-----------------------------

This template is suitable for models that do not require batch iteration like vertex classification on graph and hypergraph structures.

.. code-block:: python

    import time
    from copy import deepcopy

    import torch
    import torch.optim as optim
    import torch.nn.functional as F

    from dhg import Graph
    from dhg.data import Cora
    from dhg.models import GCN
    from dhg.random import set_seed
    from dhg.metrics import GraphVertexClassificationEvaluator as Evaluator

    # define your train function
    def train(net, X, A, lbls, train_idx, optimizer, epoch):
        net.train()

        st = time.time()
        optimizer.zero_grad()
        outs = net(X, A)
        outs, lbls = outs[train_idx], lbls[train_idx]
        loss = F.cross_entropy(outs, lbls)
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
        return loss.item()

    # define your validation and testing function
    @torch.no_grad()
    def infer(net, X, A, lbls, idx, test=False):
        net.eval()
        outs = net(X, A)
        outs, lbls = outs[idx], lbls[idx]
        if not test:
            # validation with you evaluator
            res = evaluator.validate(lbls, outs)
        else:
            # testing with you evaluator
            res = evaluator.test(lbls, outs)
        return res


    if __name__ == "__main__":
        set_seed(2022)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # config your evaluation metric here
        evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
        # load Your data here
        data = Cora()
        X, lbl = data["features"], data["labels"]
        # construct your correlation structure here
        G = Graph(data["num_vertices"], data["edge_list"])
        train_mask = data["train_mask"]
        val_mask = data["val_mask"]
        test_mask = data["test_mask"]

        # initialize your model here
        net = GCN(data["dim_features"], 16, data["num_classes"])
        optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

        X, lbl = X.to(device), lbl.to(device)
        G = G.to(device)
        net = net.to(device)

        best_state = None
        best_epoch, best_val = 0, 0
        for epoch in range(200):
            # train
            train(net, X, G, lbl, train_mask, optimizer, epoch)
            # validation
            if epoch % 1 == 0:
                with torch.no_grad():
                    val_res = infer(net, X, G, lbl, val_mask)
                if val_res > best_val:
                    print(f"update best: {val_res:.5f}")
                    best_epoch = epoch
                    best_val = val_res
                    best_state = deepcopy(net.state_dict())
        print("\ntrain finished!")
        print(f"best val: {best_val:.5f}")
        # testing
        print("test...")
        net.load_state_dict(best_state)
        res = infer(net, X, G, lbl, test_mask, test=True)
        print(f"final result: epoch: {best_epoch}")
        print(res)


.. _tutorial_training_with_batch_data:

Training with Batch Data
----------------------------

This template is suitable for models that require batch iteration like item recommender on User-Item bipartite graph structure.

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

    # define your loss function
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

    # define your train function
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

    # define your validation function
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
            # add batch data in the validation phase
            evaluator.validate_add_batch(true_rating, pred_rating)
        # return the result of the validation phase
        return evaluator.validate_epoch_res()

    # define your test function
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
            # add batch data in the testing phase
            evaluator.test_add_batch(true_rating, pred_rating)
        # return the result of the testing phase
        return evaluator.test_epoch_res()


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
        # config your evaluation metric here
        evaluator = Evaluator([{"ndcg": {"k": 20}}, {"recall": {"k": 20}}])

        # load your data here
        data = Gowalla()
        num_u, num_i = data["num_users"], data["num_items"]
        train_adj_list = data["train_adj_list"]
        test_adj_list = data["test_adj_list"]
        # Construct your correlation structure here
        ui_bigraph = BiGraph.from_adj_list(num_u, num_i, train_adj_list)
        ui_bigraph = ui_bigraph.to(device)
        train_edge_list = adj_list_to_edge_list(train_adj_list)
        test_edge_list = adj_list_to_edge_list(test_adj_list)
        # construct your dataset
        train_dataset = UserItemDataset(num_u, num_i, train_edge_list)
        test_dataset = UserItemDataset(num_u, num_i, test_edge_list, train_user_item_list=train_edge_list, phase="test")
        # construct your dataloader
        train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=False, num_workers=num_workers)

        # initialize your model here
        net = LightGCN(num_u, num_i, dim_emb)
        net = net.to(device)
        criterion = BPR_Reg(weight_decay)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        best_state, best_val, best_epoch = None, 0, -1
        for epoch in range(epoch_max):
            # training
            train(net, train_loader, optimizer, criterion, epoch)
            if epoch % val_freq == 0:
                # validation
                val_res = validate(net, test_loader)
                print(f"Validation: NDCG@20 -> {val_res}")
                if val_res > best_val:
                    best_epoch = epoch
                    best_val = val_res
                    best_state = deepcopy(net.state_dict())
        print("train finished")
        print(f"best val: {best_val}")
        print(f"best epoch: {best_epoch}")
        # testing
        print("testing...")
        net.load_state_dict(best_state)
        test_res = test(net, test_loader)
        print(f"test res: {test_res}")

