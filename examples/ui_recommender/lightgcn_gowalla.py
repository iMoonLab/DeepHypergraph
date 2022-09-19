import time
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dhg import BiGraph
from dhg.data import MovieLens1M, Gowalla
from dhg.models import LightGCN, NGCF
from dhg.nn import BPRLoss, EmbeddingRegularization
from dhg.metrics import UserItemRecommenderEvaluator as Evaluator
from dhg.random import set_seed
from dhg.utils import UserItemDataset, adj_list_to_edge_list


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


if __name__ == "__main__":
    # from dhg.utils import simple_stdout2file
    # simple_stdout2file("/home/fengyifan/lightgcn_gowalla_drop.log")
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
