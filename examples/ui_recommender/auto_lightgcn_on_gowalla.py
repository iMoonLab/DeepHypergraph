import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dhg import BiGraph
from dhg.data import MovieLens1M, Gowalla
from dhg.models import LightGCN, NGCF
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
