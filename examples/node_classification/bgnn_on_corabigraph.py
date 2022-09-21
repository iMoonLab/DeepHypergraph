from pickletools import optimize
import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dhg import BiGraph
from dhg.data import CoraBiGraph
from dhg.datapipe import min_max_scaler
from dhg.models import BGNN_Adv, BGNN_MLP
from dhg.random import set_seed
from dhg.utils import split_by_ratio
from dhg.metrics import GraphVertexClassificationEvaluator as Evaluator


class LogisticRegression(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = torch.nn.Linear(in_channels, out_channels)

    def forward(self, X):
        return self.linear(X)


def train(net, X, lbls, train_idx, optimizer, epoch):
    net.train()

    st = time.time()
    optimizer.zero_grad()
    outs = net(X)
    outs, lbls = outs[train_idx], lbls[train_idx]
    loss = F.cross_entropy(outs, lbls)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()


@torch.no_grad()
def infer(net, X, lbls, idx, test=False):
    net.eval()
    outs = net(X)
    outs, lbls = outs[idx], lbls[idx]
    if not test:
        res = evaluator.validate(lbls, outs)
    else:
        res = evaluator.test(lbls, outs)
    return res


if __name__ == "__main__":
    set_seed(2022)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data = CoraBiGraph()
    X_u, X_v, = torch.tensor(data.raw("u_features")).float(), torch.tensor(data.raw("v_features")).float()
    X_u, X_v = min_max_scaler(X_u, -1, 1), min_max_scaler(X_v, -1, 1)
    X_u, X_v = X_u.to(device), X_v.to(device)
    u_lbl = data["u_labels"]
    g = BiGraph(data["num_u_vertices"], data["num_v_vertices"], data["edge_list"])
    # train embedding
    # -----------------------------------------------------------------------------------------
    net = BGNN_Adv(data["dim_u_features"], data["dim_v_features"], layer_depth=3)
    X_u_emb = net.train_with_cascaded(X_u, X_v, g, 0.00001, 5e-4, 2, drop_rate=0.5, device="cuda:0")
    # -----------------------------------------------------------------------------------------
    # net = BGNN_MLP(data["dim_u_features"], data["dim_v_features"], 48, 24, layer_depth=3)
    # X_u_emb = net.train_with_cascaded(X_u, X_v, g, 0.00001, 5e-4, 2, device="cuda:0")
    # -----------------------------------------------------------------------------------------

    # Logistic Regression
    X, lbl = X_u_emb, u_lbl
    N, C = X.shape
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
    train_mask, val_mask, test_mask = split_by_ratio(N, v_label=lbl, train_ratio=0.8, val_ratio=0.06)
    reg_model = LogisticRegression(C, data["num_u_classes"]).to(device)
    optimizer = optim.Adam(reg_model.parameters(), lr=0.01, weight_decay=5e-4)

    train_mask, val_mask, test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)
    X, lbl = X.to(device), lbl.to(device)
    reg_model = reg_model.to(device)

    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(200):
        # train
        train(reg_model, X, lbl, train_mask, optimizer, epoch)
        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = infer(reg_model, X, lbl, val_mask)
            if val_res > best_val:
                print(f"update best: {val_res:.5f}")
                best_epoch = epoch
                best_val = val_res
                best_state = deepcopy(reg_model.state_dict())
    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")
    # test
    print("test...")
    reg_model.load_state_dict(best_state)
    res = infer(reg_model, X, lbl, test_mask, test=True)
    print(f"final result: epoch: {best_epoch}")
    print(res)
