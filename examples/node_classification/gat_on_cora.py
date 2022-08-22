import time
from copy import deepcopy

# import sys
# sys.path.append('.')
from dhg.models import GAT
from dhg.data import Cora, Pubmed, Citeseer
from dhg import Graph
from dhg.metrics import GraphVertexClassificationEvaluator as Evaluator
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(net, X, A, lbls, train_idx, optimizer, epoch):
    net.train()

    st = time.time()
    optimizer.zero_grad()
    outs = net(X, A)
    outs, lbls = outs[train_idx], lbls[train_idx]
    loss = F.cross_entropy(outs, lbls)
    # loss = F.nll_loss(outs, lbls)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()


@torch.no_grad()
def infer(net, X, A, lbls, idx, test=False):
    net.eval()
    outs = net(X, A)
    outs, lbls = outs[idx], lbls[idx]
    if not test:
        res = evaluator.validate(lbls, outs)
    else:
        res = evaluator.test(lbls, outs)
    return res


if __name__ == "__main__":
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
    data = Cora()
    # data = Pubmed()
    # data = Citeseer()
    x = data["features"]
    lbl = data["labels"]
    num_v = data["features"].shape[0]
    x_dim = data["features"].shape[1]
    G = Graph(num_v, data["edge_list"], extra_selfloop=True)
    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]

    net = GAT(x_dim, 8, lbl.max().item() + 1, num_heads=8, drop_rate=0.6)
    optimizer = optim.Adam(net.parameters(), lr=0.005, weight_decay=5e-4)

    x, lbl = x.cuda(), lbl.cuda()
    G.to(x.device)
    net = net.cuda()

    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(300):
        # train
        train(net, x, G, lbl, train_mask, optimizer, epoch)
        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = infer(net, x, G, lbl, val_mask)
            if val_res > best_val:
                print(f"update best: {val_res:.5f}")
                best_epoch = epoch
                best_val = val_res
                best_state = deepcopy(net.state_dict())
    print("\ntrain finished!")
    print(f"best val: {best_val:.5f}")
    # test
    print("test...")
    net.load_state_dict(best_state)
    res = infer(net, x, G, lbl, test_mask)
    print(f"final result: epoch: {best_epoch}")
    print(res)
