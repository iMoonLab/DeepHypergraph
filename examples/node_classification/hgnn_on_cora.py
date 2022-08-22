import torch.nn.functional as F
import torch.optim as optim
import torch

# import sys
# sys.path.append('.')
import dhg
from dhg import Graph, Hypergraph
from dhg.data import Cora, Pubmed, Citeseer
from dhg.models import HGNN, HGNNP, HNHN
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
from copy import deepcopy
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(net, X, G, lbls, train_idx, optimizer, epoch):
    net.train()

    st = time.time()
    optimizer.zero_grad()
    outs = net(X, G)
    outs, lbls = outs[train_idx], lbls[train_idx]
    loss = F.cross_entropy(outs, lbls)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Time: {time.time()-st:.5f}s, Loss: {loss.item():.5f}")
    return loss.item()


@torch.no_grad()
def infer(net, X, G, lbls, idx, test=False):
    net.eval()
    outs = net(X, G)
    outs, lbls = outs[idx], lbls[idx]
    if not test:
        res = evaluator.validate(lbls, outs)
    else:
        res = evaluator.test(lbls, outs)
    return res


if __name__ == "__main__":
    # dhg.random.set_seed(2022)
    evaluator = Evaluator(["accuracy", "f1_score", {"f1_score": {"average": "micro"}}])
    data = Cora()
    # data = Pubmed()
    # data = Citeseer()
    x = data["features"]
    lbl = data["labels"]
    num_v = data["num_vertices"]
    G = Graph(num_v, data["edge_list"])
    HG = Hypergraph.from_graph(G)
    HG.add_hyperedges_from_graph_kHop(G, k=1)
    # HG = Hypergraph.from_graph_kHop(G, k=1)
    # HG.add_hyperedges_from_graph_kHop(G, k=2, only_kHop=True)
    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]

    net = HGNNP(data["dim_features"], 16, data["num_classes"])
    # net = HNHN(data["dim_features"], 16, data["num_classes"], use_bn=True)
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

    x, lbl = x.cuda(), lbl.cuda()
    HG.to(x.device)
    net = net.cuda()

    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(300):
        # train
        train(net, x, HG, lbl, train_mask, optimizer, epoch)
        # validation
        if epoch % 1 == 0:
            with torch.no_grad():
                val_res = infer(net, x, HG, lbl, val_mask)
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
    res = infer(net, x, HG, lbl, test_mask, test=True)
    print(f"final result: epoch: {best_epoch}")
    print(res)
