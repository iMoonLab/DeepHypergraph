import torch.nn.functional as F
import torch.optim as optim
import torch

# import sys
# sys.path.append('.')
import dhg
from dhg import Hypergraph
from dhg.data import Cooking200
from dhg.models import HGNN, HGNNP
from dhg.metrics import HypergraphVertexClassificationEvaluator as Evaluator
from dhg.random import set_seed
from copy import deepcopy
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
    # set_seed(2022)
    data = Cooking200()
    # data = Pubmed()
    # data = Citeseer()
    # x = data['features']

    lbl = data["labels"]
    num_v = data["labels"].shape[0]
    X = torch.eye(num_v)
    ft_dim = X.shape[1]
    G = Hypergraph(num_v, data["edge_list"])
    train_mask = data["train_mask"]
    val_mask = data["val_mask"]
    test_mask = data["test_mask"]

    # net = FHGNN(ft_dim, 32, lbl.max().item() + 1)
    net = HGNNP(ft_dim, 32, data["num_classes"], use_bn=True)
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)

    X, lbl = X.cuda(), lbl.cuda()
    G.to(X.device)
    net = net.cuda()

    best_state = None
    best_epoch, best_val = 0, 0
    for epoch in range(300):
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
    # test
    print("test...")
    net.load_state_dict(best_state)
    res = infer(net, X, G, lbl, test_mask, test=True)
    print(f"final result: epoch: {best_epoch}")
    print(res)
