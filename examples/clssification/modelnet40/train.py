import os.path as osp

import torch
import torch.nn.functional as F
from data_helper import load_ft

from SuperMoon.hyedge import neighbor_distance
from SuperMoon.hygraph import hyedge_concat
from SuperMoon.models import HGNN
from SuperMoon.utils.meter import trans_class_acc

# initialize parameters
data_root = '/repository/HyperG_example/example_data/modelnet40/processed'
result_root = '/repository/HyperG_example/tmp/modelnet40'

k_nearest = 10
feature_dir = osp.join(data_root, 'ModelNet40_mvcnn_gvcnn.mat')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data
gvcnn_ft, target, mask_train, mask_val = load_ft(feature_dir, feature_name='GVCNN')
mvcnn_ft, _, _, _ = load_ft(feature_dir, feature_name='MVCNN')

# init H and X
gvcnn_H = neighbor_distance(gvcnn_ft, k_nearest)
mvcnn_H = neighbor_distance(mvcnn_ft, k_nearest)

ft = torch.cat([mvcnn_ft, gvcnn_ft], dim=1)
H = hyedge_concat([mvcnn_H, gvcnn_H])

x_ch = ft.size(1)
n_class = target.max().item() + 1
model = HGNN(x_ch, n_class, hiddens=[128])

model, ft, H, target, mask_train, mask_val = model.to(device), ft.to(device), H.to(device), \
                                             target.to(device), mask_train.to(device), mask_val.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    optimizer.zero_grad()
    pred = model(ft, H)
    F.nll_loss(pred[mask_train], target[mask_train]).backward()
    optimizer.step()


def val():
    model.eval()
    pred = model(ft, H)

    _train_acc = trans_class_acc(pred, target, mask_train)
    _val_acc = trans_class_acc(pred, target, mask_val)

    return _train_acc, _val_acc


if __name__ == '__main__':
    best_acc, best_iou = 0.0, 0.0
    for epoch in range(1, 101):
        train()
        train_acc, val_acc = val()
        if val_acc > best_acc:
            best_acc = val_acc
        print(f'Epoch: {epoch}, Train:{train_acc:.4f}, Val:{val_acc:.4f}, '
              f'Best Val acc:{best_acc:.4f}')
