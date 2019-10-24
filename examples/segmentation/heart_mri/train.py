import os.path as osp

import torch
import torch.nn.functional as F
from data_helper import preprocess, split_train_val

from SuperMoon.models import HGNN
from SuperMoon.utils import check_dir
from SuperMoon.utils.meter import trans_class_acc, trans_iou_socre
from SuperMoon.utils.visualization import trans_vis_pred_target

# initialize parameters
data_root = '/repository/HyperG_example/example_data/heart_mri/processed'
result_root = '/repository/HyperG_example/tmp/heart_mri'
k_nearest = 7
patch_size = (5, 5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
split_dir = osp.join(result_root, 'split.pkl')
vis_dir = osp.join(result_root, 'vis')

# check directions
assert check_dir(data_root, make=False)
check_dir(result_root)
check_dir(vis_dir)

data_dict = split_train_val(data_root, ratio=0.8, save_split_dir=split_dir, resplit=True)
x, H, target, mask_train, mask_val, img_size = preprocess(data_dict, patch_size, k_nearest)

x_ch = x.size(1)
n_class = target.max().item() + 1
model = HGNN(x_ch, n_class, hiddens=[16])

model, x, H, target, mask_train, mask_val = model.to(device), x.to(device), H.to(device), \
                                            target.to(device), mask_train.to(device), mask_val.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    pred = model(x, H)
    F.nll_loss(pred[mask_train], target[mask_train]).backward()
    optimizer.step()


def val():
    model.eval()
    pred = model(x, H)

    _train_acc = trans_class_acc(pred, target, mask_train)
    _val_acc = trans_class_acc(pred, target, mask_val)

    _train_iou = trans_iou_socre(pred, target, mask_train)
    _val_iou = trans_iou_socre(pred, target, mask_val)

    return _train_acc, _val_acc, _train_iou, _val_iou


def vis(prefix):
    model.eval()
    pred = model(x, H)
    trans_vis_pred_target(pred, target, mask_train, img_size, vis_dir, f'{prefix}_train')
    trans_vis_pred_target(pred, target, mask_val, img_size, vis_dir, f'{prefix}_val')


if __name__ == '__main__':
    best_acc, best_iou = 0.0, 0.0
    for epoch in range(1, 2001):
        train()
        train_acc, val_acc, train_iou, val_iou = val()
        if val_acc > best_acc:
            best_acc = val_acc
        if val_iou[0] > best_iou:
            best_iou = val_iou[0]
            vis(f'epoch_{epoch}')
        print(f'Epoch: {epoch}, Train:{train_acc:.4f}, Val:{val_acc:.4f}, '
              f'Best Val acc:{best_acc:.4f}, Best Val iou:{best_iou:.4f}')
