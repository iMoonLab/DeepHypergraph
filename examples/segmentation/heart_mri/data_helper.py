import glob
import os
import os.path as osp
import pickle
import re

import torch

from SuperMoon.hyedge import gather_patch_ft, neighbor_grid, neighbor_distance
from SuperMoon.hygraph import hyedge_concat
from SuperMoon.utils.data import split_id
from SuperMoon.utils.data.mri import read_mri_series


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def split_train_val(data_root, ratio=0.8, save_split_dir=None, resplit=False):
    if not resplit and save_split_dir is not None and osp.exists(save_split_dir):
        with open(save_split_dir, 'rb') as f:
            result = pickle.load(f)
        return result

    all_list = glob.glob(osp.join(data_root, '*.mha'))

    all_dict = {}
    for full_dir in all_list:
        file_name = osp.split(full_dir)[1]
        id = re.split('\_|\.', file_name)[0]
        all_dict.setdefault(id, {})
        if 'seg' in file_name:
            all_dict[id]['seg_dir'] = full_dir
        else:
            all_dict[id]['img_dir'] = full_dir

    id_list = list(all_dict.keys())
    train_list, val_list = split_id(id_list, ratio)

    train_list = [all_dict[_id] for _id in train_list]
    val_list = [all_dict[_id] for _id in val_list]

    result = {'train': train_list, 'val': val_list}
    if save_split_dir is not None:
        save_folder = osp.split(save_split_dir)[0]
        if not osp.exists(save_folder):
            os.makedirs(save_folder)
        with open(save_split_dir, 'wb') as f:
            pickle.dump(result, f)

    return result


def preprocess(data_list, patch_size, k_nearest):
    train_list, val_list = data_list['train'], data_list['val']
    x, H_grid, lbl, mask_train = [], [], [], []

    for _item in train_list:
        _x, _H, _lbl, img_size = process_mri_seg(_item, patch_size)
        _node_num = _x.size(0)
        x.append(_x)
        H_grid.append(_H)
        lbl.append(_lbl)
        mask_train.extend([1] * _node_num)

    for _item in val_list:
        _x, _H, _lbl, img_size = process_mri_seg(_item, patch_size)
        _node_num = _x.size(0)
        x.append(_x)
        H_grid.append(_H)
        lbl.append(_lbl)
        mask_train.extend([0] * _node_num)

    x, lbl = torch.cat(x, dim=0), torch.cat(lbl, dim=0).long()
    x = normalize(x)

    H_grid = hyedge_concat(H_grid, same_node=False)
    mask_train = torch.tensor(mask_train).bool()
    mask_val = ~mask_train

    H_global = neighbor_distance(x, k_nearest)

    H = hyedge_concat([H_grid, H_global])

    return x, H, lbl, mask_train, mask_val, img_size


def process_mri_seg(data, patch_size):
    # M x N x C -> C x M x N -> 1 x C x M x N
    img = read_mri_series(data['img_dir']).permute(2, 0, 1).unsqueeze(0)
    row_num, col_num = img.size(2), img.size(3)
    # M x N x 1 -> M x N
    lbl = read_mri_series(data['seg_dir']).squeeze()

    # 1 x C x M x N -> 1 x Ckk x M x N -> M x N x Ckk
    img_patched = gather_patch_ft(img, patch_size).permute(2, 3, 1, 0).squeeze()

    # M x N x Ckk -> MN x Ckk
    img_patched = img_patched.view(-1, img_patched.size(2))
    # M x N -> MN
    lbl = lbl.view(-1)

    grid_H = neighbor_grid((row_num, col_num), self_loop=True)
    return img_patched, grid_H, lbl, (row_num, col_num)
