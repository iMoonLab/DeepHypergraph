import glob
import json
import os
import os.path as osp
import pickle

import numpy as np
import torch
from extract_patch_feature import extract_ft
from torch.utils.data import Dataset, DataLoader

from SuperMoon.utils.data import split_id
from SuperMoon.utils.data.pathology import sample_patch_coors, draw_patches_on_slide


def split_train_val(data_root, ratio=0.8, save_split_dir=None, resplit=True):
    if not resplit and save_split_dir is not None and osp.exists(save_split_dir):
        with open(save_split_dir, 'rb') as f:
            result = pickle.load(f)
        return result

    all_list = glob.glob(osp.join(data_root, '*.svs'))
    with open(osp.join(data_root, 'opti_survival.json'), 'r') as fp:
        lbls = json.load(fp)

    all_dict = {}
    survival_time_max = 0
    for full_dir in all_list:
        _id = get_id(full_dir)
        all_dict[_id] = {}
        st = int(lbls[_id])
        all_dict[_id]['img_dir'] = full_dir
        all_dict[_id]['survival_time'] = st
        survival_time_max = survival_time_max \
            if survival_time_max > st else st

    id_list = list(all_dict.keys())
    train_list, val_list = split_id(id_list, ratio)

    result = {'survival_time_max': survival_time_max,
              'train': {},
              'val': {}}
    for _id in train_list:
        result['train'][_id] = all_dict[_id]
    for _id in val_list:
        result['val'][_id] = all_dict[_id]

    if save_split_dir is not None:
        save_folder = osp.split(save_split_dir)[0]
        if not osp.exists(save_folder):
            os.makedirs(save_folder)
        with open(save_split_dir, 'wb') as f:
            pickle.dump(result, f)

    return result


def preprocess(data_dict, patch_ft_dir, patch_coors_dir, num_sample=2000,
               patch_size=256, sampled_vis=None, mini_frac=32):
    # check if each slide patch feature exists
    all_dir_list = []
    for phase in ['train', 'val']:
        for _id in data_dict[phase].keys():
            all_dir_list.append(data_dict[phase][_id]['img_dir'])
    to_do_list = check_patch_ft(all_dir_list, patch_ft_dir)

    if to_do_list is not None:
        for _idx, _dir in enumerate(to_do_list):
            print(f'{_idx + 1}/{len(to_do_list)}: processing slide {_dir}...')

            print(f'sampling patch...')
            _id = get_id(_dir)
            _patch_coors = sample_patch_coors(_dir, num_sample=2000, patch_size=256)

            # save sampled patch coordinates
            with open(osp.join(patch_coors_dir, f'{_id}_coors.pkl'), 'wb') as fp:
                pickle.dump(_patch_coors, fp)

            # visualize sampled patches on slide
            if sampled_vis is not None:
                _vis_img_dir = osp.join(sampled_vis, f'{_id}_sampled_patches.jpg')
                print(f'saving sampled patch_slide visualization {_vis_img_dir}...')
                _vis_img = draw_patches_on_slide(_dir, _patch_coors, mini_frac=32)
                with open(_vis_img_dir, 'w') as fp:
                    _vis_img.save(fp)

            # extract patch feature for each slide
            print(f'extracting feature...')
            fts = extract_ft(_dir, _patch_coors, depth=34, batch_size=512)
            np.save(osp.join(patch_ft_dir, f'{_id}_fts.npy'), fts.cpu().numpy())


def get_dataloaders(data_dict, patch_ft_dir):
    all_ft_list = glob.glob(osp.join(patch_ft_dir, '*_fts.npy'))

    ft_dict = {}
    for _dir in all_ft_list:
        ft_dict[get_id(_dir)] = _dir

    SP_datasets = {phase: SlidePatch(data_dict[phase], ft_dict, data_dict['survival_time_max'])
                   for phase in ['train', 'val']}
    SP_dataloaders = {phase: DataLoader(SP_datasets[phase], batch_size=1,
                                        shuffle=True, num_workers=4)
                      for phase in ['train', 'val']}
    dataset_size = {phase: len(SP_datasets[phase]) for phase in ['train', 'val']}
    len_ft = SP_datasets['train'][0][0].size(1)
    return SP_dataloaders, dataset_size, len_ft


class SlidePatch(Dataset):

    def __init__(self, data_dict: dict, ft_dict, survival_time_max):
        super().__init__()
        self.st_max = float(survival_time_max)
        self.id_list = list(data_dict.keys())
        self.data_dict = data_dict
        self.ft_dict = ft_dict

    def __getitem__(self, idx: int):
        id = self.id_list[idx]
        fts = torch.tensor(np.load(self.ft_dict[id])).float()
        st = torch.tensor(self.data_dict[id]['survival_time']).float()
        return fts, st / self.st_max

    def __len__(self) -> int:
        return len(self.id_list)


def check_patch_ft(dir_list, patch_ft_dir):
    to_do_list = []
    done_list = glob.glob(osp.join(patch_ft_dir, '*_fts.npy'))
    done_list = [get_id(_dir) for _dir in done_list]
    for _dir in dir_list:
        id = get_id(_dir)
        if id not in done_list:
            to_do_list.append(_dir)
    return to_do_list


def get_id(_dir):
    return osp.splitext(osp.split(_dir)[1])[0].split('_')[0]
