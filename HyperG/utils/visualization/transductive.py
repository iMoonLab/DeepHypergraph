import os.path as osp

import numpy as np
import torch
from PIL import Image

colormap = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                     [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
                     [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
                     [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
                     [0, 192, 0], [128, 192, 0], [0, 64, 128]], dtype=np.uint8)


def trans_vis_pred_target(pred, target, mask: torch.Tensor, img_size, save_dir, name_prefix):
    mask = mask.bool()
    pred = pred.max(1)[1]
    pred, target = pred[mask].cpu().numpy(), target[mask].cpu().numpy()

    pred_img = colormap[pred].reshape(-1, *img_size, 3)
    target_img = colormap[target].reshape(-1, *img_size, 3)

    n_img = pred_img.shape[0]

    for _idx in range(n_img):
        _img = Image.fromarray(pred_img[_idx])
        _img.save(osp.join(save_dir, f'{name_prefix}_{_idx}_predict.jpg'))
        _img.close()

        _img = Image.fromarray(target_img[_idx])
        _img.save(osp.join(save_dir, f'{name_prefix}_{_idx}_target.jpg'))
        _img.close()
