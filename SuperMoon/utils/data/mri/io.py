import os.path as osp

import SimpleITK as sitk
import matplotlib.pyplot as plt
import torch


def read_mri_series(mri_path):
    img_itk = sitk.ReadImage(mri_path)
    img_np = sitk.GetArrayFromImage(img_itk)
    # SimpleITK read image as (z, y, x), need to be transposed to (x, y, z)
    img_np = img_np.transpose((2, 1, 0)).astype('float')

    return torch.tensor(img_np).float()


def save_mri_series(mris: torch.Tensor, save_dir, name_prefix):
    # mris should be image series with dimension (x, y, z)
    assert len(mris.shape) == 3
    # (x, y, z) -> (z, x, y)
    mris = mris.permute(2, 0, 1)

    # save each mri into a specific direction
    for _idx in range(mris.size(0)):
        _mri = mris[_idx].squeeze().numpy()
        plt.imsave(osp.join(save_dir, f'{name_prefix}_{_idx}.jpg'), _mri, cmap=plt.cm.bone)
