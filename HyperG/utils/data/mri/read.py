import SimpleITK as sitk
import torch


def read_mri(mri_path):
    img_itk = sitk.ReadImage(mri_path)
    img_np = sitk.GetArrayFromImage(img_itk)
    # SimpleITK read image as (z, y, x), need to be transposed to (x, y, z)
    img_np = img_np.transpose((2, 1, 0)).astype('float')

    return torch.tensor(img_np).float()
