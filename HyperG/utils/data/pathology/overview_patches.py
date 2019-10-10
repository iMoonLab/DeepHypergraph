import numpy as np
import openslide
from PIL import Image

from .sample_patches import get_just_gt_level

SAMPLED = 2
SAMPLED_COLOR = [0, 0, 255]


def draw_patches_on_slide(slide_dir, patch_coors, mini_frac=32):
    slide = openslide.open_slide(slide_dir)
    mini_size = np.ceil(np.array(slide.level_dimensions[0]) / mini_frac).astype(np.int)
    mini_level = get_just_gt_level(slide, mini_size)

    img = slide.read_region((0, 0), mini_level, slide.level_dimensions[mini_level]).convert('RGB')
    img = img.resize(mini_size)

    sampled_mask = gather_sampled_patches(patch_coors, mini_size, mini_frac)
    sampled_patches_img = fuse_img_mask(np.asarray(img), sampled_mask)

    img.close()
    return sampled_patches_img


def gather_sampled_patches(patch_coors, mini_size, mini_frac) -> np.array:
    # generate sampled area mask
    sampled_mask = np.zeros((mini_size[1], mini_size[0]), np.uint8)
    for _coor in patch_coors:
        _mini_coor = (int(_coor[0] / mini_frac), int(_coor[1] / mini_frac))
        _mini_patch_size = (int(_coor[2] / mini_frac), int(_coor[3] / mini_frac))
        sampled_mask[_mini_coor[1]:_mini_coor[1] + _mini_patch_size[1],
        _mini_coor[0]:_mini_coor[0] + _mini_patch_size[0]] = SAMPLED
    sampled_mask = np.asarray(Image.fromarray(sampled_mask).resize(mini_size))

    return sampled_mask


def fuse_img_mask(img: np.array, mask: np.array, alpha=0.7) -> Image:
    assert img.shape[:2] == mask.shape
    img = img.copy()
    if (mask != 0).any():
        img[mask != 0] = alpha * img[mask != 0] + \
                         (1 - alpha) * np.array(SAMPLED_COLOR)
    return Image.fromarray(img)
