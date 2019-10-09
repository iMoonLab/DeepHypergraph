import os.path as osp

import openslide


def sample_patches(slide_dir):
    slide = openslide.open_slide(slide_dir)
    slide_name = osp.basename(slide_dir)
    slide_name = slide_name[:slide_name.rfind('.')]
