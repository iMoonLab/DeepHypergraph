from HyperG.utils.data import read_mri

show_img = False


def test_read_mri():
    img_dir = '/Users/fengyifan/Documents/Tsinghua/Other_Code/dcm_seg_hgnn_patch_code/data/ori/bai_de_yu_0001.mha'
    img = read_mri(img_dir)

    # print(img.shape)
    # import matplotlib.pyplot as plt
    # img = img.squeeze().numpy()
    # plt.imshow(img, cmap=plt.cm.bone)
    # plt.show()
