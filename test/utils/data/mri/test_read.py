import pytest

from HyperG.utils.data import read_mri


@pytest.mark.skip(reason='unpleasure')
def test_read_mri():
    img_dir = '/repository/HyperG_example/example_data/heart_mri/processed/0001.mha'
    img = read_mri(img_dir)

    # print(img.shape)
    # import matplotlib.pyplot as plt
    # img = img.squeeze().numpy()
    # plt.imshow(img, cmap=plt.cm.bone)
    # plt.show()
