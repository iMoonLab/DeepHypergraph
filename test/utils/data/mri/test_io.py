import pytest

from SuperMoon.utils.data.mri import read_mri_series


@pytest.mark.skip(reason='unpleasure')
def test_read_mri():
    for _img_name in ['0001', '0002', '0003', '0004', '0005']:
        img_dir = f'/repository/HyperG_example/example_data/heart_mri/processed/{_img_name}.mha'
        save_dir = f'/repository/HyperG_example/tmp/heart_mri/{_img_name}.jpg'
        img = read_mri_series(img_dir)

        print(img.shape)
        import matplotlib.pyplot as plt
        img = img.squeeze().numpy()
        plt.imshow(img, cmap=plt.cm.bone)
        plt.imsave(save_dir, img, cmap=plt.cm.bone)
        # plt.show()
