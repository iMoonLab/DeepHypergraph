import pytest
import numpy as np

import dhg.visualization as vis
import matplotlib.pyplot as plt


@pytest.fixture()
def emb1():
    return np.random.rand(100, 32)


def test_draw_in_euclidean_space(emb1, tmp_path):
    vis.draw_in_euclidean_space(emb1)
    # plt.savefig(tmp_path / "test_draw_in_euclidean_space.png")
    vis.draw_in_euclidean_space(emb1, dim=3)
    # plt.savefig(tmp_path / "test_draw_in_euclidean_space_3d.png")


def test_draw_poincare_ball(emb1, tmp_path):
    vis.draw_in_poincare_ball(emb1)
    # plt.savefig(tmp_path / "test_draw_poincare_ball.png")
    vis.draw_in_poincare_ball(emb1, dim=3)
    # plt.savefig(tmp_path / "test_draw_poincare_ball_3d.png")
