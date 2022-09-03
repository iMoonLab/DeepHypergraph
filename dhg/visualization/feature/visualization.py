from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from .utils import (
    plot_2d_embedding,
    plot_3d_embedding,
    project_to_poincare_ball,
    make_animation,
)


def draw_in_euclidean_space(
    embeddings: np.ndarray, label: Optional[np.ndarray] = None, dim: int = 2
) -> plt.figure:
    r"""Visualize embeddings in Eulidean Space with t-SNE algorithm. Return an instance of ``plt.figure``.

    Args:
        ``feature`` (``np.ndarray``): The feature matrix. Size :math:`(N, C)`.
        ``label`` (``np.ndarray``, optional): The label matrix. Size :math:`(N, )`.
        ``dim`` (``int``): Project the embedding into ``dim``-dimensional space, which is ``2`` or ``3``. Defaults to ``2``.
    """
    assert dim in [2, 3], "dim must be 2 or 3."
    tsne = TSNE(n_components=dim, init="pca")
    emb_low = tsne.fit_transform(embeddings)
    if dim == 2:
        fig = plot_2d_embedding(emb_low, label)
    elif dim == 3:
        fig = plot_3d_embedding(emb_low, label)
    else:
        raise ValueError("dim must be 2 or 3.")
    return fig


def draw_in_poincare_ball(
    embeddings: np.ndarray,
    label: Optional[np.ndarray] = None,
    dim: int = 2,
    reduce_method: str = "pca",
) -> plt.figure:
    r"""Visualize embeddings in Poincare Ball. Return an instance of ``plt.figure``.

    Args:
        ``feature`` (``np.ndarray``): The feature matrix. Size :math:`(N, C)`.
        ``label`` (``np.ndarray``, optional): The label matrix. Size :math:`(N, )`. Defaults to ``None``.
        ``dim`` (``int``): Project the embedding into ``dim``-dimensional space, which is ``2`` or ``3``. Defaults to ``2``.
        ``reduce_method`` (``str``): The method to project the embedding into low-dimensional space. It can be ``pca`` or ``tsne``. Defaults to ``pca``.
    """
    emb_low = project_to_poincare_ball(embeddings, dim, reduce_method)
    if dim == 2:
        fig = plot_2d_embedding(emb_low, label)
    elif dim == 3:
        fig = plot_3d_embedding(emb_low, label)
    else:
        raise ValueError("dim must be 2 or 3.")
    return fig


def animation_of_3d_poincare_ball(
    embeddings: np.ndarray,
    label: Optional[np.ndarray] = None,
    reduce_method: str = "pca",
):
    r"""Play animation of embeddings visualization on Poincare Ball.

    Args:
        ``feature`` (``np.ndarray``): The feature matrix. Size :math:`(N, C)`.
        ``label`` (``np.ndarray``, optional): The label matrix. Size :math:`(N, )`. Defaults to ``None``.
        ``reduce_method`` (``str``): The method to project the embedding into low-dimensional space. It can be ``pca`` or ``tsne``. Defaults to ``pca``.
    """
    emb_low = project_to_poincare_ball(embeddings, 3, reduce_method)
    colors = label if label is not None else "b"
    make_animation(emb_low, colors)


def animation_of_3d_euclidean_space(
    embeddings: np.ndarray, label: Optional[np.ndarray] = None,
):
    r"""Play animation of embeddings visualization of tSNE algorithm.

    Args:
        ``feature`` (``np.ndarray``): The feature matrix. Size :math:`(N, C)`.
        ``label`` (``np.ndarray``, optional): The label matrix. Size :math:`(N, )`. Defaults to ``None``.
    """
    tsne = TSNE(n_components=3, init="pca")
    emb_low = tsne.fit_transform(embeddings)
    colors = label if label is not None else "b"
    make_animation(emb_low, colors)


if __name__ == "__main__":
    file_dir = "data/modelnet40/train_img_feat_4.npy"
    # save_dir = "./tmp/figure"
    save_dir = None  # None for show now or file name to save
    low_demen_method = "TSNE"  # vis for poincare_ball, PCA or TSNE
    show_method = "Rotation"  # None for 2d or Rotation and Drag for 3d
    label = np.load("data/modelnet40/train_label.npy")
    ft = np.load(file_dir)
    d = 3
    # vis_tsne(ft, save_dir,d)
    draw_in_poincare_ball(
        ft, save_dir, d, label, reduce_method=low_demen_method, auto_play=show_method
    )
