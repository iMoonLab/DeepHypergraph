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
    embeddings: np.ndarray, label: Optional[np.ndarray] = None, dim: int = 2, cmap="viridis"
) -> plt.figure:
    r"""Visualize embeddings in Eulidean Space with t-SNE algorithm.

    Args:
        ``feature`` (``np.ndarray``): The feature matrix. Size :math:`(N, C)`.
        ``label`` (``np.ndarray``, optional): The label matrix. Size :math:`(N, )`.
        ``dim`` (``int``): Project the embedding into ``dim``-dimensional space, which is ``2`` or ``3``. Defaults to ``2``.
        ``cmap`` (``str``, optional): The `color map <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_. Defaults to ``"viridis"``.
    """
    assert dim in [2, 3], "dim must be 2 or 3."
    tsne = TSNE(n_components=dim, init="pca")
    emb_low = tsne.fit_transform(embeddings)
    if dim == 2:
        plot_2d_embedding(emb_low, label, cmap=cmap)
    elif dim == 3:
        plot_3d_embedding(emb_low, label, cmap=cmap)
    else:
        raise ValueError("dim must be 2 or 3.")


def draw_in_poincare_ball(
    embeddings: np.ndarray,
    label: Optional[np.ndarray] = None,
    dim: int = 2,
    reduce_method: str = "pca",
    cmap="viridis",
) -> plt.figure:
    r"""Visualize embeddings in Poincare Ball.

    Args:
        ``feature`` (``np.ndarray``): The feature matrix. Size :math:`(N, C)`.
        ``label`` (``np.ndarray``, optional): The label matrix. Size :math:`(N, )`. Defaults to ``None``.
        ``dim`` (``int``): Project the embedding into ``dim``-dimensional space, which is ``2`` or ``3``. Defaults to ``2``.
        ``reduce_method`` (``str``): The method to project the embedding into low-dimensional space. It can be ``pca`` or ``tsne``. Defaults to ``pca``.
        ``cmap`` (``str``, optional): The `color map <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_. Defaults to ``"viridis"``.
    """
    emb_low = project_to_poincare_ball(embeddings, dim, reduce_method)
    if dim == 2:
        plot_2d_embedding(emb_low, label, cmap=cmap)
    elif dim == 3:
        plot_3d_embedding(emb_low, label, cmap=cmap)
    else:
        raise ValueError("dim must be 2 or 3.")


def animation_of_3d_poincare_ball(
    embeddings: np.ndarray, label: Optional[np.ndarray] = None, reduce_method: str = "pca", cmap="viridis"
):
    r"""Make 3D animation of embeddings visualization on Poincare Ball. 
    This function will return the animation object `ani <https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html>`_.
    You can save the animation by ``ani.save("animation.gif")``.

    Args:
        ``feature`` (``np.ndarray``): The feature matrix. Size :math:`(N, C)`.
        ``label`` (``np.ndarray``, optional): The label matrix. Size :math:`(N, )`. Defaults to ``None``.
        ``reduce_method`` (``str``): The method to project the embedding into low-dimensional space. It can be ``pca`` or ``tsne``. Defaults to ``pca``.
        ``cmap`` (``str``, optional): The `color map <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_. Defaults to ``"viridis"``.
    
    Example:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from dhg.visualization import animation_of_3d_poincare_ball
        >>> x = np.random.rand(100, 32)
        >>> ani = animation_of_3d_poincare_ball(x)
        >>> plt.show()
        >>> ani.save('a.gif')
    """
    emb_low = project_to_poincare_ball(embeddings, 3, reduce_method)
    colors = label if label is not None else "r"
    return make_animation(emb_low, colors, cmap=cmap)


def animation_of_3d_euclidean_space(
    embeddings: np.ndarray, label: Optional[np.ndarray] = None, cmap="viridis",
):
    r"""Make 3D animation of embeddings visualization of tSNE algorithm.
    This function will return the animation object `ani <https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html>`_.
    You can save the animation by ``ani.save("animation.gif")``.

    Args:
        ``feature`` (``np.ndarray``): The feature matrix. Size :math:`(N, C)`.
        ``label`` (``np.ndarray``, optional): The label matrix. Size :math:`(N, )`. Defaults to ``None``.
        ``cmap`` (``str``, optional): The `color map <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_. Defaults to ``"viridis"``.
    
    Example:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from dhg.visualization import animation_of_3d_euclidean_space
        >>> x = np.random.rand(100, 32)
        >>> ani = animation_of_3d_euclidean_space(x)
        >>> plt.show()
        >>> ani.save('a.gif')
    """
    tsne = TSNE(n_components=3, init="pca")
    emb_low = tsne.fit_transform(embeddings)
    colors = label if label is not None else "r"
    return make_animation(emb_low, colors, cmap=cmap)

