from typing import Optional, Union

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

eps = 1e-5
min_norm = 1e-15


def make_animation(embeddings: np.ndarray, colors: Union[np.ndarray, str]):
    r"""Make an animation of embeddings.

    Args:
        ``embeddings`` (``np.ndarray``): The embedding matrix. Size :math:`(N, 3)`. 
        ``colors`` (``Union[np.ndarray, str]``): The color matrix. ``str`` or Size :math:`(N, )`. 
    """
    x, y, z = embeddings[:, 0], embeddings[:, 1], embeddings[:, 2]
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    plt.ion()
    for i in range(30000):
        # Clear the previously drawn picture
        plt.clf()
        # Get the current picture
        fig = plt.gcf()
        # Get the current axis
        ax = fig.gca(projection="3d")
        if colors is not None:
            ax.scatter(x, y, z, c=colors, cmap="viridis")
        else:
            ax.scatter(x, y, z, cmap="viridis")
        # Elevation angle Azimuth angle
        ax.view_init(elev=20, azim=i % 360)
        # Pause for a period of time
        plt.pause(0.001)
        # Close the drawing window


def plot_2d_embedding(embeddings: np.ndarray, label: Optional[np.ndarray] = None):
    r"""Plot the embedding in 2D.
    
    Args:
        ``embeddings`` (``np.ndarray``): The embedding matrix. Size :math:`(N, 2)`.
        ``label`` (``np.ndarray``, optional): The label matrix.
    """
    fig = plt.figure()
    if label is not None:
        plt.scatter(embeddings[:, 0], embeddings[:, 1], c=label, cmap="viridis")
    else:
        plt.scatter(embeddings[:, 0], embeddings[:, 1], cmap="viridis")
    return fig


def plot_3d_embedding(embeddings: np.ndarray, label: Optional[np.ndarray] = None):
    r"""Plot the embedding in 3D.
    
    Args:
        ``embeddings`` (``np.ndarray``): The embedding matrix. Size :math:`(N, 3)`.
        ``label`` (``np.ndarray``, optional): The label matrix.
    """
    x, y, z = embeddings[:, 0], embeddings[:, 1], embeddings[:, 2]
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    if label is not None:
        ax.scatter(x, y, z, c=label, cmap="viridis")
    else:
        ax.scatter(x, y, z, cmap="viridis")
    return fig


# for poincare_ball
def tanh(x, clamp=15):
    r"""Calculate the tanh value of the matrix x.
    
    Args:
        ``x`` (``np.ndarray``): The feature matrix. Size :math:`(N, C)`.
        ``clap`` (``int``): Boundary value.
    """
    return np.tanh((np.clip(x, -clamp, clamp)))


def proj(x, c):
    r"""Regulation of feature in Hyperbolic space.

    Args:
        ``x`` (``np.ndarray``): The feature matrix. Size :math:`(N, C)`.
        ``c`` (``int``): Curvature of Hyperbolic space.
    """
    norm = np.clip(LA.norm(x, axis=-1, keepdims=True), a_min=min_norm, a_max=None)
    maxnorm = (1 - eps) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return np.where(cond, projected, x)


def expmap0(u, c):
    r"""Map feature from Euclidean space to Hyperbolic space with curvature of c, taking the origin as a reference point.
    Args:
        ``u`` (``np.ndarray``): The feature matrix. Size :math:`(N, C)`.
        ``c`` (``int``): Curvature of Hyperbolic space.
    """
    sqrt_c = c ** 0.5
    u_norm = np.clip(LA.norm(u, axis=-1, keepdims=True), a_min=min_norm, a_max=None)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return gamma_1


def proj_tan0(u, c):
    r"""Regulation of feature in Euclidean space.
    Args:
        ``u`` (``np.ndarray``): The feature matrix. Size :math:`(N, C)`.
        ``c`` (``int``): Curvature of Hyperbolic space.
    """
    return u


def logmap0(p, c):
    r"""Map feature from Hyperbolic space to Euclidean space with curvature of c, taking the origin as a reference point.
    Args:
        ``p`` (``np.ndarray``): The feature matrix. Size :math:`(N, C)`.
        ``c`` (``int``): Curvature ofHyperbolic space.
    """
    sqrt_c = c ** 0.5
    p_norm = np.clip(LA.norm(p, axis=-1, keepdims=True), a_min=min_norm, a_max=None)
    scale = 1.0 / sqrt_c * np.arctanh(sqrt_c * p_norm) / p_norm
    return scale * p


def project_to_poincare_ball(
    embeddings: np.ndarray, dim: int = 2, reduce_method: str = "pca"
) -> np.ndarray:
    r"""Project embeddings from Euclidean space to Hyperbolic space.

    Args:
        ``feature`` (``np.ndarray``): The feature matrix. Size :math:`(N, C)`.
        ``dim`` (``int``): Project the embedding into ``dim``-dimensional space, which is ``2`` or ``3``. Defaults to ``2``.
        ``reduce_method`` (``str``): The method to project the embedding into low-dimensional space. It can be ``pca`` or ``tsne``. Defaults to ``pca``.
    """
    assert dim in [2, 3], "dim must be 2 or 3."
    assert reduce_method in ["pca", "tsne"], "reduce_method must be pca or tsne."
    # Curvature
    c = 2.0
    embeddings = embeddings / LA.norm(embeddings, axis=1, keepdims=True)
    o = np.zeros_like(embeddings)
    embeddings = np.concatenate([o[:, 0:1], embeddings], axis=1)
    # H encoder Pre-stage
    x_hyp = proj(expmap0(proj_tan0(embeddings, c), c=c), c=c)
    x_tangent = logmap0(x_hyp, c=c)
    if reduce_method == "tsne":
        tsne = TSNE(n_components=dim, init="pca")
        emb_low = tsne.fit_transform(x_tangent)
    elif reduce_method == "pca":
        pca = PCA(n_components=dim)
        emb_low = pca.fit_transform(x_tangent)
    else:
        raise ValueError("reduce_method must be pca or tsne.")
    x_min, x_max = np.min(emb_low, 0), np.max(emb_low, 0)
    # Normalisation
    emb_low = (emb_low - x_min) / (x_max - x_min)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-2, 2))
    emb_low = min_max_scaler.fit_transform(emb_low)
    # Important step
    emb_low = expmap0(emb_low, c=c)
    # Based on the result of previous step, Regularisation
    emb_low = proj(emb_low, c)
    return emb_low

