from typing import Union, List, Callable, Any
import torch
import numpy as np
import scipy.sparse


def to_tensor(
    X: Union[list, np.ndarray, torch.Tensor, scipy.sparse.csr_matrix]
) -> torch.Tensor:
    r"""Convert ``List``, ``numpy.ndarray``, ``scipy.sparse.csr_matrix`` to ``torch.Tensor``.

    Args:
        ``X`` (``Union[List, np.ndarray, torch.Tensor, scipy.sparse.csr_matrix]``): Input.
    """
    if isinstance(X, list):
        X = torch.tensor(X)
    elif isinstance(X, scipy.sparse.csr_matrix):
        X = X.todense()
        X = torch.tensor(X)
    elif isinstance(X, np.ndarray):
        X = torch.tensor(X)
    else:
        X = torch.tensor(X)
    return X.float()


def to_bool_tensor(X: Union[List, np.ndarray, torch.Tensor]) -> torch.BoolTensor:
    r"""Convert ``List``, ``numpy.ndarray``, ``torch.Tensor`` to ``torch.BoolTensor``.

    Args:
        ``X`` (``Union[List, np.ndarray, torch.Tensor]``): Input.
    """
    if isinstance(X, list):
        X = torch.tensor(X)
    elif isinstance(X, np.ndarray):
        X = torch.tensor(X)
    else:
        X = torch.tensor(X)
    return X.bool()


def to_long_tensor(X: Union[List, np.ndarray, torch.Tensor]) -> torch.LongTensor:
    r"""Convert ``List``, ``numpy.ndarray``, ``torch.Tensor`` to ``torch.LongTensor``.

    Args:
        ``X`` (``Union[List, np.ndarray, torch.Tensor]``): Input.
    """
    if isinstance(X, list):
        X = torch.tensor(X)
    elif isinstance(X, np.ndarray):
        X = torch.tensor(X)
    else:
        X = torch.tensor(X)
    return X.long()


def compose_pipes(*pipes: Callable) -> Callable:
    r""" Compose datapipe functions.

    Args:
        ``pipes`` (``Callable``): Datapipe functions to compose.
    """

    def composed_pipes(X: Any) -> torch.Tensor:
        for pipe in pipes:
            X = pipe(X)
        return X

    return composed_pipes
