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

    Examples:
        >>> import dhg.datapipe as dd
        >>> X = [[0.1, 0.2, 0.5],
                 [0.5, 0.2, 0.3],
                 [0.3, 0.2, 0]]
        >>> dd.to_tensor(X)
        tensor([[0.1000, 0.2000, 0.5000],
                [0.5000, 0.2000, 0.3000],
                [0.3000, 0.2000, 0.0000]])
    """
    if isinstance(X, list):
        X = torch.tensor(X)
    elif isinstance(X, scipy.sparse.csr_matrix):
        X = X.todense()
        X = torch.tensor(X)
    elif isinstance(X, scipy.sparse.coo_matrix):
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

    Examples:
        >>> import dhg.datapipe as dd
        >>> X = [[0.1, 0.2, 0.5],
                 [0.5, 0.2, 0.3],
                 [0.3, 0.2, 0]]
        >>> dd.to_bool_tensor(X)
        tensor([[ True,  True,  True],
                [ True,  True,  True],
                [ True,  True, False]])
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

    Examples:
        >>> import dhg.datapipe as dd
        >>> X = [[1, 2, 5],
                 [5, 2, 3],
                 [3, 2, 0]]
        >>> dd.to_long_tensor(X)
        tensor([[1, 2, 5],
                [5, 2, 3],
                [3, 2, 0]])
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
