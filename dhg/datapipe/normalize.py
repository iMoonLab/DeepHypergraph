from typing import Optional, Union
import torch


def norm_ft(
    X: torch.Tensor, ord: Optional[Union[int, float]] = None
) -> torch.Tensor:
    r"""Normalize the input feature matrix with specified ``ord`` refer to pytorch's `torch.linalg.norm <https://pytorch.org/docs/stable/generated/torch.linalg.norm.html#torch.linalg.norm>`_ function.

    .. note::
        The input feature matrix is expected to be a 1D vector or a 2D tensor with shape (num_samples, num_features).

    Args:
        ``X`` (``torch.Tensor``): The input feature.
        ``ord`` (``Union[int, float]``, optional): The order of the norm can be either an ``int``, ``float``. If ``ord`` is ``None``, the norm is computed with the 2-norm. Defaults to ``None``.

    Examples:
        >>> import dhg.datapipe as dd
        >>> import torch
        >>> X = torch.tensor([
                    [0.1, 0.2, 0.5],
                    [0.5, 0.2, 0.3],
                    [0.3, 0.2, 0]
                ])
        >>> dd.norm_ft(X)
        tensor([[0.1826, 0.3651, 0.9129],
                [0.8111, 0.3244, 0.4867],
                [0.8321, 0.5547, 0.0000]])
    """
    if X.dim() == 1:
        X_norm = 1 / torch.linalg.norm(X, ord=ord)
        X_norm[torch.isinf(X_norm)] = 0
        return X * X_norm
    elif X.dim() == 2:
        X_norm = 1 / torch.linalg.norm(X, ord=ord, dim=1, keepdim=True)
        X_norm[torch.isinf(X_norm)] = 0
        return X * X_norm
    else:
        raise ValueError(
            "The input feature matrix is expected to be a 1D verter or a 2D tensor with shape (num_samples, num_features)."
        )

