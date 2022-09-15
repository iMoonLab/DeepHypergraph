import torch


def sparse_dropout(sp_mat: torch.Tensor, drop_rate: float, fill_value: float = 0.0) -> torch.Tensor:
    r"""Dropout function for sparse matrix.
    
    Args:
        ``sp_mat`` (``torch.Tensor``): The sparse matrix with format ``torch.sparse_coo_tensor``.
        ``drop_rate`` (``float``): Dropout rate.
        ``fill_value`` (``float``): The fill value for dropped elements. Defaults to ``0.0``.
    """
    sp_mat = sp_mat.coalesce()
    assert 0 <= drop_rate <= 1
    if drop_rate == 0:
        return sp_mat
    p = torch.ones(sp_mat._nnz()) * drop_rate
    drop_mask = torch.bernoulli(p).bool()
    sp_mat._values()[drop_mask] = fill_value
    return sp_mat


def sparse_mul(*sp_mats: torch.Tensor) -> torch.Tensor:
    r"""Multiply sparse matrices. The input can be a series of sparse matrices.
    
    Args:
        ``*sp_mats`` (``torch.Tensor``): The sparse matrices with format ``torch.sparse_coo_tensor``.
    """
    assert len(sp_mats) >= 2
    sp_mat = sp_mats[0]
    for i in range(1, len(sp_mats)):
        sp_mat = torch.sparse.mm(sp_mat, sp_mats[i])
    return sp_mat
