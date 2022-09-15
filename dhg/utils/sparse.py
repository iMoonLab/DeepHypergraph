import torch


def sparse_dropout(sp_mat: torch.Tensor, p: float, fill_value: float = 0.0) -> torch.Tensor:
    r"""Dropout function for sparse matrix. This function will return a new sparse matrix with the same shape as the input sparse matrix, but with some elements dropped out.
    
    Args:
        ``sp_mat`` (``torch.Tensor``): The sparse matrix with format ``torch.sparse_coo_tensor``.
        ``p`` (``float``): Probability of an element to be dropped. 
        ``fill_value`` (``float``): The fill value for dropped elements. Defaults to ``0.0``.
    """
    device = sp_mat.device
    sp_mat = sp_mat.coalesce()
    assert 0 <= p <= 1
    if p == 0:
        return sp_mat
    p = torch.ones(sp_mat._nnz(), device=device) * p
    keep_mask = torch.bernoulli(1 - p).to(device)
    fill_values = torch.logical_not(keep_mask) * fill_value
    new_sp_mat = torch.sparse_coo_tensor(
        sp_mat._indices(),
        sp_mat._values() * keep_mask + fill_values,
        size=sp_mat.size(),
        device=sp_mat.device,
        dtype=sp_mat.dtype,
    )
    return new_sp_mat

