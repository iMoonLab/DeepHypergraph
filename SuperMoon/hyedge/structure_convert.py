import torch


def index2matrix(index):
    assert index.size(0) == 2

    index = index.long()
    v_len = index.size(1)
    v = torch.ones(v_len).float()
    matrix = torch.sparse_coo_tensor(index, v).to_dense()
    return matrix


def matrix2index(matrix: torch.tensor):
    i_v = matrix.to_sparse()
    index = i_v.indices()
    return index;
