import torch


def pairwise_euclidean_distance(x: torch.Tensor):
    """
    generate N x N node distance matrix
    :param x: a tensor of size N x C (N nodes with C feature dimension)
    :return: a tensor of N x N (distance matrix for each node pair)
    """
    assert isinstance(x, torch.Tensor)
    assert len(x.shape) == 2
    x = x.float()

    x_transpose = torch.transpose(x, dim0=0, dim1=1)
    x_inner = torch.matmul(x, x_transpose)
    x_inner = -2 * x_inner
    x_square = torch.sum(x ** 2, dim=1, keepdim=True)
    x_square_transpose = torch.transpose(x_square, dim0=0, dim1=1)
    dis = x_square + x_inner + x_square_transpose
    return dis
