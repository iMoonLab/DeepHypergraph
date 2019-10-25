import torch

from SuperMoon.hyedge import matrix2index, index2matrix


def test_matrix2index():
    matrix = torch.tensor([[0, 1, 0, 0],
                           [1, 1, 0, 1],
                           [0, 0, 1, 0]])
    index = matrix2index(matrix)
    assert torch.all(index == torch.tensor([[0, 1, 1, 1, 2],
                                            [1, 0, 1, 3, 2]]))


def test_index2matrix():
    index = torch.tensor([[0, 1, 1, 1, 2],
                          [1, 0, 1, 3, 2]])
    matrix = index2matrix(index)
    assert torch.all(matrix == torch.tensor([[0, 1, 0, 0],
                                             [1, 1, 0, 1],
                                             [0, 0, 1, 0]]).float())
