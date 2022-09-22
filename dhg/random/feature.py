from typing import Union
import torch
import numpy as np


def normal_features(labels: Union[list, np.ndarray, torch.Tensor], noise: float = 1.0):
    r"""Generate random features that are satisfying the normal distribution.
    
    Args:
        ``labels`` (``Union[list, np.ndarray, torch.Tensor]``): The label list.
        ``noise`` (``float``, optional): The noise of the normal distribution. Defaults to ``1.0``.
    
    Examples:
        >>> import dhg
        >>> label = [1, 3, 5, 2, 1, 5]
        >>> dhg.random.normal_features(label)
        tensor([[ 0.3204, -0.3059, -0.3103, -0.6558],
                [-1.0128,  0.0846,  0.4317, -0.1427],
                [ 0.0776, -0.6265, -0.7592, -0.5559],
                [ 0.8282, -0.5076, -1.1508,  0.6998],
                [ 0.4600, -0.8477,  0.8881,  0.7426],
                [-0.4456,  0.8452, -1.2390,  2.3204]])
    """
    if isinstance(labels, list):
        labels = np.array(labels)
    elif isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    elif not isinstance(labels, np.ndarray):
        raise TypeError("The type of labels must be list, np.ndarray or torch.Tensor.")
    assert len(labels.shape) == 1, "The shape of labels must be (num_vertices, )."
    label_set = np.unique(labels).tolist()
    N, C = labels.shape[0], len(label_set)
    lebel_list = []
    for i in range(N):
        lebel_list.append(label_set.index(labels[i]))
    labels = np.array(lebel_list)
    centers = np.zeros((N, C))
    centers[np.arange(N), labels] = 1
    features = np.random.normal(centers, noise, size=(N, C))
    return torch.from_numpy(features).float()

