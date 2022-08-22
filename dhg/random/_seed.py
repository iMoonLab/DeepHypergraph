import time
import random

import torch
import numpy as np

_MANUAL_SEED = None


def seed():
    r"""Return current random seed of DHG. Defaultly, the random seed is synchronized with the value of ``int(time.time())``.
    """
    global _MANUAL_SEED
    if _MANUAL_SEED is None:
        return int(time.time())
    else:
        return _MANUAL_SEED


def set_seed(seed: int):
    r"""Set the random seed of DHG.
    
    .. note::
        When you call this function, the random seeds of ``random``, ``numpy``, and ``pytorch`` will be set, simultaneously.
    
    Args:
        ``seed`` (``int``): The specified random seed.
    """
    global _MANUAL_SEED
    _MANUAL_SEED = seed
    random.seed(_MANUAL_SEED)
    np.random.seed(_MANUAL_SEED)
    torch.manual_seed(_MANUAL_SEED)
