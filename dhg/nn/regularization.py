from typing import Union, List

import torch
import torch.nn as nn


class EmbeddingRegularization(nn.Module):
    r"""Regularization function for embeddings.

    Args:
        ``p`` (``int``): The power to use in the regularization. Defaults to ``2``.
        ``weight_decay`` (``float``): The weight of the regularization. Defaults to ``1e-4``.
    """

    def __init__(self, p: int = 2, weight_decay: float = 1e-4):
        super().__init__()
        self.p = p
        self.weight_decay = weight_decay

    def forward(self, *embs: List[torch.Tensor]):
        r"""The forward function.

        Args:
            ``embs`` (``List[torch.Tensor]``): The input embeddings.
        """
        loss = 0
        for emb in embs:
            loss += 1 / self.p * emb.pow(self.p).sum(dim=1).mean()
        return self.weight_decay * loss
