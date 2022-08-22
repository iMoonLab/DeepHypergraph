import pytest
import torch
from dhg.nn import EmbeddingRegularization


def test_embedding_reg():
    emb_reg = EmbeddingRegularization(p=2, weight_decay=1e-4)
    embs = [torch.randn(10, 3), torch.randn(10, 3)]
    loss = emb_reg(*embs)
    true_loss = 0
    for emb in embs:
        true_loss += 1 / 2 * emb.norm(2).pow(2) / 10
    assert loss.item() == pytest.approx(1e-4 * true_loss.item())
