from typing import Tuple

import torch
import torch.nn as nn

from dhg.structure.graphs import BiGraph


class BGNN_Adv(nn.Module):
    r"""The BGNN-Adv model proposed in `Cascade-BGNN: Toward Efficient Self-supervised Representation Learning on Large-scale Bipartite Graphs <https://arxiv.org/pdf/1906.11994.pdf>`_ paper (TNNLS 2020).

    Args:
        ``num_users`` (``int``): The Number of users.
        ``num_items`` (``int``): The Number of items.
        ``emb_dim`` (``int``): Embedding dimension.
        ``num_layers`` (``int``): The Number of layers. Defaults to ``3``.
        ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in training stage with probability ``drop_rate``. Default: ``0.0``.
    """

    def __init__(
        self, num_users: int, num_items: int, emb_dim: int, num_layers: int = 3, drop_rate: float = 0.0
    ) -> None:

        super().__init__()
        self.num_users, self.num_items = num_users, num_items
        self.num_layers = num_layers
        self.drop_rate = drop_rate
        self.u_embedding = nn.Embedding(num_users, emb_dim)
        self.i_embedding = nn.Embedding(num_items, emb_dim)
        self.reset_parameters()

    def reset_parameters(self):
        r"""Initialize learnable parameters.
        """
        nn.init.normal_(self.u_embedding.weight, 0, 0.1)
        nn.init.normal_(self.i_embedding.weight, 0, 0.1)

    def forward(self, ui_bigraph: BiGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""The forward function.

        Args:
            ``ui_bigraph`` (``dhg.BiGraph``): The user-item bipartite graph.
        """
        drop_rate = self.drop_rate if self.training else 0.0
        u_embs = self.u_embedding.weight
        i_embs = self.i_embedding.weight
        all_embs = torch.cat([u_embs, i_embs], dim=0)

        embs_list = [all_embs]
        for _ in range(self.num_layers):
            all_embs = ui_bigraph.smoothing_with_GCN(all_embs, drop_rate=drop_rate)
            embs_list.append(all_embs)
        embs = torch.stack(embs_list, dim=1)
        embs = torch.mean(embs, dim=1)

        u_embs, i_embs = torch.split(embs, [self.num_users, self.num_items], dim=0)
        return u_embs, i_embs
