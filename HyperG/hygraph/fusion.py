from typing import Union, Tuple, List

import torch

from HyperG.hyedge import count_hyedge, count_node


def hyedge_concat(Hs: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], same_node=True):
    node_num = 0
    hyedge_num = 0
    Hs_new = []
    for H in Hs:
        if not same_node:
            H[0, :] += node_num
        H[1, :] += hyedge_num

        Hs_new.append(H)

        hyedge_num += count_hyedge(H)
        node_num += count_node(H)

    return torch.cat(Hs_new, dim=1)
