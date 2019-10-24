from typing import Union, Tuple, List

import torch

from SuperMoon.hyedge import count_hyedge, count_node, contiguous_hyedge_idx


def hyedge_concat(Hs: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], same_node=True):
    node_num = 0
    hyedge_num = 0
    Hs_new = []
    for H in Hs:
        _H = H.clone()
        if not same_node:
            _H[0, :] += node_num
        _H[1, :] += hyedge_num

        Hs_new.append(_H)

        hyedge_num += count_hyedge(H)
        node_num += count_node(H)
    Hs_new = torch.cat(Hs_new, dim=1)
    return contiguous_hyedge_idx(Hs_new)
