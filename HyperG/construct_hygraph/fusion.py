from typing import Union, Tuple, List

import torch

from HyperG.construct_hyedge import hyedge_count


def hyedge_concat(Hs: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]):
    hyedge_num = 0
    Hs_new = []
    for H in Hs:
        H[1, :] += hyedge_num
        hyedge_num = hyedge_count(H)
        Hs_new.append(H)
    return torch.cat(Hs_new, dim=1)
