from typing import Any

from torch import nn
from torch.nn.modules.module import T_co


class HyConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *input: Any, **kwargs: Any) -> T_co:
        return super().forward(*input, **kwargs)
