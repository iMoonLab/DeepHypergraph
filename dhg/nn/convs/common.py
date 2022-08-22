from typing import List, Optional

import torch
import torch.nn as nn


class MLP(nn.Module):
    r"""A Multi-Layer Perception (MLP) model.

    Args:
        ``channel_list`` (``List[int]``): The list of channels of each layer.
        ``act_name`` (``str``): The name of activation function can be any `activation layer <https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity>`_ in Pytorch.
        ``act_kwargs`` (``dict``, optional): The keyword arguments of activation function. Defaults to ``None``.
        ``use_bn`` (``bool``): Whether to use batch normalization.
        ``drop_rate`` (``float``): Dropout ratio. Defaults to ``0.5``.
        ``is_last`` (``bool``): If set to True, the last layer will not use activation, batch normalization, and dropout.
    """

    def __init__(
        self,
        channel_list: List[int],
        act_name: str = "ReLU",
        act_kwargs: Optional[dict] = None,
        use_bn: bool = True,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ) -> None:
        assert len(channel_list) >= 2, "At least two elements in channel_list."
        super().__init__()
        act_module = getattr(nn.modules.activation, act_name)
        self.layers = nn.ModuleList()

        for _idx in range(1, len(channel_list) - 1):
            self.layers.append(nn.Linear(channel_list[_idx - 1], channel_list[_idx]))
            self.layers.append(act_module(**({} if act_kwargs is None else act_kwargs)))
            if use_bn:
                self.layers.append(nn.BatchNorm1d(channel_list[_idx]))
            self.layers.append(nn.Dropout(drop_rate))

        if is_last:
            self.layers.append(nn.Linear(channel_list[-2], channel_list[-1]))
        else:
            self.layers.append(nn.Linear(channel_list[-2], channel_list[-1]))
            self.layers.append(act_module(**({} if act_kwargs is None else act_kwargs)))
            if use_bn:
                self.layers.append(nn.BatchNorm1d(channel_list[-1]))
            self.layers.append(nn.Dropout(drop_rate))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        r"""The forward function.
        """
        for layer in self.layers:
            X = layer(X)
        return X
