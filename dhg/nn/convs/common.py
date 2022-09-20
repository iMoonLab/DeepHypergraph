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


class MultiHeadWrapper(nn.Module):
    r"""A wrapper to apply multiple heads to a given layer.

    Args:
        ``num_heads`` (``int``): The number of heads.
        ``readout`` (``bool``): The readout method. Can be ``"mean"``, ``"max"``, ``"sum"``, or ``"concat"``.
        ``layer`` (``nn.Module``): The layer to apply multiple heads.
        ``**kwargs``: The keyword arguments for the layer.
    
    Example:
        >>> import torch
        >>> import dhg
        >>> from dhg.nn import GATConv, MultiHeadWrapper
        >>> multi_head_layer = MultiHeadWrapper(
                4,
                "concat",
                GATConv,
                in_channels=16,
                out_channels=8,
            )
        >>> X = torch.rand(20, 16)
        >>> g = dhg.random.graph_Gnm(20, 15)
        >>> X_ = multi_head_layer(X=X, g=g)
    """

    def __init__(self, num_heads: int, readout: str, layer: nn.Module, **kwargs) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_heads):
            self.layers.append(layer(**kwargs))
        self.num_heads = num_heads
        self.readout = readout

    def forward(self, **kwargs) -> torch.Tensor:
        r"""The forward function.

        .. note::
            You must explicitly pass the keyword arguments to the layer. For example, if the layer is ``GATConv``, you must pass ``X=X`` and ``g=g``.
        """
        if self.readout == "concat":
            return torch.cat([layer(**kwargs) for layer in self.layers], dim=-1)
        else:
            outs = torch.stack([layer(**kwargs) for layer in self.layers])
            if self.readout == "mean":
                return outs.mean(dim=0)
            elif self.readout == "max":
                return outs.max(dim=0)[0]
            elif self.readout == "sum":
                return outs.sum(dim=0)
            else:
                raise ValueError("Unknown readout type")


class Discriminator(nn.Module):
    r"""The Discriminator for Generative Adversarial Networks (GANs).

    Args:
        ``in_channels`` (``int``): The number of input channels.
        ``hid_channels`` (``int``): The number of hidden channels.
        ``out_channels`` (``int``): The number of output channels.
        ``drop_rate`` (``float``): Dropout ratio. Defaults to ``0.5``.
    """

    def __init__(self, in_channels: int, hid_channels: int, out_channels: int, drop_rate: float = 0.5):

        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.LeakyReLU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(hid_channels, out_channels),
            nn.Sigmoid(),
        )

    def forward(self, X):
        """The forward function.
        
        Args:
            ``X`` (``torch.Tensor``): The input tensor.
        """
        X = self.layers(X)
        return X
