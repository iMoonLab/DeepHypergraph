import torch
import torch.nn as nn

import dhg
from dhg.nn import MLP
from dhg.nn import GINConv


class GIN(nn.Module):
    r"""The GIN model proposed in `How Powerful are Graph Neural Networks? <https://arxiv.org/pdf/1810.00826>`_ paper (ICLR 2019).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``num_layers`` (``int``): The number of layers in the GIN model. In the original `code <https://github.com/weihua916/powerful-gnns/blob/master/main.py#L102>`_, it is set to ``5``.
        ``num_mlp_layers`` (``int``): The number of layers in the MLP. Defaults to ``2``.
        ``eps`` (``float``): The epsilon value. Defaults to ``0.0``.
        ``train_eps`` (``bool``): If set to ``True``, the epsilon value will be trainable. Defaults to ``False``.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): The dropout ratio. Defaults to ``0.5``.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        num_layers: int,
        num_mlp_layers: int = 2,
        eps: float = 0.0,
        train_eps: bool = False,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        assert num_layers >= 2, "num_layers must be greater than or equal to 2."
        self.layers = nn.ModuleList()
        self.layers.append(
            GINConv(
                MLP(
                    [in_channels] + [hid_channels] * num_mlp_layers,
                    use_bn=use_bn,
                    drop_rate=drop_rate,
                ),
                eps,
                train_eps,
            )
        )
        for _ in range(num_layers - 1):
            self.layers.append(
                GINConv(
                    MLP(
                        [hid_channels] * (num_mlp_layers + 1),
                        use_bn=use_bn,
                        drop_rate=drop_rate,
                    ),
                    eps,
                    train_eps,
                )
            )
        self.pred_layers = nn.ModuleList()
        self.pred_layers.append(nn.Linear(in_channels, num_classes))
        for _ in range(num_layers):
            self.pred_layers.append(nn.Linear(hid_channels, num_classes))

    def forward(self, X: torch.Tensor, g: "dhg.Graph") -> torch.Tensor:
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``g`` (``dhg.Graph``): The graph structure that contains :math:`N` vertices.
        """
        pred = self.pred_layers[0](X)
        for idx, layer in enumerate(self.layers):
            X = layer(X, g)
            pred += self.pred_layers[idx + 1](X)
        return pred
