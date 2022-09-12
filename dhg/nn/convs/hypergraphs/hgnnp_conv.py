import torch
import torch.nn as nn

from dhg.structure.hypergraphs import Hypergraph


class HGNNPConv(nn.Module):
    r"""The HGNN :sup:`+` convolution layer proposed in `HGNN+: General Hypergraph Neural Networks <https://ieeexplore.ieee.org/document/9795251>`_ paper (IEEE T-PAMI 2022).

    Sparse Format:
    
    .. math::

        \left\{
            \begin{aligned}
                m_{\beta}^{t} &=\sum_{\alpha \in \mathcal{N}_{v}(\beta)} M_{v}^{t}\left(x_{\alpha}^{t}\right) \\
                y_{\beta}^{t} &=U_{e}^{t}\left(w_{\beta}, m_{\beta}^{t}\right) \\
                m_{\alpha}^{t+1} &=\sum_{\beta \in \mathcal{N}_{e}(\alpha)} M_{e}^{t}\left(x_{\alpha}^{t}, y_{\beta}^{t}\right) \\
                x_{\alpha}^{t+1} &=U_{v}^{t}\left(x_{\alpha}^{t}, m_{\alpha}^{t+1}\right) \\
            \end{aligned}
        \right.

    Matrix Format:

    .. math::
        \mathbf{X}^{\prime} = \sigma \left( \mathbf{D}_v^{-1} \mathbf{H} \mathbf{W}_e 
        \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{X} \mathbf{\Theta} \right).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``out_channels`` (int): :math:`C_{out}` is the number of output channels.
        ``bias`` (``bool``): If set to ``False``, the layer will not learn the bias parameter. Defaults to ``True``.
        ``use_bn`` (``bool``): If set to ``True``, the layer will use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``): If set to a positive number, the layer will use dropout. Defaults to ``0.5``.
        ``is_last`` (``bool``): If set to ``True``, the layer will not apply the final activation and dropout functions. Defaults to ``False``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        r"""The forward function.

        Args:
            X (``torch.Tensor``): Input vertex feature matrix. Size :math:`(|\mathcal{V}|, C_{in})`.
            hg (``dhg.Hypergraph``): The hypergraph structure that contains :math:`|\mathcal{V}|` vertices.
        """
        X = self.theta(X)
        if self.bn is not None:
            X = self.bn(X)
        X = hg.v2v(X, aggr="mean")
        if not self.is_last:
            X = self.drop(self.act(X))
        return X
