from typing import Tuple, Callable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dhg.structure.graphs import BiGraph
from dhg.nn.convs.common import Discriminator


class BGNN_Adv(nn.Module):
    r"""The BGNN-Adv model proposed in `Cascade-BGNN: Toward Efficient Self-supervised Representation Learning on Large-scale Bipartite Graphs <https://arxiv.org/pdf/1906.11994.pdf>`_ paper (TNNLS 2020).

    Args:
        ``u_dim`` (``int``): The dimension of the vertex feature in set :math:`U`.
        ``v_dim`` (``int``): The dimension of the vertex feature in set :math:`V`.
        ``layer_depth`` (``int``): The depth of layers.
    """

    def __init__(self, u_dim: int, v_dim: int, layer_depth: int = 3) -> None:

        super().__init__()
        self.layer_depth = layer_depth
        self.layers = nn.ModuleList()

        for _idx in range(layer_depth):
            if _idx % 2 == 0:
                self.layers.append(nn.Linear(v_dim, u_dim))
            else:
                self.layers.append(nn.Linear(u_dim, v_dim))

    def forward(self, X_u: torch.Tensor, X_v: torch.Tensor, g: BiGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""The forward function.

        Args:
            ``X_u`` (``torch.Tensor``): The feature matrix of vertices in set :math:`U`.
            ``X_v`` (``torch.Tensor``): The feature matrix of vertices in set :math:`V`.
            ``g`` (``BiGraph``): The bipartite graph.
        """
        last_X_u, last_X_v = X_u, X_v
        for _idx in range(self.layer_depth):
            if _idx % 2 == 0:
                _tmp = self.layers[_idx](last_X_v)
                last_X_u = torch.tanh(g.v2u(_tmp, aggr="sum"))
            else:
                _tmp = self.layers[_idx](last_X_u)
                last_X_v = torch.tanh(g.u2v(_tmp, aggr="sum"))
        return last_X_u

    def train_one_layer(
        self,
        X_true: torch.Tensor,
        X_other: torch.Tensor,
        mp_func: Callable,
        layer: nn.Module,
        lr: float,
        weight_decay: float,
        max_epoch: int,
        drop_rate: float = 0.5,
        device: str = "cpu",
    ):
        netG = layer.to(device)
        netD = Discriminator(X_true.shape[1], 16, 1, drop_rate=drop_rate).to(device)

        optimizer_G = optim.Adam(netG.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer_D = optim.Adam(netD.parameters(), lr=lr, weight_decay=weight_decay)

        X_true, X_other = X_true.detach().to(device), X_other.detach().to(device)
        lbl_real = torch.ones(X_true.shape[0], 1, requires_grad=False).to(device)
        lbl_fake = torch.zeros(X_true.shape[0], 1, requires_grad=False).to(device)

        netG.train(), netD.train()
        for _ in range(max_epoch):
            X_real = X_true
            X_fake = torch.tanh(mp_func(netG(X_other)))

            # step 1: train Discriminator
            optimizer_D.zero_grad()

            pred_real = netD(X_real)
            pred_fake = netD(X_fake.detach())

            loss_D = F.binary_cross_entropy(pred_real, lbl_real) + F.binary_cross_entropy(pred_fake, lbl_fake)
            loss_D.backward()
            optimizer_D.step()

            # step 2: train Generator
            optimizer_G.zero_grad()

            pred_fake = netD(X_fake)

            loss_G = F.binary_cross_entropy(pred_fake, lbl_real)
            loss_G.backward()
            optimizer_G.step()

    def train_with_cascaded(
        self,
        X_u: torch.Tensor,
        X_v: torch.Tensor,
        g: BiGraph,
        lr: float,
        weight_decay: float,
        max_epoch: int,
        drop_rate: float = 0.5,
        device: str = "cpu",
    ):
        r"""Train the model with cascaded strategy.

        Args:
            ``X_u`` (``torch.Tensor``): The feature matrix of vertices in set :math:`U`.
            ``X_v`` (``torch.Tensor``): The feature matrix of vertices in set :math:`V`.
            ``g`` (``BiGraph``): The bipartite graph.
            ``lr`` (``float``): The learning rate.
            ``weight_decay`` (``float``): The weight decay.
            ``max_epoch`` (``int``): The maximum number of epochs.
            ``drop_rate`` (``float``): The dropout rate. Default: ``0.5``.
            ``device`` (``str``): The device to use. Default: ``"cpu"``.
        """
        self = self.to(device)
        last_X_u, last_X_v = X_u.to(device), X_v.to(device)
        for _idx in range(self.layer_depth):
            if _idx % 2 == 0:
                self.train_one_layer(
                    last_X_u,
                    last_X_v,
                    lambda x: g.v2u(x, aggr="sum"),
                    self.layers[_idx],
                    lr,
                    weight_decay,
                    max_epoch,
                    drop_rate,
                    device,
                )
                with torch.no_grad():
                    last_X_u = torch.tanh(g.v2u(self.layers[_idx](last_X_v), aggr="sum"))
            else:
                self.train_one_layer(
                    last_X_v,
                    last_X_u,
                    lambda x: g.u2v(x, aggr="sum"),
                    self.layers[_idx],
                    lr,
                    weight_decay,
                    max_epoch,
                    drop_rate,
                    device,
                )
                with torch.no_grad():
                    last_X_v = torch.tanh(g.u2v(self.layers[_idx](last_X_u), aggr="sum"))
        return last_X_u


class BGNN_MLP(nn.Module):
    r"""The BGNN-MLP model proposed in `Cascade-BGNN: Toward Efficient Self-supervised Representation Learning on Large-scale Bipartite Graphs <https://arxiv.org/pdf/1906.11994.pdf>`_ paper (TNNLS 2020).

    Args:
        ``u_dim`` (``int``): The dimension of the vertex feature in set :math:`U`.
        ``v_dim`` (``int``): The dimension of the vertex feature in set :math:`V`.
        ``hid_dim`` (``int``): The dimension of the hidden layer.
        ``decoder_hid_dim`` (``int``): The dimension of the hidden layer in the decoder.
        ``drop_rate`` (``float``): The dropout rate. Default: ``0.5``.
        ``layer_depth`` (``int``): The depth of layers. Default: ``3``.
    """

    def __init__(
        self, u_dim: int, v_dim: int, hid_dim: int, decoder_hid_dim: int, drop_rate: float = 0.5, layer_depth: int = 3,
    ) -> None:

        super().__init__()
        self.layer_depth = layer_depth
        self.layers = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for _idx in range(layer_depth):
            if _idx % 2 == 0:
                self.layers.append(nn.Linear(v_dim, hid_dim))
                self.decoders.append(Decoder(hid_dim, decoder_hid_dim, u_dim, drop_rate=drop_rate))
            else:
                self.layers.append(nn.Linear(u_dim, hid_dim))
                self.decoders.append(Decoder(hid_dim, decoder_hid_dim, v_dim, drop_rate=drop_rate))

    def forward(self, X_u: torch.Tensor, X_v: torch.Tensor, g: BiGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""The forward function.

        Args:
            ``X_u`` (``torch.Tensor``): The feature matrix of vertices in set :math:`U`.
            ``X_v`` (``torch.Tensor``): The feature matrix of vertices in set :math:`V`.
            ``g`` (``BiGraph``): The bipartite graph.
        """
        last_X_u, last_X_v = X_u, X_v
        for _idx in range(self.layer_depth):
            if _idx % 2 == 0:
                _tmp = self.layers[_idx](last_X_v)
                last_X_u = self.decoders[_idx](torch.tanh(g.v2u(_tmp, aggr="sum")))
            else:
                _tmp = self.layers[_idx](last_X_u)
                last_X_v = self.decoders[_idx](torch.tanh(g.u2v(_tmp, aggr="sum")))
        return last_X_u

    def train_one_layer(
        self,
        X_true: torch.Tensor,
        X_other: torch.Tensor,
        mp_func: Callable,
        layer: nn.Module,
        decoder: nn.Module,
        lr: float,
        weight_decay: float,
        max_epoch: int,
        device: str = "cpu",
    ):
        netG = layer.to(device)
        netD = decoder.to(device)

        optimizer = optim.Adam([*netG.parameters(), *netD.parameters()], lr=lr, weight_decay=weight_decay)

        X_true, X_other = X_true.detach().to(device), X_other.detach().to(device)

        netG.train(), netD.train()
        for _ in range(max_epoch):
            X_real = X_true
            X_fake = netD(torch.tanh(mp_func(netG(X_other))))

            optimizer.zero_grad()
            loss = F.mse_loss(X_fake, X_real)
            loss.backward()
            optimizer.step()

    def train_with_cascaded(
        self,
        X_u: torch.Tensor,
        X_v: torch.Tensor,
        g: BiGraph,
        lr: float,
        weight_decay: float,
        max_epoch: int,
        device: str = "cpu",
    ):
        r"""Train the model with cascaded strategy.

        Args:
            ``X_u`` (``torch.Tensor``): The feature matrix of vertices in set :math:`U`.
            ``X_v`` (``torch.Tensor``): The feature matrix of vertices in set :math:`V`.
            ``g`` (``BiGraph``): The bipartite graph.
            ``lr`` (``float``): The learning rate.
            ``weight_decay`` (``float``): The weight decay.
            ``max_epoch`` (``int``): The maximum number of epochs.
            ``device`` (``str``): The device to use. Default: ``"cpu"``.
        """
        self = self.to(device)
        last_X_u, last_X_v = X_u.to(device), X_v.to(device)
        for _idx in range(self.layer_depth):
            if _idx % 2 == 0:
                self.train_one_layer(
                    last_X_u,
                    last_X_v,
                    lambda x: g.v2u(x, aggr="sum"),
                    self.layers[_idx],
                    self.decoders[_idx],
                    lr,
                    weight_decay,
                    max_epoch,
                    device,
                )
                with torch.no_grad():
                    self.decoders[_idx].eval()
                    last_X_u = self.decoders[_idx](torch.tanh(g.v2u(self.layers[_idx](last_X_v), aggr="sum")))
            else:
                self.train_one_layer(
                    last_X_v,
                    last_X_u,
                    lambda x: g.u2v(x, aggr="sum"),
                    self.layers[_idx],
                    self.decoders[_idx],
                    lr,
                    weight_decay,
                    max_epoch,
                    device,
                )
                with torch.no_grad():
                    self.decoders[_idx].eval()
                    last_X_v = self.decoders[_idx](torch.tanh(g.u2v(self.layers[_idx](last_X_u), aggr="sum")))
        return last_X_u


class Decoder(nn.Module):
    def __init__(self, in_channels: int, hid_channels: int, out_channels: int, drop_rate: float = 0.5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(hid_channels, out_channels),
            nn.Tanh(),
        )

    def forward(self, X):
        X = self.layers(X)
        return X
