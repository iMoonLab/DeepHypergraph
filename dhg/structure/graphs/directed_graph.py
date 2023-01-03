import random
import pickle
from pathlib import Path
from copy import deepcopy
from typing import Union, Optional, List, Tuple, Dict, Any

import torch
import numpy as np
import scipy.spatial

from dhg.visualization.structure.draw import draw_digraph
from ..base import BaseGraph
from dhg.utils.sparse import sparse_dropout


class DiGraph(BaseGraph):
    r""" Class for directed graph.

        Args:
            ``num_v`` (``int``): The Number of vertices.
            ``e_list`` (``Union[List[int], List[List[int]]]``, optional): Initial edge set. Defaults to ``None``.
            ``e_weight`` (``Union[float, List[float]]``, optional): A list of weights for edges. Defaults to ``None``.
            ``extra_selfloop`` (``bool``, optional): Whether to add extra self-loop to the directed graph. Defaults to ``False``.
            ``merge_op`` (``str``): The operation to merge those conflicting edges, which can be one of ``'mean'``, ``'sum'``, or ``'max'``. Defaults to ``'mean'``.
            ``device`` (``torch.device``, optional): The device to store the directed graph. Defaults to ``torch.device('cpu')``.
    """

    def __init__(
        self,
        num_v: int,
        e_list: Optional[Union[List[int], List[List[int]]]] = None,
        e_weight: Optional[Union[float, List[float]]] = None,
        extra_selfloop: bool = False,
        merge_op: str = "mean",
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(num_v, extra_selfloop=extra_selfloop, device=device)
        if e_list is not None:
            self.add_edges(e_list, e_weight, merge_op=merge_op)

    def __repr__(self) -> str:
        r"""Print the directed graph information.
        """
        return f"Directed Graph(num_v={self.num_v}, num_e={self.num_e})"

    @property
    def state_dict(self) -> Dict[str, Any]:
        r"""Get the state dict of the directed graph.
        """
        return {
            "num_v": self.num_v,
            "raw_e_dict": self._raw_e_dict,
            "raw_selfloop_dict": self._raw_selfloop_dict,
            "has_extra_selfloop": self._has_extra_selfloop,
        }

    def save(self, file_path: Union[str, Path]):
        r"""Save the DHG's directed graph structure to a file.

        Args:
            ``file_path`` (``Union[str, Path]``): The file path to store the DHG's directed graph structure.
        """
        file_path = Path(file_path)
        assert file_path.parent.exists(), "The directory does not exist."
        data = {"class": "DiGraph", "state_dict": self.state_dict}
        with open(file_path, "wb") as fp:
            pickle.dump(data, fp)

    @staticmethod
    def load(file_path: Union[str, Path]):
        r"""Load the DHG's directed graph structure from a file.

        Args:
            ``file_path`` (``Union[str, Path]``): The file path to load the DHG's directed graph structure.
        """
        file_path = Path(file_path)
        assert file_path.exists(), "The file does not exist."
        with open(file_path, "rb") as fp:
            data = pickle.load(fp)
        assert data["class"] == "DiGraph", "The file is not a DHG's directed graph."
        return DiGraph.from_state_dict(data["state_dict"])

    def draw(
        self,
        e_style: str = "line",
        v_label: Optional[List[str]] = None,
        v_size: Union[float, list] = 1.0,
        v_color: Union[str, list] = "r",
        v_line_width: Union[str, list] = 1.0,
        e_color: Union[str, list] = "gray",
        e_line_width: Union[str, list] = 1.0,
        font_size: int = 1.0,
        font_family: str = "sans-serif",
        push_v_strength: float = 1.0,
        push_e_strength: float = 1.0,
        pull_e_strength: float = 1.0,
        pull_center_strength: float = 1.0,
    ):
        r"""Draw the directed graph structure. 

        Args:
            ``e_style`` (``str``): The edge style. The supported styles are only ``'line'``. Defaults to ``'line'``.
            ``v_label`` (``list``): The vertex label. Defaults to ``None``.
            ``v_size`` (``Union[str, list]``): The vertex size. If ``v_size`` is a ``float``, all vertices will have the same size. If ``v_size`` is a ``list``, the size of each vertex will be set according to the corresponding element in the list. Defaults to ``1.0``.
            ``v_color`` (``Union[str, list]``): The vertex `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_. If ``v_color`` is a ``str``, all vertices will have the same color. If ``v_color`` is a ``list``, the color of each vertex will be set according to the corresponding element in the list. Defaults to ``'r'``.
            ``v_line_width`` (``Union[str, list]``): The vertex line width. If ``v_line_width`` is a ``float``, all vertices will have the same line width. If ``v_line_width`` is a ``list``, the line width of each vertex will be set according to the corresponding element in the list. Defaults to ``1.0``.
            ``e_color`` (``Union[str, list]``): The edge `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_. If ``e_color`` is a ``str``, all edges will have the same color. If ``e_color`` is a ``list``, the color of each edge will be set according to the corresponding element in the list. Defaults to ``'gray'``.
            ``e_line_width`` (``Union[str, list]``): The edge line width. If ``e_line_width`` is a ``float``, all edges will have the same line width. If ``e_line_width`` is a ``list``, the line width of each edge will be set according to the corresponding element in the list. Defaults to ``1.0``.
            ``font_size`` (``int``): The font size. Defaults to ``1.0``.
            ``font_family`` (``str``): The font family. Defaults to ``'sans-serif'``.
            ``push_v_strength`` (``float``): The vertex push strength. Defaults to ``1.0``.
            ``push_e_strength`` (``float``): The edge push strength. Defaults to ``1.0``.
            ``pull_e_strength`` (``float``): The edge pull strength. Defaults to ``1.0``.
            ``pull_center_strength`` (``float``): The center pull strength. Defaults to ``1.0``.
        """
        draw_digraph(
            self,
            e_style,
            v_label,
            v_size,
            v_color,
            v_line_width,
            e_color,
            e_line_width,
            font_size,
            font_family,
            push_v_strength,
            push_e_strength,
            pull_e_strength,
            pull_center_strength,
        )

    def clear(self):
        r"""Remove all edges in the directed graph.
        """
        return super().clear()

    def clone(self):
        r"""Clone the directed graph.
        """
        _g = DiGraph(self.num_v, extra_selfloop=self._has_extra_selfloop, device=self.device)
        if self._raw_e_dict is not None:
            _g._raw_e_dict = deepcopy(self._raw_e_dict)
        if self._raw_selfloop_dict is not None:
            _g._raw_selfloop_dict = deepcopy(self._raw_selfloop_dict)
        _g.cache = deepcopy(self.cache)
        return _g

    def to(self, device: torch.device):
        r"""Move the directed graph to the specified device.

        Args:
            ``device`` (``torch.device``): The device to store the directed graph.
        """
        return super().to(device)

    # =====================================================================================
    # some construction functions
    @staticmethod
    def from_state_dict(state_dict: dict):
        r"""Load the directed graph from the state dict.

        Args:
            ``state_dict`` (``dict``): The state dict to load the directed graph.
        """
        _g = DiGraph(state_dict["num_v"], extra_selfloop=state_dict["has_extra_selfloop"])
        _g._raw_e_dict = deepcopy(state_dict["raw_e_dict"])
        _g._raw_selfloop_dict = deepcopy(state_dict["raw_selfloop_dict"])
        return _g

    @staticmethod
    def from_adj_list(
        num_v: int,
        adj_list: List[List[int]],
        extra_selfloop: bool = False,
        device: torch.device = torch.device("cpu"),
    ) -> "DiGraph":
        r"""Construct a directed graph from the adjacency list. Each line in the adjacency list has two components. The first element in each line is the source vertex index, and the rest elements are the target vertex indices that connected to the source vertex.

        .. note::
            This function can only construct the unweighted directed graph.

        Args:
            ``num_v`` (``int``): The number of vertices.
            ``adj_list`` (``List[List[int]]``): Adjacency list.
            ``extra_selfloop`` (``bool``): Whether to add extra self-loop. Defaults to ``False``.
            ``device`` (``torch.device``): The device to store the directed graph. Defaults to ``torch.device('cpu')``.
        """
        e_list = []
        for line in adj_list:
            if len(line) <= 1:
                continue
            v_src = line[0]
            e_list.extend([(v_src, v_dst) for v_dst in line[1:]])
        _g = DiGraph(num_v, e_list, extra_selfloop=extra_selfloop, device=device)
        return _g

    @staticmethod
    def from_feature_kNN(
        features: torch.Tensor,
        k: int,
        p: int = 2,
        distance2weight: bool = False,
        include_center: bool = False,
        center_as_src: bool = True,
        device=torch.device("cpu"),
    ) -> "DiGraph":
        r"""Construct a directed graph from feature matrix with ``kNN`` algorithm.

        Args:
            ``features`` (``torch.Tensor``): Feature tensor. Size: :math:`(N_v \times C)`.
            ``k`` (``int``): The Number of nearest neighbors for each vertex.
            ``p`` (``int``): The p-norm for distance computation. Defaults to ``2``.
            ``distance2weight`` (``bool``): Whether to use distance as weight. If set to ``True``,
                this function will project the distance to weight by :math:`e^{-x}`, where :math:`x`
                is the computed distance. If set to ``False``, this function will set the weight of
                all edges to ``1``. Defaults to ``False``.
            ``include_center`` (``bool``): Whether the k-neighborhood includes the center vertex itself. Defaults to ``False``.
            ``center_as_src`` (``bool``): Whether the center vertex is the source vertex of the edge. Defaults to ``True``.
            ``device`` (``torch.device``): The device to store the directed graph. Defaults to ``torch.device('cpu')``.
        """
        features = features.cpu().numpy()
        assert features.ndim == 2, "Feature matrix should be 2-D."
        assert (
            k <= features.shape[0]
        ), "The number of nearest neighbors should be less than or equal to the number of vertices."
        num_v = features.shape[0]
        tree = scipy.spatial.cKDTree(features)
        if include_center:
            find_tk = k
        else:
            find_tk = k + 1
        nbr_dist, nbr_idx = tree.query(features, k=find_tk, p=p)
        center_idx = np.arange(num_v).reshape(-1, 1).repeat(find_tk - 1, 1)
        nbr_dist = nbr_dist[:, 1:]
        nbr_idx = nbr_idx[:, 1:]
        if center_as_src:
            e_list = np.concatenate([center_idx.reshape(-1, 1), nbr_idx.reshape(-1, 1)], axis=1).tolist()
        else:
            e_list = np.concatenate([nbr_idx.reshape(-1, 1), center_idx.reshape(-1, 1)], axis=1).tolist()
        if distance2weight:
            e_weight = np.exp(-nbr_dist).reshape(-1)
        else:
            e_weight = np.ones_like(nbr_dist).reshape(-1)
        _g = DiGraph(num_v, e_list, e_weight, device=device)
        if include_center:
            _g.add_extra_selfloop()
        return _g

    # =====================================================================================
    # some structure modification functions
    def add_edges(
        self,
        e_list: Union[List[int], List[List[int]]],
        e_weight: Optional[Union[float, List[float]]] = None,
        merge_op: str = "mean",
    ):
        r"""Add edges to the directed graph.

        Args:
            ``e_list`` (``Union[List[int], List[List[int]]]``): Edge list.
            ``e_weight`` (``Union[float, List[float]], optional``): A list of weights for edges. Defaults to ``None``.
            ``merge_op`` (``str``): The operation to merge those conflicting edges, which can be one of ``'mean'``, ``'sum'``, or ``'max'``. Defaults to ``'mean'``.
        """
        if len(e_list) == 0:
            return
        e_list, e_weight = self._format_edges(e_list, e_weight)
        super().add_edges(e_list, e_weight, merge_op=merge_op)
        self._clear_cache()

    def remove_edges(self, e_list: Union[List[int], List[List[int]]]):
        r"""Remove specifed edges in the directed graph.

        Args:
            ``e_list`` (``Union[List[int], List[List[int]]]``): Edges to be removed.
        """
        e_list, _ = self._format_edges(e_list)
        for src, dst in e_list:
            self._remove_edge(src, dst)
        self._clear_cache()

    def add_extra_selfloop(self):
        r"""Add extra selfloops to the directed graph.
        """
        return super().add_extra_selfloop()

    def remove_extra_selfloop(self):
        r"""Remove extra selfloops from the directed graph.
        """
        return super().remove_extra_selfloop()

    def remove_selfloop(self):
        r"""Remove all selfloops from the directed graph.
        """
        return super().remove_selfloop()

    def reverse_direction(self):
        r"""Reverse the direction of edges in directed graph.
        """
        self._raw_e_dict = {(dst, src): w for (src, dst), w in self._raw_e_dict.items()}

    def drop_edges(self, drop_rate: float, ord: str = "uniform"):
        r"""Randomly drop edges from the directed graph. This function will return a new directed graph with non-dropped edges.

        Args:
            ``drop_rate`` (``float``): The drop rate of edges.
            ``ord`` (``str``): The order of dropping edges. Currently, only ``'uniform'`` is supported. Defaults to ``uniform``.
        """
        if ord == "uniform":
            _raw_e_dict = {k: v for k, v in self._raw_e_dict.items() if random.random() > drop_rate}
            _raw_selfloop_dict = {k: v for k, v in self._raw_selfloop_dict.items() if random.random() > drop_rate}
            state_dict = {
                "num_v": self.num_v,
                "raw_e_dict": _raw_e_dict,
                "raw_selfloop_dict": _raw_selfloop_dict,
                "has_extra_selfloop": self._has_extra_selfloop,
            }
            _g = DiGraph.from_state_dict(state_dict)
            _g.to(self.device)
        else:
            raise ValueError(f"Unknown drop order: {ord}.")
        return _g

    # ==============================================================================
    # properties for representation
    @property
    def v(self) -> List[int]:
        r"""Return the list of vertices.
        """
        return super().v

    @property
    def e(self) -> Tuple[List[List[int]], List[float]]:
        r"""Return edges and their weights in the directed graph with ``(edge_list, edge_weight_list)``
        format. ``i-th`` element in the ``edge_list`` denotes ``i-th`` edge, :math:`[v_{src} \longrightarrow v_{dst}]`.
        ``i-th`` element in ``edge_weight_list`` denotes the weight of ``i-th`` edge, :math:`e_{w}`.
        The lenght of the two lists are both :math:`|\mathcal{E}|`.
        """
        return super().e

    @property
    def num_v(self) -> int:
        r"""Return the number of vertices in the directed graph.
        """
        return super().num_v

    @property
    def num_e(self) -> int:
        r"""Return the number of edges in the directed graph.
        """
        return super().num_e

    @property
    def deg_v_in(self) -> torch.Tensor:
        r"""Return the in degree list of each vertices in the directed graph.
        """
        return self.D_v_in._values().cpu().numpy().tolist()

    @property
    def deg_v_out(self) -> torch.Tensor:
        r"""Return the out degree list of each vertices in the directed graph.
        """
        return self.D_v_out._values().cpu().numpy().tolist()

    def nbr_v_in(self, v_idx: int) -> torch.Tensor:
        r"""Return a vertex list of the predecessors of the vertex ``v_idx``.

        Args:
            ``v_idx`` (``int``): The index of the vertex.
        """
        return self.N_v_in(v_idx).cpu().numpy().tolist()

    def nbr_v_out(self, v_idx: int) -> torch.Tensor:
        r"""Return a vertex list of the successors of the vertex ``v_idx``.

        Args:
            ``v_idx`` (``int``): The index of the vertex.
        """
        return self.N_v_out(v_idx).cpu().numpy().tolist()

    # =====================================================================================
    # properties for deep learning
    @property
    def vars_for_DL(self) -> List[str]:
        r"""Return a name list of available variables for deep learning in the directed graph including

        Sparse Matrices:

        .. math::
            \mathbf{A}, \mathbf{A}^\top

        Sparse Diagnal Matrices:

        .. math::
            \mathbf{D}_{v_{in}}, \mathbf{D}_{v_{out}}, \mathbf{D}_{v_{in}}^{-1}, \mathbf{D}_{v_{out}}^{-1}
        
        Vectors:

        .. math::
            \vec{e}_{src}, \vec{e}_{dst}, \vec{e}_{weight}
        """
        return [
            "A",
            "A_T",
            "D_v_in",
            "D_v_out",
            "D_v_in_neg_1",
            "D_v_out_neg_1",
            "e_src",
            "e_dst",
            "e_weight",
        ]

    @property
    def A(self) -> torch.Tensor:
        r"""Return the adjacency matrix :math:`\mathbf{A}` of the directed graph with ``torch.sparse_coo_tensor`` format. Size :math:`(|\mathcal{V}|, |\mathcal{V}|)`.
        """
        if self.cache.get("A", None) is None:
            if self.num_e == 0:
                self.cache["A"] = torch.sparse_coo_tensor(size=(self.num_v, self.num_v))
            else:
                e_list, e_weight = self.e
                self.cache["A"] = torch.sparse_coo_tensor(
                    indices=torch.tensor(e_list).t(),
                    values=torch.tensor(e_weight),
                    size=(self.num_v, self.num_v),
                    device=self.device,
                ).coalesce()
        return self.cache["A"]

    @property
    def A_T(self) -> torch.Tensor:
        r"""Return the transposed adjacency matrix :math:`\mathbf{A}^\top` of the directed graph with ``torch.sparse_coo_tensor`` format. Size :math:`(|\mathcal{V}|, |\mathcal{V}|)`.
        """
        if self.cache.get("A_T", None) is None:
            self.cache["A_T"] = self.A.t().coalesce()
        return self.cache["A_T"]

    @property
    def D_v_in(self) -> torch.Tensor:
        r"""Return the diagnal matrix of vertex in degree :math:`\mathbf{D}_{v_{in}}` with ``torch.sparse_coo_tensor`` format. Size :math:`(|\mathcal{V}|, |\mathcal{V}|)`.
        """
        if self.cache.get("D_v_in", None) is None:
            _tmp = torch.sparse.sum(self.A_T, dim=1).to_dense().clone().view(-1)
            self.cache["D_v_in"] = torch.sparse_coo_tensor(
                indices=torch.arange(0, self.num_v, device=self.device).view(1, -1).repeat(2, 1),
                values=_tmp,
                size=torch.Size([self.num_v, self.num_v]),
                device=self.device,
            ).coalesce()
        return self.cache["D_v_in"]

    @property
    def D_v_out(self) -> torch.Tensor:
        r"""Return the diagnal matrix of vertex out degree :math:`\mathbf{D}_{v_{out}}` with ``torch.sparse_coo_tensor`` format. Size :math:`(|\mathcal{V}|, |\mathcal{V}|)`.
        """
        if self.cache.get("D_v_out", None) is None:
            _tmp = torch.sparse.sum(self.A, dim=1).to_dense().clone().view(-1)
            self.cache["D_v_out"] = torch.sparse_coo_tensor(
                indices=torch.arange(0, self.num_v, device=self.device).view(1, -1).repeat(2, 1),
                values=_tmp,
                size=torch.Size([self.num_v, self.num_v]),
                device=self.device,
            ).coalesce()
        return self.cache["D_v_out"]

    @property
    def D_v_in_neg_1(self) -> torch.Tensor:
        r"""Return the nomalized diagnal matrix of vertex in degree :math:`\mathbf{D}_{v_{in}}^{-1}` with ``torch.sparse_coo_tensor`` format. Size :math:`(|\mathcal{V}|, |\mathcal{V}|)`.
        """
        if self.cache.get("D_v_in_neg_1", None) is None:
            _mat = self.D_v_in.clone()
            _val = _mat._values() ** -1
            _val[torch.isinf(_val)] = 0
            self.cache["D_v_in_neg_1"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.cache["D_v_in_neg_1"]

    @property
    def D_v_out_neg_1(self) -> torch.Tensor:
        r"""Return the nomalized diagnal matrix of vertex out degree :math:`\mathbf{D}_{v_{out}}^{-1}` with ``torch.sparse_coo_tensor`` format. Size :math:`(|\mathcal{V}|, |\mathcal{V}|)`.
        """
        if self.cache.get("D_v_out_neg_1", None) is None:
            _mat = self.D_v_out.clone()
            _val = _mat._values() ** -1
            _val[torch.isinf(_val)] = 0
            self.cache["D_v_out_neg_1"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.cache["D_v_out_neg_1"]

    def N_v_in(self, v_idx: int) -> torch.Tensor:
        r"""Return the predecessors of the vertex ``v_idx`` with ``torch.Tensor`` format.

        Args:
            ``v_idx`` (``int``): The index of the vertex.
        """
        sub_v_set = self.A_T[v_idx]._indices()[0].clone()
        return sub_v_set

    def N_v_out(self, v_idx: int) -> torch.Tensor:
        r"""Return the successors of the vertex ``v_idx`` with ``torch.Tensor`` format.

        Args:
            ``v_idx`` (``int``): The index of the vertex.
        """
        sub_v_set = self.A[v_idx]._indices()[0].clone()
        return sub_v_set

    @property
    def e_src(self) -> torch.Tensor:
        r"""Return the index vector :math:`\vec{e}_{src}` of source vertices in the directed graph with ``torch.Tensor`` format. Size :math:`(|\mathcal{E}|,)`.
        """
        return self.A._indices()[0, :].clone()

    @property
    def e_dst(self) -> torch.Tensor:
        r"""Return the index vector :math:`\vec{e}_{dst}` of destination vertices in the directed graph with ``torch.Tensor`` format. Size :math:`(|\mathcal{E}|,)`.
        """
        return self.A._indices()[1, :].clone()

    @property
    def e_weight(self) -> torch.Tensor:
        r"""Return the weight vector :math:`\vec{e}_{weight} of edges` in the directed graph with ``torch.Tensor`` format. Size :math:`(|\mathcal{E}|,)`.
        """
        return self.A._values().clone()

    # ==============================================================================
    # spectral-based convolution/smoothing

    def smoothing(self, X: torch.Tensor, L: torch.Tensor, lamb: float) -> torch.Tensor:
        return super().smoothing(X, L, lamb)

    # ==============================================================================
    # spatial-based convolution/message-passing functions
    # general message passing
    def v2v(
        self,
        X: torch.Tensor,
        aggr: str = "mean",
        e_weight: Optional[torch.Tensor] = None,
        direction: str = "dst2src",
        drop_rate: float = 0.0,
    ) -> torch.Tensor:
        r"""Message passing from vertex to vertex on the directed graph structure.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size: :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``, optional): Aggregation function for neighbor messages, which can be ``'mean'``, ``'sum'``, or ``'softmax_then_sum'``. Default: ``'mean'``.
            ``e_weight`` (``torch.Tensor``, optional): The edge weight vector. Size: :math:`(|\mathcal{E}|,)`. Defaults to ``None``.
            ``direction`` (``str``, optional): The direction of message passing. Can be ``'src2dst'`` or ``'dst2src'``. Default: ``'dst2src'``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in adjacency matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert aggr in ["mean", "sum", "softmax_then_sum",], "aggr must be one of ['mean', 'sum', 'softmax_then_sum']"
        assert direction in ["src2dst", "dst2src",], "message passing direction must be one of ['src2dst', 'dst2src']"
        if self.device != X.device:
            self.to(X.device)
        if direction == "dst2src":
            if e_weight is None:
                if drop_rate > 0.0:
                    P = sparse_dropout(self.A, drop_rate)
                else:
                    P = self.A
                # message passing
                if aggr == "mean":
                    X = torch.sparse.mm(P, X)
                    X = torch.sparse.mm(self.D_v_out_neg_1, X)
                elif aggr == "sum":
                    X = torch.sparse.mm(P, X)
                elif aggr == "softmax_then_sum":
                    P = torch.sparse.softmax(P, dim=1)
                    X = torch.sparse.mm(P, X)
                else:
                    pass
            else:
                # init adjacency matrix
                assert (
                    e_weight.shape[0] == self.e_weight.shape[0]
                ), "The size of e_weight must be equal to the size of self.e_weight."
                P = torch.sparse_coo_tensor(self.A._indices(), e_weight, self.A.shape, device=self.device).coalesce()
                if drop_rate > 0.0:
                    P = sparse_dropout(P, drop_rate)
                # message passing
                if aggr == "mean":
                    X = torch.sparse.mm(P, X)
                    D_v_in_neg_1 = torch.sparse.sum(P, dim=1).to_dense().view(-1, 1)
                    D_v_in_neg_1[torch.isinf(D_v_in_neg_1)] = 0
                    X = D_v_in_neg_1 * X
                elif aggr == "sum":
                    X = torch.sparse.mm(P, X)
                elif aggr == "softmax_then_sum":
                    P = torch.sparse.softmax(P, dim=1)
                    X = torch.sparse.mm(P, X)
                else:
                    pass
        else:  # direction == "src2dst":
            if e_weight is None:
                if drop_rate > 0.0:
                    P = sparse_dropout(self.A_T, drop_rate)
                else:
                    P = self.A_T
                # message passing
                if aggr == "mean":
                    X = torch.sparse.mm(P, X)
                    X = torch.sparse.mm(self.D_v_in_neg_1, X)
                elif aggr == "sum":
                    X = torch.sparse.mm(P, X)
                elif aggr == "softmax_then_sum":
                    P = torch.sparse.softmax(P, dim=1)
                    X = torch.sparse.mm(P, X)
                else:
                    pass
            else:
                # init adjacency matrix
                assert (
                    e_weight.shape[0] == self.e_weight.shape[0]
                ), "The size of e_weight must be equal to the size of self.e_weight."
                P = (
                    torch.sparse_coo_tensor(self.A._indices(), e_weight, self.A.shape, device=self.device)
                    .t()
                    .coalesce()
                )
                if drop_rate > 0.0:
                    P = sparse_dropout(P, drop_rate)
                # message passing
                if aggr == "mean":
                    X = torch.sparse.mm(P, X)
                    D_v_in_neg_1 = torch.sparse.sum(P, dim=1).to_dense().view(-1, 1)
                    D_v_in_neg_1[torch.isinf(D_v_in_neg_1)] = 0
                    X = D_v_in_neg_1 * X
                elif aggr == "sum":
                    X = torch.sparse.mm(P, X)
                elif aggr == "softmax_then_sum":
                    P = torch.sparse.softmax(P, dim=1)
                    X = torch.sparse.mm(P, X)
                else:
                    pass
        return X
