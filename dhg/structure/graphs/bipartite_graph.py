import random
import pickle
from pathlib import Path
from copy import deepcopy
from typing import Union, Optional, List, Tuple, Any, Dict, TYPE_CHECKING

import torch
import numpy as np

from dhg.structure.hypergraphs import Hypergraph
from dhg.visualization.structure.draw import draw_bigraph
from ..base import BaseGraph
from dhg.utils.sparse import sparse_dropout


class BiGraph(BaseGraph):
    r""" Class for bipartite graph.

        Args:
            ``num_u`` (``int``): The Number of vertices in set :math:`\mathcal{U}`.
            ``num_v`` (``int``): The Number of vertices in set :math:`\mathcal{V}`.
            ``e_list`` (``Union[List[int], List[List[int]]], optional``): Initial edge set. Defaults to ``None``.
            ``e_weight`` (``Union[float, List[float]], optional``): A list of weights for edges. Defaults to ``None``.
            ``merge_op`` (``str``): The operation to merge those conflicting edges, which can be one of ``'mean'``, ``'sum'``, or ``'max'``. Defaults to ``'mean'``.
            ``device`` (``torch.device``, optional): The device to store the bipartite graph. Defaults to ``torch.device('cpu')``.
    """

    def __init__(
        self,
        num_u: int,
        num_v: int,
        e_list: Optional[Union[List[int], List[List[int]]]] = None,
        e_weight: Optional[Union[float, List[float]]] = None,
        merge_op: str = "mean",
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(num_v, device=device)
        self._num_u = num_u
        if e_list is not None:
            self.add_edges(e_list, e_weight, merge_op=merge_op)

    def __repr__(self) -> str:
        r"""Print the bipartite graph information.
        """
        return f"Bipartite Graph(num_u={self.num_u}, num_v={self.num_v}, num_e={self.num_e})"

    @property
    def state_dict(self) -> Dict[str, Any]:
        r"""Get the state dict of the bipartite graph.
        """
        return {
            "num_u": self.num_u,
            "num_v": self.num_v,
            "raw_e_dict": self._raw_e_dict,
        }

    def save(self, file_path: Union[str, Path]):
        r"""Save the DHG's bipartite graph structure to a file.

        Args:
            ``file_path`` (``Union[str, Path]``): The file path to store the DHG's bipartite graph structure.
        """
        file_path = Path(file_path)
        assert file_path.parent.exists(), "The directory does not exist."
        data = {
            "class": "BiGraph",
            "state_dict": self.state_dict,
        }
        with open(file_path, "wb") as fp:
            pickle.dump(data, fp)

    @staticmethod
    def load(file_path: Union[str, Path]):
        r"""Load the DHG's bipartite graph structure from a file.

        Args:
            ``file_path`` (``Union[str, Path]``): The file path to load the DHG's bipartite graph structure.
        """
        file_path = Path(file_path)
        assert file_path.exists(), "The file does not exist."
        with open(file_path, "rb") as fp:
            data = pickle.load(fp)
        assert data["class"] == "BiGraph", "The file is not a bipartite graph."
        return BiGraph.from_state_dict(data["state_dict"])

    def draw(
        self,
        e_style: str = "line",
        u_label: Optional[List[str]] = None,
        u_size: Union[float, list] = 1.0,
        u_color: Union[str, list] = "m",
        u_line_width: Union[str, list] = 1.0,
        v_label: Optional[List[str]] = None,
        v_size: Union[float, list] = 1.0,
        v_color: Union[str, list] = "r",
        v_line_width: Union[str, list] = 1.0,
        e_color: Union[str, list] = "gray",
        e_line_width: Union[str, list] = 1.0,
        u_font_size: float = 1.0,
        v_font_size: float = 1.0,
        font_family: str = "sans-serif",
        push_u_strength: float = 1.0,
        push_v_strength: float = 1.0,
        push_e_strength: float = 1.0,
        pull_e_strength: float = 1.0,
        pull_u_center_strength: float = 1.0,
        pull_v_center_strength: float = 1.0,
    ):
        r"""Draw the bipartite graph structure.
        
        Args:
            ``e_style`` (``str``): The edge style. The supported edge styles are only ``'line'``. Defaults to ``'line'``.
            ``u_label`` (``list``): The label of vertices in set :math:`\mathcal{U}`. Defaults to ``None``.
            ``u_size`` (``Union[str, list]``): The size of vertices in set :math:`\mathcal{U}`. If ``u_size`` is a ``float``, all vertices will have the same size. If ``u_size`` is a ``list``, the size of each vertex will be set according to the corresponding element in the list. Defaults to ``1.0``.
            ``u_color`` (``Union[str, list]``): The `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ of vertices in set :math:`\mathcal{U}`. If ``u_color`` is a ``str``, all vertices will have the same color. If ``u_color`` is a ``list``, the color of each vertex will be set according to the corresponding element in the list. Defaults to ``'m'``.
            ``u_line_width`` (``Union[str, list]``): The line width of vertices in set :math:`\mathcal{U}`. If ``u_line_width`` is a ``float``, all vertices will have the same line width. If ``u_line_width`` is a ``list``, the line width of each vertex will be set according to the corresponding element in the list. Defaults to ``1.0``.
            ``v_label`` (``list``): The label of vertices in set :math:`\mathcal{V}`. Defaults to ``None``.
            ``v_size`` (``Union[str, list]``): The size of vertices in set :math:`\mathcal{V}`. If ``v_size`` is a ``float``, all vertices will have the same size. If ``v_size`` is a ``list``, the size of each vertex will be set according to the corresponding element in the list. Defaults to ``1.0``.
            ``v_color`` (``Union[str, list]``): The `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ of vertices in set :math:`\mathcal{V}`. If ``v_color`` is a ``str``, all vertices will have the same color. If ``v_color`` is a ``list``, the color of each vertex will be set according to the corresponding element in the list. Defaults to ``'r'``.
            ``v_line_width`` (``Union[str, list]``): The line width of vertices in set :math:`\mathcal{V}`. If ``v_line_width`` is a ``float``, all vertices will have the same line width. If ``v_line_width`` is a ``list``, the line width of each vertex will be set according to the corresponding element in the list. Defaults to ``1.0``.
            ``e_color`` (``Union[str, list]``): The `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ of edges. If ``e_color`` is a ``str``, all edges will have the same color. If ``e_color`` is a ``list``, the color of each edge will be set according to the corresponding element in the list. Defaults to ``'gray'``.
            ``e_line_width`` (``Union[str, list]``): The line width of edges. If ``e_line_width`` is a ``float``, all edges will have the same line width. If ``e_line_width`` is a ``list``, the line width of each edge will be set according to the corresponding element in the list. Defaults to ``1.0``.
            ``u_font_size`` (``float``): The font size of vertex labels in set :math:`\mathcal{U}`. Defaults to ``1.0``.
            ``v_font_size`` (``float``): The font size of vertex labels in set :math:`\mathcal{V}`. Defaults to ``1.0``.
            ``font_family`` (``str``): The font family of vertex labels. Defaults to ``'sans-serif'``.
            ``push_u_strength`` (``float``): The strength of pushing vertices in set :math:`\mathcal{U}`. Defaults to ``1.0``.
            ``push_v_strength`` (``float``): The strength of pushing vertices in set :math:`\mathcal{V}`. Defaults to ``1.0``.
            ``push_e_strength`` (``float``): The strength of pushing edges. Defaults to ``1.0``.
            ``pull_e_strength`` (``float``): The strength of pulling edges. Defaults to ``1.0``.
            ``pull_u_center_strength`` (``float``): The strength of pulling vertices in set :math:`\mathcal{U}` to the center. Defaults to ``1.0``.
            ``pull_v_center_strength`` (``float``): The strength of pulling vertices in set :math:`\mathcal{V}` to the center. Defaults to ``1.0``.
        """
        draw_bigraph(
            self,
            e_style,
            u_label,
            u_size,
            u_color,
            u_line_width,
            v_label,
            v_size,
            v_color,
            v_line_width,
            e_color,
            e_line_width,
            u_font_size,
            v_font_size,
            font_family,
            push_u_strength,
            push_v_strength,
            push_e_strength,
            pull_e_strength,
            pull_u_center_strength,
            pull_v_center_strength,
        )

    def clear(self):
        r"""Remove all edges in the bipartite graph.
        """
        return super().clear()

    def clone(self):
        r"""Clone the bipartite graph.
        """
        _g = BiGraph(self.num_u, self.num_v, device=self.device)
        if self._raw_e_dict is not None:
            _g._raw_e_dict = deepcopy(self._raw_e_dict)
        _g.cache = deepcopy(self.cache)
        return _g

    def to(self, device: torch.device):
        r"""Move the bipartite graph to the specified device.

        Args:
            ``device`` (``torch.device``): The device to store the bipartite graph.
        """
        return super().to(device)

    # utils
    def _format_edges(
        self, e_list: Union[List[int], List[List[int]]], e_weight: Optional[Union[float, List[float]]] = None,
    ) -> Tuple[List[List[int]], List[float]]:
        r"""Check the format of input e_list, and convert raw edge list into edge list.

        .. note::
            If edges in ``e_list`` only have two elements, we will append default weight ``1`` to all edges.

        Args:
            ``e_list`` (``List[List[int]]``): Edge list should be a list of edge with pair elements.
            ``e_weight`` (``List[float]``, optional): Edge weights for each edge. Defaults to ``None``.
        """
        if e_list is None:
            return [], []
        # only one edge
        if isinstance(e_list[0], int) and len(e_list) == 2:
            e_list = [e_list]
            if e_weight is not None:
                e_weight = [e_weight]
        e_array = np.array(e_list)
        assert e_array[:, 0].max() < self.num_u, "The u_idx in e_list is out of range."
        assert e_array[:, 1].max() < self.num_v, "The v_idx in e_list is out of range."
        # complete the weight
        if e_weight is None:
            e_weight = [1.0] * len(e_list)
        return e_list, e_weight

    # =====================================================================================
    # some construction functions
    @staticmethod
    def from_state_dict(state_dict: dict):
        r"""Load the bipartite graph structure from a state dictionary.

        Args:
            ``state_dict`` (``dict``): The state dictionary to load the bipartite graph structure.
        """
        _g = BiGraph(state_dict["num_u"], state_dict["num_v"])
        _g._raw_e_dict = deepcopy(state_dict["raw_e_dict"])
        return _g

    @staticmethod
    def from_adj_list(
        num_u: int, num_v: int, adj_list: List[List[int]], device: torch.device = torch.device("cpu"),
    ) -> "BiGraph":
        r"""Construct a bipartite graph from the adjacency list. Each line in the adjacency list has two components. The first element in each line is the ``u_idx``, and the rest elements are the ``v_idx`` that connected to the ``u_idx``.

        .. note::
            This function can only construct the unweighted bipartite graph.

        Args:
            ``num_u`` (``int``): The number of vertices in set :math:`\mathcal{U}`.
            ``num_v`` (``int``): The number of vertices in set :math:`\mathcal{V}`.
            ``adj_list`` (``List[List[int]]``): Adjacency list.
            ``device`` (``torch.device``): The device to store the bipartite graph. Defaults to ``torch.device('cpu')``.
        """
        e_list = []
        for line in adj_list:
            if len(line) <= 1:
                continue
            u_idx = line[0]
            e_list.extend([[u_idx, v_idx] for v_idx in line[1:]])
        _g = BiGraph(num_u, num_v, e_list, device=device)
        return _g

    @staticmethod
    def from_hypergraph(
        hypergraph: Hypergraph,
        vertex_as_U: bool = True,
        weighted: bool = False,
        device: torch.device = torch.device("cpu"),
    ) -> "BiGraph":
        r"""Construct a bipartite graph from the hypergraph.

        Args:
            ``hypergraph`` (``Hypergraph``): Hypergraph.
            ``vertex_as_U`` (``bool``): If set to ``True``, vertices in hypergraph will be transformed to vertices in set :math:`U`, and hyperedges in hypergraph will be transformed to vertices in set :math:`V`. Otherwise, vertices in hypergraph will be transformed to vertices in set :math:`V`, and hyperedges in hypergraph will be transformed to vertices in set :math:`U`. Defaults to ``True``.
            ``weighted`` (``bool``): If set to ``True``, the bipartite graph will be constructed with weighted edges. The weight of each edge is assigned by the weight of the associated hyperedge in the original hypergraph. Defaults to ``False``.
            ``device`` (``torch.device``): The device to store the bipartite graph. Defaults to ``torch.device('cpu')``.
        """
        assert isinstance(hypergraph, Hypergraph), "The input `hypergraph` should be a instance of `Hypergraph` class."
        raw_e_list, raw_e_weight = deepcopy(hypergraph.e)
        e_weight = None
        if vertex_as_U:
            num_u, num_v = hypergraph.num_v, hypergraph.num_e
            e_list = [(v_idx, e_idx) for e_idx, v_list in enumerate(raw_e_list) for v_idx in v_list]
            if weighted:
                e_weight = [
                    e_weight for e_idx, e_weight in enumerate(raw_e_weight) for _ in range(len(raw_e_list[e_idx]))
                ]
        else:
            num_u, num_v = hypergraph.num_e, hypergraph.num_v
            e_list = [(e_idx, v_idx) for e_idx, v_list in enumerate(raw_e_list) for v_idx in v_list]
            if weighted:
                e_weight = [
                    e_weight for e_idx, e_weight in enumerate(raw_e_weight) for _ in range(len(raw_e_list[e_idx]))
                ]
        _g = BiGraph(num_u, num_v, e_list, e_weight, device=device)
        return _g

    # =====================================================================================
    # some structure modification functions
    def add_edges(
        self,
        e_list: Union[List[int], List[List[int]]],
        e_weight: Optional[Union[float, List[float]]] = None,
        merge_op: str = "mean",
    ):
        r"""Add edges to the bipartite graph.

        Args:
            ``e_list`` (``Union[List[int], List[List[int]]]``): Edge list.
            ``e_weight`` (``Union[float, List[float]], optional``): A list of weights for edges. Defaults to ``None``.
            ``merge_op`` (``str``): The operation to merge those conflicting edges, which can be one of ``'mean'``, ``'sum'``, or ``'max'``. Defaults to ``'mean'``.
        """
        if len(e_list) == 0:
            return
        e_list, e_weight = self._format_edges(e_list, e_weight)
        for (src, dst), w in zip(e_list, e_weight):
            self._add_edge(src, dst, w, merge_op)
        self._clear_cache()

    def _add_edge(self, src: int, dst: int, w: float, merge_op: str):
        r"""Add an edge to the bipartite graph.

        Args:
            ``src`` (``int``): Source vertex index.
            ``dst`` (``int``): Destination vertex index.
            ``w`` (``float``): Edge weight.
            ``merge_op`` (``str``): The merge operation for the conflicting edges.
        """
        if merge_op == "mean":
            merge_func = lambda x, y: (x + y) / 2
        elif merge_op == "max":
            merge_func = lambda x, y: max(x, y)
        elif merge_op == "sum":
            merge_func = lambda x, y: x + y
        else:
            raise ValueError(f"Unknown edge merge operation: {merge_op}.")

        if (src, dst) in self._raw_e_dict:
            self._raw_e_dict[(src, dst)] = merge_func(self._raw_e_dict[(src, dst)], w)
        else:
            self._raw_e_dict[(src, dst)] = w
        self._clear_cache()

    def remove_edges(self, e_list: Union[List[int], List[List[int]]]):
        r"""Remove specifed edges in the bipartite graph.

        Args:
            ``e_list`` (``Union[List[int], List[List[int]]]``): Edges to be removed.
        """
        e_list, _ = self._format_edges(e_list)
        for src, dst in e_list:
            self._remove_edge(src, dst)
        self._clear_cache()

    def switch_uv(self):
        r"""Switch the set :math:`\mathcal{U}` and set :math:`\mathcal{V}` of the bipartite graph, and return the vertex set switched bipartite graph.
        """
        _g = self.clone()
        _g._num_u, _g._num_v = self.num_v, self.num_u
        _g._raw_e_dict = {(v, u): w for (u, v), w in self._raw_e_dict.items()}
        _g._clear_cache()
        return _g

    def drop_edges(self, drop_rate: float, ord: str = "uniform"):
        r"""Randomly drop edges from the bipartite graph. This function will return a new bipartite graph with non-dropped edges.

        Args:
            ``drop_rate`` (``float``): The drop rate of edges.
            ``ord`` (``str``): The order of dropping edges. Currently, only ``'uniform'`` is supported. Defaults to ``uniform``.
        """
        if ord == "uniform":
            _raw_e_dict = {k: v for k, v in self._raw_e_dict.items() if random.random() > drop_rate}
            state_dict = {
                "num_u": self.num_u,
                "num_v": self.num_v,
                "raw_e_dict": _raw_e_dict,
            }
            _g = BiGraph.from_state_dict(state_dict)
            _g = _g.to(self.device)
        else:
            raise ValueError(f"Unknown drop order: {ord}.")
        return _g

    # ==============================================================================
    # properties for representation
    @property
    def u(self) -> List[int]:
        r"""Return the list of vertices in set :math:`\mathcal{U}`.
        """
        return list(range(self.num_u))

    @property
    def v(self) -> List[int]:
        r"""Return the list of vertices in set :math:`\mathcal{V}`.
        """
        return super().v

    @property
    def e(self) -> Tuple[List[List[int]], List[float]]:
        r"""Return edges and their weights in the bipartite graph with ``(edge_list, edge_weight_list)``
        format. ``i-th`` element in the ``edge_list`` denotes ``i-th`` edge, :math:`[u \longleftrightarrow v]`.
        ``i-th`` element in ``edge_weight_list`` denotes the weight of ``i-th`` edge, :math:`e_{w}`.
        The lenght of the two lists are both :math:`|\mathcal{E}|`.
        """
        return super().e

    @property
    def num_u(self) -> int:
        r"""Return the number of vertices in set :math:`\mathcal{U}`.
        """
        return self._num_u

    @property
    def num_v(self) -> int:
        r"""Return the number of vertices in set :math:`\mathcal{V}`.
        """
        return super().num_v

    @property
    def num_e(self) -> int:
        r"""Return the number of edges in the bipartite graph.
        """
        return super().num_e

    @property
    def deg_u(self) -> torch.Tensor:
        r"""Return the degree list of vertices in set :math:`\mathcal{U}`.
        """
        return self.D_u._values().cpu().numpy().tolist()

    @property
    def deg_v(self) -> torch.Tensor:
        r"""Return the degree list of vertices in set :math:`\mathcal{V}`.
        """
        return self.D_v._values().cpu().numpy().tolist()

    def nbr_v(self, u_idx: int) -> torch.Tensor:
        r"""Return a neighbor vertex list in set :math:`\mathcal{V}` of the specified vertex ``u_idx``.

        Args:
            ``u_idx`` (``int``): The index of the vertex in set :math:`\mathcal{U}`.
        """
        return self.N_v(u_idx).cpu().numpy().tolist()

    def nbr_u(self, v_idx: int) -> torch.Tensor:
        r"""Return a neighbor vertex list in set :math:`\mathcal{U}` of the specified vertex ``v_idx``.

        Args:
            ``v_idx`` (``int``): The index of the vertex in set :math:`\mathcal{V}`.
        """
        return self.N_u(v_idx).cpu().numpy().tolist()

    # =====================================================================================
    # properties for deep learning
    @property
    def vars_for_DL(self) -> List[str]:
        r"""Return a name list of available variables for deep learning in the bipartite graph including

        Sparse Matrices:

        .. math::
            \mathbf{A}, \mathbf{B}, \mathbf{B}^\top

        Sparse Diagnal Matrices:

        .. math::
            \mathbf{D}_u, \mathbf{D}_v, \mathbf{D}_u^{-1}, \mathbf{D}_v^{-1}

        Vectors:

        .. math::
            \vec{e}_{u}, \vec{e}_{v}, \vec{e}_{weight}
        """
        return [
            "A",
            "B",
            "B_T",
            "D_u",
            "D_v",
            "D_u_neg_1",
            "D_v_neg_1",
            "e_u",
            "e_v",
            "e_weight",
        ]

    @property
    def A(self) -> torch.Tensor:
        r"""Return the adjacency matrix :math:`\mathbf{A}` of the bipartite graph with ``torch.sparse_coo_tensor`` format. Size :math:`(|\mathcal{U}| + |\mathcal{V}|, |\mathcal{U}| + |\mathcal{V}|)`.
        """
        if self.cache.get("A", None) is None:
            UU = torch.sparse_coo_tensor(size=(self.num_u, self.num_u), device=self.device)
            VV = torch.sparse_coo_tensor(size=(self.num_v, self.num_v), device=self.device)
            A_up = torch.hstack([UU, self.B])
            A_down = torch.hstack([self.B_T, VV])
            self.cache["A"] = torch.vstack([A_up, A_down]).coalesce()
        return self.cache["A"]

    @property
    def B(self) -> torch.Tensor:
        r"""Return the bipartite adjacency matrix :math:`\mathbf{B}` of the bipartite graph with ``torch.sparse_coo_tensor`` format. Size :math:`(|\mathcal{U}|, |\mathcal{V}|)`.
        """
        if self.cache.get("B", None) is None:
            if self.num_e == 0:
                self.cache["B"] = torch.sparse_coo_tensor(size=(self.num_u, self.num_v))
            else:
                e_list, e_weight = self.e
                self.cache["B"] = torch.sparse_coo_tensor(
                    indices=torch.tensor(e_list).t(),
                    values=torch.tensor(e_weight),
                    size=(self.num_u, self.num_v),
                    device=self.device,
                ).coalesce()
        return self.cache["B"]

    @property
    def B_T(self) -> torch.Tensor:
        r"""Return the transposed bipartite adjacency matrix :math:`\mathbf{B}^\top` of the bipartite graph with ``torch.sparse_coo_tensor`` format. Size :math:`(|\mathcal{V}|, |\mathcal{U}|)`.
        """
        if self.cache.get("B_T", None) is None:
            self.cache["B_T"] = self.B.t().coalesce()
        return self.cache["B_T"]

    @property
    def D_u(self) -> torch.Tensor:
        r"""Return the diagnal matrix of vertex in degree :math:`\mathbf{D}_u` with ``torch.sparse_coo_tensor`` format. Size :math:`(|\mathcal{U}|, |\mathcal{U}|)`.
        """
        if self.cache.get("D_u", None) is None:
            _tmp = torch.sparse.sum(self.B, dim=1).to_dense().clone().view(-1)
            self.cache["D_u"] = torch.sparse_coo_tensor(
                indices=torch.arange(0, self.num_u, device=self.device).view(1, -1).repeat(2, 1),
                values=_tmp,
                size=torch.Size([self.num_u, self.num_u]),
                device=self.device,
            ).coalesce()
        return self.cache["D_u"]

    @property
    def D_v(self) -> torch.Tensor:
        r"""Return the diagnal matrix of vertex out degree :math:`\mathbf{D}_v` with ``torch.sparse_coo_tensor`` format. Size :math:`(|\mathcal{V}|, |\mathcal{V}|)`.
        """
        if self.cache.get("D_v", None) is None:
            _tmp = torch.sparse.sum(self.B_T, dim=1).to_dense().clone().view(-1)
            self.cache["D_v"] = torch.sparse_coo_tensor(
                indices=torch.arange(0, self.num_v, device=self.device).view(1, -1).repeat(2, 1),
                values=_tmp,
                size=torch.Size([self.num_v, self.num_v]),
                device=self.device,
            ).coalesce()
        return self.cache["D_v"]

    @property
    def D_u_neg_1(self) -> torch.Tensor:
        r"""Return the nomalized diagnal matrix of vertex in degree :math:`\mathbf{D}_u^{-1}` with ``torch.sparse_coo_tensor`` format. Size :math:`(|\mathcal{U}|, |\mathcal{U}|)`.
        """
        if self.cache.get("D_u_neg_1", None) is None:
            _mat = self.D_u.clone()
            _val = _mat._values() ** -1
            _val[torch.isinf(_val)] = 0
            self.cache["D_u_neg_1"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.cache["D_u_neg_1"]

    @property
    def D_v_neg_1(self) -> torch.Tensor:
        r"""Return the nomalized diagnal matrix of vertex out degree :math:`\mathbf{D}_v^{-1}` with ``torch.sparse_coo_tensor`` format. Size :math:`(|\mathcal{V}|, |\mathcal{V}|)`.
        """
        if self.cache.get("D_v_neg_1", None) is None:
            _mat = self.D_v.clone()
            _val = _mat._values() ** -1
            _val[torch.isinf(_val)] = 0
            self.cache["D_v_neg_1"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.cache["D_v_neg_1"]

    def N_v(self, u_idx: int) -> torch.Tensor:
        r"""Return neighbor vertices in set :math:`\mathcal{V}` of the specified vertex ``u_idx`` with ``torch.Tensor`` format.

        Args:
            ``u_idx`` (``int``): The index of the vertex.
        """
        sub_v_set = self.B[u_idx]._indices()[0].clone()
        return sub_v_set

    def N_u(self, v_idx: int) -> torch.Tensor:
        r"""Return neighbor vertices in set :math:`\mathcal{U}` of the specified vertex ``v_idx`` with ``torch.Tensor`` format.

        Args:
            ``v_idx`` (``int``): The index of the vertex.
        """
        sub_u_set = self.B_T[v_idx]._indices()[0].clone()
        return sub_u_set

    @property
    def e_u(self) -> torch.Tensor:
        r"""Return the index vector :math:`\vec{e}_{u}` of vertices in set :math:`\mathcal{U}` in the bipartite graph with ``torch.Tensor`` format. Size :math:`(|\mathcal{E}|,)`.
        """
        return self.B._indices()[0, :].clone()

    @property
    def e_v(self) -> torch.Tensor:
        r"""Return the index vector :math:`\vec{e}_{v}` of vertices in set :math:`\mathcal{V}` in the bipartite graph with ``torch.Tensor`` format. Size :math:`(|\mathcal{E}|,)`.
        """
        return self.B._indices()[1, :].clone()

    @property
    def e_weight(self) -> torch.Tensor:
        r"""Return the weight vector :math:`\vec{e}_{weight}`  of edges in the bipartite graph with ``torch.Tensor`` format. Size :math:`(|\mathcal{E}|,)`.
        """
        return self.B._values().clone()

    # ==============================================================================
    # spectral-based convolution/smoothing

    def smoothing(self, X: torch.Tensor, L: torch.Tensor, lamb: float) -> torch.Tensor:
        return super().smoothing(X, L, lamb)

    @property
    def L_GCN(self) -> torch.Tensor:
        r"""Return the GCN Laplacian matrix of the bipartite graph with ``torch.Tensor`` format. Size :math:`(|\mathcal{U}| + |\mathcal{V}|, |\mathcal{U}| + |\mathcal{V}|)`.
        """
        if self.cache.get("L_GCN", None) is None:
            selfloop_indices = torch.arange(0, self.num_u + self.num_v).view(1, -1).repeat(2, 1)
            selfloop_values = torch.ones(self.num_u + self.num_v).view(-1)
            A_ = torch.sparse_coo_tensor(
                indices=torch.hstack([self.A._indices().cpu(), selfloop_indices]),
                values=torch.hstack([self.A._values().cpu(), selfloop_values]),
                size=torch.Size([self.num_u + self.num_v, self.num_u + self.num_v]),
                device=self.device,
            ).coalesce()
            D_v_neg_1_2 = torch.sparse.sum(A_, dim=1).to_dense().view(-1) ** (-0.5)
            D_v_neg_1_2[torch.isinf(D_v_neg_1_2)] = 0
            D_v_neg_1_2 = torch.sparse_coo_tensor(
                indices=selfloop_indices,
                values=D_v_neg_1_2,
                size=torch.Size([self.num_u + self.num_v, self.num_u + self.num_v]),
                device=self.device,
            ).coalesce()
            self.cache["L_GCN"] = D_v_neg_1_2.mm(A_).mm(D_v_neg_1_2).clone().coalesce()
        return self.cache["L_GCN"]

    def smoothing_with_GCN(self, X: torch.Tensor, drop_rate: float = 0.0) -> torch.Tensor:
        r"""Return the smoothed feature matrix with GCN Laplacian matrix :math:`\mathcal{L}_{GCN}`.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix of the bipartite graph. Size :math:`(|\mathcal{U}| + |\mathcal{V}|, C)`.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in adjacency matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        if self.device != X.device:
            X = X.to(self.device)
        if drop_rate > 0.0:
            L_GCN = sparse_dropout(self.L_GCN, drop_rate)
        else:
            L_GCN = self.L_GCN
        return L_GCN.mm(X)

    # ==============================================================================
    # spatial-based convolution/message-passing functions
    # general message passing
    def u2v(
        self, X: torch.Tensor, aggr: str = "mean", e_weight: Optional[torch.Tensor] = None, drop_rate: float = 0.0
    ) -> torch.Tensor:
        r"""Message passing from vertices in set :math:`\mathcal{U}` to vertices in set :math:`\mathcal{V}` on the bipartite graph structure.

        Args:
            ``X`` (``torch.Tensor``): Feature matrix of vertices in set :math:`\mathcal{U}`. Size: :math:`(|\mathcal{U}|, C)`.
            ``aggr`` (``str``, optional): Aggregation function for neighbor messages, which can be ``'mean'``, ``'sum'``, or ``'softmax_then_sum'``. Default: ``'mean'``.
            ``e_weight`` (``torch.Tensor``, optional): The edge weight vector. Size: :math:`(|\mathcal{E}|,)`. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in adjacency matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert aggr in ["mean", "sum", "softmax_then_sum",], "aggr must be one of ['mean', 'sum', 'softmax_then_sum']"
        if self.device != X.device:
            self.to(X.device)
        if e_weight is None:
            if drop_rate > 0.0:
                P = sparse_dropout(self.B_T, drop_rate)
            else:
                P = self.B_T
            # message passing
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                X = torch.sparse.mm(self.D_v_neg_1, X)
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
            P = torch.sparse_coo_tensor(self.B._indices(), e_weight, self.B.shape, device=self.device).t().coalesce()
            if drop_rate > 0.0:
                P = sparse_dropout(P, drop_rate)
            # message passing
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                D_v_neg_1 = torch.sparse.sum(P, dim=1).to_dense().view(-1, 1)
                D_v_neg_1[torch.isinf(D_v_neg_1)] = 0
                X = D_v_neg_1 * X
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                pass
        return X

    def v2u(
        self, X: torch.Tensor, aggr: str = "mean", e_weight: Optional[torch.Tensor] = None, drop_rate: float = 0.0
    ) -> torch.Tensor:
        r"""Message passing from vertices in set :math:`\mathcal{V}` to vertices in set :math:`\mathcal{U}` on the bipartite graph structure.

        Args:
            ``X`` (``torch.Tensor``): Feature matrix of vertices in set :math:`\mathcal{V}`. Size: :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``, optional): Aggregation function for neighbor messages, which can be ``'mean'``, ``'sum'``, or ``'softmax_then_sum'``. Default: ``'mean'``.
            ``e_weight`` (``torch.Tensor``, optional): The edge weight vector. Size: :math:`(|\mathcal{E}|,)`. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in adjacency matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert aggr in ["mean", "sum", "softmax_then_sum",], "aggr must be one of ['mean', 'sum', 'softmax_then_sum']"
        if self.device != X.device:
            self.to(X.device)
        if e_weight is None:
            if drop_rate > 0.0:
                P = sparse_dropout(self.B, drop_rate)
            else:
                P = self.B
            # message passing
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                X = torch.sparse.mm(self.D_u_neg_1, X)
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
            P = torch.sparse_coo_tensor(self.B._indices(), e_weight, self.B.shape, device=self.device).coalesce()
            if drop_rate > 0.0:
                P = sparse_dropout(P, drop_rate)
            # message passing
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                D_u_neg_1 = torch.sparse.sum(P, dim=1).to_dense().view(-1, 1)
                D_u_neg_1[torch.isinf(D_u_neg_1)] = 0
                X = D_u_neg_1 * X
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                pass
        return X
