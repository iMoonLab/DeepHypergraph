import random
import pickle
from pathlib import Path
from copy import deepcopy
from typing import Dict, Union, Optional, List, Tuple, Any, TYPE_CHECKING

import torch

from dhg.visualization.structure.draw import draw_graph
from ..base import BaseGraph
from dhg.utils.sparse import sparse_dropout

if TYPE_CHECKING:
    from ..hypergraphs import Hypergraph


class Graph(BaseGraph):
    r""" Class for graph (undirected graph).

        Args:
            ``num_v`` (``int``): The number of vertices.
            ``e_list`` (``Union[List[int], List[List[int]]], optional``): Edge list. Defaults to ``None``.
            ``e_weight`` (``Union[float, List[float]], optional``): A list of weights for edges. Defaults to ``None``.
            ``extra_selfloop`` (``bool, optional``): Whether to add extra self-loop to the graph. Defaults to ``False``.
            ``merge_op`` (``str``): The operation to merge those conflicting edges, which can be ``'mean'``, ``'sum'`` or ``'max'``. Defaults to ``'mean'``.
            ``device`` (``torch.device``, optional): The device to store the graph. Defaults to ``torch.device('cpu')``.
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
        r"""Print the graph information.
        """
        return f"Graph(num_v={self.num_v}, num_e={self.num_e})"

    @property
    def state_dict(self) -> Dict[str, Any]:
        r"""Get the state dict of the graph.
        """
        return {
            "num_v": self.num_v,
            "raw_e_dict": self._raw_e_dict,
            "raw_selfloop_dict": self._raw_selfloop_dict,
            "has_extra_selfloop": self._has_extra_selfloop,
        }

    def save(self, file_path: Union[str, Path]):
        r"""Save the DHG's graph structure to a file.

        Args:
            ``file_path`` (``Union[str, Path]``): The file path to store the DHG's graph structure.
        """
        file_path = Path(file_path)
        assert file_path.parent.exists(), "The directory does not exist."
        data = {
            "class": "Graph",
            "state_dict": self.state_dict,
        }
        with open(file_path, "wb") as fp:
            pickle.dump(data, fp)

    @staticmethod
    def load(file_path: Union[str, Path]):
        r"""Load the DHG's graph structure from a file.

        Args:
            ``file_path`` (``Union[str, Path]``): The file path to load the DHG's graph structure.
        """
        file_path = Path(file_path)
        assert file_path.exists(), "The file does not exist."
        with open(file_path, "rb") as fp:
            data = pickle.load(fp)
        assert data["class"] == "Graph", "The file is not a DHG's graph structure."
        return Graph.from_state_dict(data["state_dict"])

    def draw(
        self,
        e_style: str = "line",
        v_label: Optional[List[str]] = None,
        v_size: Union[float, list] = 1.0,
        v_color: Union[str, list] = "r",
        v_line_width: Union[str, list] = 1.0,
        e_color: Union[str, list] = "gray",
        e_fill_color: Union[str, list] = "whitesmoke",
        e_line_width: Union[str, list] = 1.0,
        font_size: int = 1.0,
        font_family: str = "sans-serif",
        push_v_strength: float = 1.0,
        push_e_strength: float = 1.0,
        pull_e_strength: float = 1.0,
        pull_center_strength: float = 1.0,
    ):
        r"""Draw the graph structure. The supported edge styles are: ``'line'`` and ``'circle'``.

        Args:
            ``e_style`` (``str``): The edge style. The supported edge styles are: ``'line'`` and ``'circle'``. Defaults to ``'line'``.
            ``v_label`` (``list``, optional): A list of vertex labels. Defaults to ``None``.
            ``v_size`` (``Union[float, list]``): The vertex size. If ``v_size`` is a ``float``, all vertices will have the same size. If ``v_size`` is a ``list``, the size of each vertex will be set according to the corresponding element in the list. Defaults to ``1.0``.
            ``v_color`` (``Union[str, list]``): The vertex `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_. If ``v_color`` is a ``str``, all vertices will have the same color. If ``v_color`` is a ``list``, the color of each vertex will be set according to the corresponding element in the list. Defaults to ``'r'``.
            ``v_line_width`` (``Union[str, list]``): The vertex line width. If ``v_line_width`` is a ``float``, all vertices will have the same line width. If ``v_line_width`` is a ``list``, the line width of each vertex will be set according to the corresponding element in the list. Defaults to ``1.0``.
            ``e_color`` (``Union[str, list]``): The edge `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_. If ``e_color`` is a ``str``, all edges will have the same color. If ``e_color`` is a ``list``, the color of each edge will be set according to the corresponding element in the list. Defaults to ``'gray'``.
            ``e_fill_color`` (``Union[str, list]``): The edge fill color. If ``e_fill_color`` is a ``str``, all edges will have the same fill color. If ``e_fill_color`` is a ``list``, the fill color of each edge will be set according to the corresponding element in the list. Defaults to ``'whitesmoke'``. This argument is only valid when ``e_style`` is ``'circle'``.
            ``e_line_width`` (``Union[str, list]``): The edge line width. If ``e_line_width`` is a ``float``, all edges will have the same line width. If ``e_line_width`` is a ``list``, the line width of each edge will be set according to the corresponding element in the list. Defaults to ``1.0``.
            ``font_size`` (``int``): The font size. Defaults to ``1.0``.
            ``font_family`` (``str``): The font family. Defaults to ``'sans-serif'``.
            ``push_v_strength`` (``float``): The vertex push strength. Defaults to ``1.0``.
            ``push_e_strength`` (``float``): The edge push strength. Defaults to ``1.0``.
            ``pull_e_strength`` (``float``): The edge pull strength. Defaults to ``1.0``.
            ``pull_center_strength`` (``float``): The center pull strength. Defaults to ``1.0``.
        """
        draw_graph(
            self,
            e_style,
            v_label,
            v_size,
            v_color,
            v_line_width,
            e_color,
            e_fill_color,
            e_line_width,
            font_size,
            font_family,
            push_v_strength,
            push_e_strength,
            pull_e_strength,
            pull_center_strength,
        )

    def clear(self):
        r"""Remove all edges in this graph.
        """
        return super().clear()

    def clone(self):
        r"""Clone the graph.
        """
        _g = Graph(self.num_v, extra_selfloop=self._has_extra_selfloop, device=self.device)
        if self._raw_e_dict is not None:
            _g._raw_e_dict = deepcopy(self._raw_e_dict)
        if self._raw_selfloop_dict is not None:
            _g._raw_selfloop_dict = deepcopy(self._raw_selfloop_dict)
        _g.cache = deepcopy(self.cache)
        return _g

    def to(self, device: torch.device):
        r"""Move the graph to the specified device.

        Args:
            ``device`` (``torch.device``): The device to store the graph.
        """
        return super().to(device)

    # =====================================================================================
    # some construction functions

    @staticmethod
    def from_state_dict(state_dict: dict):
        r"""Load the DHG's graph structure from the state dict.

        Args:
            ``state_dict`` (``dict``): The state dict to load the DHG's graph.
        """
        _g = Graph(state_dict["num_v"], extra_selfloop=state_dict["has_extra_selfloop"])
        _g._raw_e_dict = deepcopy(state_dict["raw_e_dict"])
        _g._raw_selfloop_dict = deepcopy(state_dict["raw_selfloop_dict"])
        return _g

    @staticmethod
    def from_adj_list(
        num_v: int,
        adj_list: Union[List[List[int]], List[List[Tuple[int, float]]]],
        extra_selfloop: bool = False,
        device: torch.device = torch.device("cpu"),
    ) -> "Graph":
        r"""Construct a graph from the adjacency list. Each line in the adjacency list has two components. The first element in each line is the source vertex index, and the rest elements are the target vertex indices that connected to the source vertex.

        .. note::
            This function can only construct the unweighted graph.

        Args:
            ``num_v`` (``int``): The number of vertices.
            ``adj_list`` (``List[List[int]]``): Adjacency list.
            ``extra_selfloop`` (``bool``): Whether to add extra self-loop. Defaults to ``False``.
            ``device`` (``torch.device``): The device to store the graph. Defaults to ``torch.device("cpu")``.
        """
        e_list = []
        for line in adj_list:
            if len(line) <= 1:
                continue
            v_src = line[0]
            e_list.extend([(v_src, v_dst) for v_dst in line[1:]])
        _g = Graph(num_v, e_list, extra_selfloop=extra_selfloop, device=device)
        return _g

    @staticmethod
    def from_hypergraph_star(
        hypergraph: "Hypergraph",
        weighted: bool = False,
        merge_op: str = "sum",
        device: torch.device = torch.device("cpu"),
    ) -> "Graph":
        r"""Construct a graph from a hypergraph with star expansion refering to `Higher Order Learning with Graphs <https://homes.cs.washington.edu/~sagarwal/holg.pdf>`_ paper.

        Args:
            ``hypergraph`` (``Hypergraph``): The source hypergraph.
            ``weighted`` (``bool``, optional): Whether to construct a weighted graph. Defaults to ``False``.
            ``merge_op`` (``str``): The operation to merge those conflicting edges, which can be ``'mean'``, ``'sum'`` or ``'max'``. Defaults to ``'sum'``.
            ``device`` (``torch.device``, optional): The device to store the graph. Defaults to ``torch.device("cpu")``.
        """
        v_idx, e_idx = hypergraph.H._indices().clone()
        num_v, num_e = hypergraph.num_v, hypergraph.num_e
        fake_v_idx = e_idx + num_v
        e_list = torch.stack([v_idx, fake_v_idx]).t().cpu().numpy().tolist()
        if weighted:
            e_weight = hypergraph.H._values().clone().cpu().numpy().tolist()
            _g = Graph(num_v + num_e, e_list, e_weight, merge_op=merge_op, device=device,)
        else:
            _g = Graph(num_v + num_e, e_list, merge_op=merge_op, device=device)
        vertex_mask = torch.hstack([torch.ones(num_v), torch.zeros(num_e)]).bool().to(device)
        return _g, vertex_mask

    @staticmethod
    def from_hypergraph_clique(
        hypergraph: "Hypergraph", weighted: bool = False, miu: float = 1.0, device: torch.device = torch.device("cpu"),
    ) -> "Graph":
        r"""Construct a graph from a hypergraph with clique expansion refering to `Higher Order Learning with Graphs <https://homes.cs.washington.edu/~sagarwal/holg.pdf>`_ paper.

        Args:
            ``hypergraph`` (``Hypergraph``): The source hypergraph.
            ``weighted`` (``bool``, optional): Whether to construct a weighted graph. Defaults to ``False``.
            ``miu`` (``float``, optional): The parameter of clique expansion. Defaults to ``1.0``.
            ``device`` (``torch.device``): The device to store the graph. Defaults to ``torch.device("cpu")``.
        """
        num_v = hypergraph.num_v
        miu = 1.0
        adj = miu * hypergraph.H.mm(hypergraph.H_T).coalesce().cpu().clone()
        src_idx, dst_idx = adj._indices()
        edge_mask = src_idx < dst_idx
        edge_list = torch.stack([src_idx[edge_mask], dst_idx[edge_mask]]).t().cpu().numpy().tolist()
        if weighted:
            e_weight = adj._values()[edge_mask].numpy().tolist()
            _g = Graph(num_v, edge_list, e_weight, merge_op="sum", device=device)
        else:
            _g = Graph(num_v, edge_list, merge_op="mean", device=device)
        return _g

    @staticmethod
    def from_hypergraph_hypergcn(
        hypergraph: "Hypergraph",
        feature: torch.Tensor,
        with_mediator: bool = False,
        remove_selfloop: bool = True,
        device: torch.device = torch.device("cpu"),
    ) -> "Graph":
        r"""Construct a graph from a hypergraph with methods proposed in `HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs <https://arxiv.org/pdf/1809.02589.pdf>`_ paper .

        Args:
            ``hypergraph`` (``Hypergraph``): The source hypergraph.
            ``feature`` (``torch.Tensor``): The feature of the vertices.
            ``with_mediator`` (``str``): Whether to use mediator to transform the hyperedges to edges in the graph. Defaults to ``False``.
            ``remove_selfloop`` (``bool``): Whether to remove self-loop. Defaults to ``True``.
            ``device`` (``torch.device``): The device to store the graph. Defaults to ``torch.device("cpu")``.
        """
        num_v = hypergraph.num_v
        assert num_v == feature.shape[0], "The number of vertices in hypergraph and feature.shape[0] must be equal!"
        e_list, new_e_list, new_e_weight = hypergraph.e[0], [], []
        rv = torch.rand((feature.shape[1], 1), device=feature.device)
        for e in e_list:
            num_v_in_e = len(e)
            assert num_v_in_e >= 2, "The number of vertices in an edge must be greater than or equal to 2!"
            p = torch.mm(feature[e, :], rv).squeeze()
            v_a_idx, v_b_idx = torch.argmax(p), torch.argmin(p)
            if not with_mediator:
                new_e_list.append([e[v_a_idx], e[v_b_idx]])
                new_e_weight.append(1.0 / num_v_in_e)
            else:
                w = 1.0 / (2 * num_v_in_e - 3)
                for mid_v_idx in range(num_v_in_e):
                    if mid_v_idx != v_a_idx and mid_v_idx != v_b_idx:
                        new_e_list.append([e[v_a_idx], e[mid_v_idx]])
                        new_e_weight.append(w)
                        new_e_list.append([e[v_b_idx], e[mid_v_idx]])
                        new_e_weight.append(w)
        # remove selfloop
        if remove_selfloop:
            new_e_list = torch.tensor(new_e_list, dtype=torch.long)
            new_e_weight = torch.tensor(new_e_weight, dtype=torch.float)
            e_mask = (new_e_list[:, 0] != new_e_list[:, 1]).bool()
            new_e_list = new_e_list[e_mask].numpy().tolist()
            new_e_weight = new_e_weight[e_mask].numpy().tolist()
        _g = Graph(num_v, new_e_list, new_e_weight, merge_op="sum", device=device)
        return _g

    # =====================================================================================
    # some structure modification functions
    def add_edges(
        self,
        e_list: Union[List[int], List[List[int]]],
        e_weight: Optional[Union[float, List[float]]] = None,
        merge_op: str = "mean",
    ):
        r"""Add edges to the graph.

        Args:
            ``e_list`` (``Union[List[int], List[List[int]]]``): Edge list.
            ``e_weight`` (``Union[float, List[float]], optional``): A list of weights for edges. Defaults to ``None``.
            ``merge_op`` (``str``): The operation to merge those conflicting edges, which can be ``'mean'``, ``'sum'`` or ``'max'``. Defaults to ``'mean'``.
        """
        if len(e_list) == 0:
            return
        e_list, e_weight = self._format_edges(e_list, e_weight)
        for e, w, in zip(e_list, e_weight):
            e = sorted(list(e))
            self._add_edge(e[0], e[1], w, merge_op)
        self._clear_cache()

    def remove_edges(self, e_list: Union[List[int], List[List[int]]]):
        r"""Remove specified edges in the graph.

        Args:
            ``e_list`` (``Union[List[int], List[List[int]]]``): Edges to be removed.
        """
        e_list, _ = self._format_edges(e_list)
        for src, dst in e_list:
            if src > dst:
                src, dst = dst, src
            self._remove_edge(src, dst)
        self._clear_cache()

    def add_extra_selfloop(self):
        r"""Add extra selfloops to the graph.
        """
        return super().add_extra_selfloop()

    def remove_extra_selfloop(self):
        r"""Remove extra selfloops from the graph.
        """
        return super().remove_extra_selfloop()

    def remove_selfloop(self):
        r"""Remove all selfloops from the graph.
        """
        return super().remove_selfloop()

    def drop_edges(self, drop_rate: float, ord: str = "uniform"):
        r"""Randomly drop edges from the graph. This function will return a new graph with non-dropped edges.

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
            _g = Graph.from_state_dict(state_dict)
            _g = _g.to(self.device)
        else:
            raise ValueError(f"Unknown drop order: {ord}.")
        return _g

    # =====================================================================================
    # properties for representation
    @property
    def v(self) -> List[int]:
        r"""Return the list of vertices.
        """
        return super().v

    @property
    def e(self) -> Tuple[List[List[int]], List[float]]:
        r"""Return edges and their weights in the graph with ``(edge_list, edge_weight_list)``
        format. ``i-th`` element in the ``edge_list`` denotes ``i-th`` edge, :math:`[v_{src} \longleftrightarrow v_{dst}]`.
        ``i-th`` element in ``edge_weight_list`` denotes the weight of ``i-th`` edge, :math:`e_{w}`.
        The lenght of the two lists are both :math:`|\mathcal{E}|`.
        """
        return super().e

    @property
    def e_both_side(self) -> Tuple[List[List], List[float]]:
        r"""Return the list of edges including both directions.
        """
        if self.cache.get("e_both_side", None) is None:
            e_list, e_weight = deepcopy(self.e)
            e_list.extend([(dst, src) for src, dst in self._raw_e_dict.keys()])
            e_weight.extend(self._raw_e_dict.values())
            self.cache["e_both_side"] = e_list, e_weight
        return self.cache["e_both_side"]

    @property
    def num_v(self) -> int:
        r"""Return the number of vertices in the graph.
        """
        return super().num_v

    @property
    def num_e(self) -> int:
        r"""Return the number of edges in the graph.
        """
        return super().num_e

    @property
    def deg_v(self) -> List[int]:
        r"""Return the degree list of each vertex in the graph.
        """
        return self.D_v._values().cpu().numpy().tolist()

    def nbr_v(self, v_idx: int, hop: int = 1) -> List[int]:
        r""" Return a vertex list of the ``k``-hop neighbors of the vertex ``v_idx``.

        Args:
            ``v_idx`` (``int``): The index of the vertex.
            ``hop`` (``int``): The number of the hop.
        """
        return self.N_v(v_idx, hop).cpu().numpy().tolist()

    # =====================================================================================
    # properties for deep learning
    @property
    def vars_for_DL(self) -> List[str]:
        r"""Return a name list of available variables for deep learning in the graph including 

        Sparse Matrices:

        .. math::
            \mathbf{A}, \mathcal{L}, \mathcal{L}_{sym}, \mathcal{L}_{rw}, \mathcal{L}_{GCN}
        
        Sparse Diagonal Matrices:

        .. math::
            \mathbf{D}_v, \mathbf{D}_v^{-1}, \mathbf{D}_v^{-\frac{1}{2}}, 
        
        Vectors:

        .. math::
            \vec{e}_{src}, \vec{e}_{dst}, \vec{e}_{weight}
        """
        return [
            "A",
            "L",
            "L_sym",
            "L_rw",
            "L_GCN",
            "D_v",
            "D_v_neg_1",
            "D_v_neg_1_2",
            "e_src",
            "e_dst",
            "e_weight",
        ]

    @property
    def A(self) -> torch.Tensor:
        r"""Return the adjacency matrix :math:`\mathbf{A}` of the sample graph with ``torch.sparse_coo_tensor`` format. Size :math:`(|\mathcal{V}|, |\mathcal{V}|)`.
        """
        if self.cache.get("A", None) is None:
            if self.num_e == 0:
                self.cache["A"] = torch.sparse_coo_tensor(size=(self.num_v, self.num_v), device=self.device)
            else:
                e_list, e_weight = self.e_both_side
                self.cache["A"] = torch.sparse_coo_tensor(
                    indices=torch.tensor(e_list).t(),
                    values=torch.tensor(e_weight),
                    size=(self.num_v, self.num_v),
                    device=self.device,
                ).coalesce()
        return self.cache["A"]

    @property
    def D_v(self) -> torch.Tensor:
        r"""Return the diagnal matrix of vertex degree :math:`\mathbf{D}_v` with ``torch.sparse_coo_tensor`` format. Size :math:`(|\mathcal{V}|, |\mathcal{V}|)`.
        """
        if self.cache.get("D_v") is None:
            _tmp = torch.sparse.sum(self.A, dim=1).to_dense().clone().view(-1)
            self.cache["D_v"] = torch.sparse_coo_tensor(
                torch.arange(0, self.num_v).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([self.num_v, self.num_v]),
                device=self.device,
            ).coalesce()
        return self.cache["D_v"]

    @property
    def D_v_neg_1(self) -> torch.Tensor:
        r"""Return the nomalized diagnal matrix of vertex degree :math:`\mathbf{D}_v^{-1}` with ``torch.sparse_coo_tensor`` format. Size :math:`(|\mathcal{V}|, |\mathcal{V}|)`.
        """
        if self.cache.get("D_v_neg_1") is None:
            _mat = self.D_v.clone()
            _val = _mat._values() ** -1
            _val[torch.isinf(_val)] = 0
            self.cache["D_v_neg_1"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.cache["D_v_neg_1"]

    @property
    def D_v_neg_1_2(self,) -> torch.Tensor:
        r"""Return the nomalized diagnal matrix of vertex degree :math:`\mathbf{D}_v^{-\frac{1}{2}}` with ``torch.sparse_coo_tensor`` format. Size :math:`(|\mathcal{V}|, |\mathcal{V}|)`.
        """
        if self.cache.get("D_v_neg_1_2") is None:
            _mat = self.D_v.clone()
            _val = _mat._values() ** -0.5
            _val[torch.isinf(_val)] = 0
            self.cache["D_v_neg_1_2"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.cache["D_v_neg_1_2"]

    def N_v(self, v_idx: int, hop: int = 1) -> List[int]:
        r""" Return the ``k``-hop neighbors of the vertex ``v_idx`` with ``torch.Tensor`` format.

        Args:
            ``v_idx`` (``int``): The index of the vertex.
            ``hop`` (``int``): The number of the hop.
        """
        assert hop >= 1, "``hop`` must be a number larger than or equal to 1."
        if hop == 1:
            A_k = self.A
        else:
            if self.cache.get(f"A_{hop}") is None:
                A_1, A_k = self.A.clone(), self.A.clone()
                for _ in range(hop - 1):
                    A_k = torch.sparse.mm(A_k, A_1)
                self.cache[f"A_{hop}"] = A_k
            else:
                A_k = self.cache[f"A_{hop}"]
        sub_v_set = A_k[v_idx]._indices()[0].clone()
        return sub_v_set

    @property
    def e_src(self) -> torch.Tensor:
        r"""Return the index vector :math:`\vec{e}_{src}` of source vertices in the graph with ``torch.Tensor`` format. Size :math:`(|\mathcal{E}|,)`.
        """
        return self.A._indices()[1, :].clone()

    @property
    def e_dst(self) -> torch.Tensor:
        r"""Return the index vector :math:`\vec{e}_{dst}` of destination vertices in the graph with ``torch.Tensor`` format. Size :math:`(|\mathcal{E}|,)`.
        """
        return self.A._indices()[0, :].clone()

    @property
    def e_weight(self) -> torch.Tensor:
        r"""Return the weight vector :math:`\vec{e}_{weight}` of edges in the graph with ``torch.Tensor`` format. Size :math:`(|\mathcal{E}|,)`.
        """
        return self.A._values().clone()

    # ==============================================================================
    # spectral-based convolution/smoothing

    def smoothing(self, X: torch.Tensor, L: torch.Tensor, lamb: float) -> torch.Tensor:
        return super().smoothing(X, L, lamb)

    @property
    def L(self) -> torch.Tensor:
        r"""Return the Laplacian matrix :math:`\mathbf{L}` of the sample graph with ``torch.sparse_coo_tensor`` format. Size :math:`(|\mathcal{V}|, |\mathcal{V}|)`.
        
        .. math::
            \mathbf{L} = \mathbf{D}_v - \mathbf{A}
        """
        if self.cache.get("L") is None:
            _tmp_g = self.clone()
            _tmp_g.remove_selfloop()
            self.cache["L"] = _tmp_g.D_v - _tmp_g.A
        return self.cache["L"]

    @property
    def L_sym(self) -> torch.Tensor:
        r"""Return the symmetric Laplacian matrix :math:`\mathcal{L}_{sym}` of the graph with ``torch.sparse_coo_tensor`` format. Size :math:`(|\mathcal{V}|, |\mathcal{V}|)`.

        .. math::
            \mathcal{L}_{sym} = \mathbf{I} - \mathbf{D}_v^{-\frac{1}{2}} \mathbf{A} \mathbf{D}_v^{-\frac{1}{2}}
        """
        if self.cache.get("L_sym") is None:
            _tmp_g = self.clone()
            _tmp_g.remove_selfloop()
            _L = _tmp_g.D_v_neg_1_2.mm(_tmp_g.A).mm(_tmp_g.D_v_neg_1_2).clone()
            self.cache["L_sym"] = torch.sparse_coo_tensor(
                torch.hstack([torch.arange(0, self.num_v, device=self.device).view(1, -1).repeat(2, 1), _L._indices(),]),
                torch.hstack([torch.ones(self.num_v, device=self.device), -_L._values()]),
                torch.Size([self.num_v, self.num_v]),
                device=self.device,
            ).coalesce()
        return self.cache["L_sym"]

    @property
    def L_rw(self) -> torch.Tensor:
        r"""Return the random walk Laplacian matrix :math:`\mathcal{L}_{rw}` of the graph with ``torch.sparse_coo_tensor`` format. Size :math:`(|\mathcal{V}|, |\mathcal{V}|)`.

        .. math::
            \mathcal{L}_{rw} = \mathbf{I} - \mathbf{D}_v^{-1} \mathbf{A}
        """
        if self.cache.get("L_rw") is None:
            _tmp_g = self.clone()
            _tmp_g.remove_selfloop()
            _L = _tmp_g.D_v_neg_1.mm(_tmp_g.A).clone()
            self.cache["L_rw"] = torch.sparse_coo_tensor(
                torch.hstack([torch.arange(0, self.num_v, device=self.device).view(1, -1).repeat(2, 1), _L._indices(),]),
                torch.hstack([torch.ones(self.num_v, device=self.device), -_L._values()]),
                torch.Size([self.num_v, self.num_v]),
                device=self.device,
            ).coalesce()
        return self.cache["L_rw"]

    ## GCN Laplacian smoothing
    @property
    def L_GCN(self) -> torch.Tensor:
        r"""Return the GCN Laplacian matrix :math:`\mathcal{L}_{GCN}` of the graph with ``torch.sparse_coo_tensor`` format. Size :math:`(|\mathcal{V}|, |\mathcal{V}|)`.

        .. math::
            \mathcal{L}_{GCN} = \mathbf{\hat{D}}_v^{-\frac{1}{2}} \mathbf{\hat{A}} \mathbf{\hat{D}}_v^{-\frac{1}{2}}

        """
        if self.cache.get("L_GCN") is None:
            _tmp_g = self.clone()
            _tmp_g.add_extra_selfloop()
            self.cache["L_GCN"] = _tmp_g.D_v_neg_1_2.mm(_tmp_g.A).mm(_tmp_g.D_v_neg_1_2).clone().coalesce()
        return self.cache["L_GCN"]

    def smoothing_with_GCN(self, X: torch.Tensor, drop_rate: float = 0.0) -> torch.Tensor:
        r"""Return the smoothed feature matrix with GCN Laplacian matrix :math:`\mathcal{L}_{GCN}`.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in adjacency matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        if self.device != X.device:
            self.to(X.device)
        if drop_rate > 0.0:
            L_GCN = sparse_dropout(self.L_GCN, drop_rate)
        else:
            L_GCN = self.L_GCN
        return L_GCN.mm(X)

    # =====================================================================================
    # spatial-based convolution/message-passing
    ## general message passing functions
    def v2v(
        self, X: torch.Tensor, aggr: str = "mean", e_weight: Optional[torch.Tensor] = None, drop_rate: float = 0.0
    ):
        r"""Message passing from vertex to vertex on the graph structure.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size: :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``, optional): Aggregation function for neighbor messages, which can be ``'mean'``, ``'sum'``, or ``'softmax_then_sum'``. Default: ``'mean'``.
            ``e_weight`` (``torch.Tensor``, optional): The edge weight vector. Size: :math:`(|\mathcal{E}|,)`. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in adjacency matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert aggr in ["mean", "sum", "softmax_then_sum",], "aggr must be one of ['mean', 'sum', 'softmax_then_sum']"
        if self.device != X.device:
            self.to(X.device)
        if e_weight is None:
            if drop_rate > 0.0:
                P = sparse_dropout(self.A, drop_rate)
            else:
                P = self.A
            # message passing
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                X = torch.sparse.mm(self.D_v_neg_1, X)
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.mm(P, dim=1)
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
