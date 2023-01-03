import random
import pickle
from pathlib import Path
from copy import deepcopy
from typing import Optional, Union, List, Tuple, Dict, Any, TYPE_CHECKING

import torch
import scipy.spatial

from dhg.structure import BaseHypergraph
from dhg.visualization.structure.draw import draw_hypergraph
from dhg.utils.sparse import sparse_dropout

if TYPE_CHECKING:
    from ..graphs import Graph, BiGraph


class Hypergraph(BaseHypergraph):
    r"""The ``Hypergraph`` class is developed for hypergraph structures.

    Args:
        ``num_v`` (``int``): The number of vertices in the hypergraph.
        ``e_list`` (``Union[List[int], List[List[int]]]``, optional): A list of hyperedges describes how the vertices point to the hyperedges. Defaults to ``None``.
        ``e_weight`` (``Union[float, List[float]]``, optional): A list of weights for hyperedges. If set to ``None``, the value ``1`` is used for all hyperedges. Defaults to ``None``.
        ``v_weight`` (``Union[List[float]]``, optional): A list of weights for vertices. If set to ``None``, the value ``1`` is used for all vertices. Defaults to ``None``.
        ``merge_op`` (``str``): The operation to merge those conflicting hyperedges in the same hyperedge group, which can be ``'mean'``, ``'sum'`` or ``'max'``. Defaults to ``'mean'``.
        ``device`` (``torch.device``, optional): The deivce to store the hypergraph. Defaults to ``torch.device('cpu')``.
    """

    def __init__(
        self,
        num_v: int,
        e_list: Optional[Union[List[int], List[List[int]]]] = None,
        e_weight: Optional[Union[float, List[float]]] = None,
        v_weight: Optional[List[float]] = None,
        merge_op: str = "mean",
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(num_v, device=device)
        # init vertex weight
        if v_weight is None:
            self._v_weight = [1.0] * self.num_v
        else:
            assert len(v_weight) == self.num_v, "The length of vertex weight is not equal to the number of vertices."
            self._v_weight = v_weight
        # init hyperedges
        if e_list is not None:
            self.add_hyperedges(e_list, e_weight, merge_op=merge_op)

    def __repr__(self) -> str:
        r"""Print the hypergraph information.
        """
        return f"Hypergraph(num_v={self.num_v}, num_e={self.num_e})"

    @property
    def state_dict(self) -> Dict[str, Any]:
        r"""Get the state dict of the hypergraph.
        """
        return {"num_v": self.num_v, "raw_groups": self._raw_groups}

    def save(self, file_path: Union[str, Path]):
        r"""Save the DHG's hypergraph structure a file.

        Args:
            ``file_path`` (``Union[str, Path]``): The file path to store the DHG's hypergraph structure.
        """
        file_path = Path(file_path)
        assert file_path.parent.exists(), "The directory does not exist."
        data = {
            "class": "Hypergraph",
            "state_dict": self.state_dict,
        }
        with open(file_path, "wb") as fp:
            pickle.dump(data, fp)

    @staticmethod
    def load(file_path: Union[str, Path]):
        r"""Load the DHG's hypergraph structure from a file.

        Args:
            ``file_path`` (``Union[str, Path]``): The file path to load the DHG's hypergraph structure.
        """
        file_path = Path(file_path)
        assert file_path.exists(), "The file does not exist."
        with open(file_path, "rb") as fp:
            data = pickle.load(fp)
        assert data["class"] == "Hypergraph", "The file is not a DHG's hypergraph file."
        return Hypergraph.from_state_dict(data["state_dict"])

    def draw(
        self,
        e_style: str = "circle",
        v_label: Optional[List[str]] = None,
        v_size: Union[float, list] = 1.0,
        v_color: Union[str, list] = "r",
        v_line_width: Union[str, list] = 1.0,
        e_color: Union[str, list] = "gray",
        e_fill_color: Union[str, list] = "whitesmoke",
        e_line_width: Union[str, list] = 1.0,
        font_size: float = 1.0,
        font_family: str = "sans-serif",
        push_v_strength: float = 1.0,
        push_e_strength: float = 1.0,
        pull_e_strength: float = 1.0,
        pull_center_strength: float = 1.0,
    ):
        r"""Draw the hypergraph structure.

        Args:
            ``e_style`` (``str``): The style of hyperedges. The available styles are only ``'circle'``. Defaults to ``'circle'``.
            ``v_label`` (``list``): The labels of vertices. Defaults to ``None``.
            ``v_size`` (``float`` or ``list``): The size of vertices. Defaults to ``1.0``.
            ``v_color`` (``str`` or ``list``): The `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ of vertices. Defaults to ``'r'``.
            ``v_line_width`` (``float`` or ``list``): The line width of vertices. Defaults to ``1.0``.
            ``e_color`` (``str`` or ``list``): The `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ of hyperedges. Defaults to ``'gray'``.
            ``e_fill_color`` (``str`` or ``list``): The fill `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ of hyperedges. Defaults to ``'whitesmoke'``.
            ``e_line_width`` (``float`` or ``list``): The line width of hyperedges. Defaults to ``1.0``.
            ``font_size`` (``float``): The font size of labels. Defaults to ``1.0``.
            ``font_family`` (``str``): The font family of labels. Defaults to ``'sans-serif'``.
            ``push_v_strength`` (``float``): The strength of pushing vertices. Defaults to ``1.0``.
            ``push_e_strength`` (``float``): The strength of pushing hyperedges. Defaults to ``1.0``.
            ``pull_e_strength`` (``float``): The strength of pulling hyperedges. Defaults to ``1.0``.
            ``pull_center_strength`` (``float``): The strength of pulling vertices to the center. Defaults to ``1.0``.
        """
        draw_hypergraph(
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
        r"""Clear all hyperedges and caches from the hypergraph.
        """
        return super().clear()

    def clone(self) -> "Hypergraph":
        r"""Return a copy of the hypergraph.
        """
        hg = Hypergraph(self.num_v, device=self.device)
        hg._raw_groups = deepcopy(self._raw_groups)
        hg.cache = deepcopy(self.cache)
        hg.group_cache = deepcopy(self.group_cache)
        return hg

    def to(self, device: torch.device):
        r"""Move the hypergraph to the specified device.

        Args:
            ``device`` (``torch.device``): The target device.
        """
        return super().to(device)

    # =====================================================================================
    # some construction functions
    @staticmethod
    def from_state_dict(state_dict: dict):
        r"""Load the hypergraph from the state dict.

        Args:
            ``state_dict`` (``dict``): The state dict to load the hypergraph.
        """
        _hg = Hypergraph(state_dict["num_v"])
        _hg._raw_groups = deepcopy(state_dict["raw_groups"])
        return _hg

    @staticmethod
    def _e_list_from_feature_kNN(features: torch.Tensor, k: int):
        r"""Construct hyperedges from the feature matrix. Each hyperedge in the hypergraph is constructed by the central vertex ans its :math:`k-1` neighbor vertices.

        Args:
            ``features`` (``torch.Tensor``): The feature matrix.
            ``k`` (``int``): The number of nearest neighbors.
        """
        features = features.cpu().numpy()
        assert features.ndim == 2, "The feature matrix should be 2-D."
        assert (
            k <= features.shape[0]
        ), "The number of nearest neighbors should be less than or equal to the number of vertices."
        tree = scipy.spatial.cKDTree(features)
        _, nbr_array = tree.query(features, k=k)
        return nbr_array.tolist()

    @staticmethod
    def from_feature_kNN(features: torch.Tensor, k: int, device: torch.device = torch.device("cpu")):
        r"""Construct the hypergraph from the feature matrix. Each hyperedge in the hypergraph is constructed by the central vertex ans its :math:`k-1` neighbor vertices.

        .. note::
            The constructed hypergraph is a k-uniform hypergraph. If the feature matrix has the size :math:`N \times C`, the number of vertices and hyperedges of the constructed hypergraph are both :math:`N`.

        Args:
            ``features`` (``torch.Tensor``): The feature matrix.
            ``k`` (``int``): The number of nearest neighbors.
            ``device`` (``torch.device``, optional): The device to store the hypergraph. Defaults to ``torch.device('cpu')``.
        """
        e_list = Hypergraph._e_list_from_feature_kNN(features, k)
        hg = Hypergraph(features.shape[0], e_list, device=device)
        return hg

    @staticmethod
    def from_graph(graph: "Graph", device: torch.device = torch.device("cpu")) -> "Hypergraph":
        r"""Construct the hypergraph from the graph. Each edge in the graph is treated as a hyperedge in the constructed hypergraph.

        .. note::
            The construsted hypergraph is a 2-uniform hypergraph, and has the same number of vertices and edges/hyperedges as the graph.

        Args:
            ``graph`` (``Graph``): The graph to construct the hypergraph.
            ``device`` (``torch.device``, optional): The device to store the hypergraph. Defaults to ``torch.device('cpu')``.
        """
        e_list, e_weight = graph.e
        hg = Hypergraph(graph.num_v, e_list, e_weight=e_weight, device=device)
        return hg

    @staticmethod
    def _e_list_from_graph_kHop(graph: "Graph", k: int, only_kHop: bool = False,) -> List[tuple]:
        r"""Construct the hyperedge list from the graph by k-Hop neighbors. Each hyperedge in the hypergraph is constructed by the central vertex and its :math:`k`-Hop neighbor vertices.

        .. note::
            If the graph have :math:`|\mathcal{V}|` vertices, the constructed hypergraph will have :math:`|\mathcal{V}|` vertices and equal to or less than :math:`|\mathcal{V}|` hyperedges.

        Args:
            ``graph`` (``Graph``): The graph to construct the hypergraph.
            ``k`` (``int``): The number of hop neighbors.
            ``only_kHop`` (``bool``, optional): If set to ``True``, only the central vertex and its :math:`k`-th Hop neighbors are used to construct the hyperedges. By default, the constructed hyperedge will include the central vertex and its [ :math:`1`-th, :math:`2`-th, :math:`\cdots`, :math:`k`-th ] Hop neighbors. Defaults to ``False``.
        """
        assert k >= 1, "The number of hop neighbors should be larger than or equal to 1."
        A_1, A_k = graph.A.clone(), graph.A.clone()
        A_history = []
        for _ in range(k - 1):
            A_k = torch.sparse.mm(A_k, A_1)
            if not only_kHop:
                A_history.append(A_k.clone())
        if not only_kHop:
            A_k = A_1
            for A_ in A_history:
                A_k = A_k + A_
        e_list = [
            tuple(set([v_idx] + A_k[v_idx]._indices().cpu().squeeze(0).tolist())) for v_idx in range(graph.num_v)
        ]
        return e_list

    @staticmethod
    def from_graph_kHop(
        graph: "Graph", k: int, only_kHop: bool = False, device: torch.device = torch.device("cpu"),
    ) -> "Hypergraph":
        r"""Construct the hypergraph from the graph by k-Hop neighbors. Each hyperedge in the hypergraph is constructed by the central vertex and its :math:`k`-Hop neighbor vertices.

        .. note::
            If the graph have :math:`|\mathcal{V}|` vertices, the constructed hypergraph will have :math:`|\mathcal{V}|` vertices and equal to or less than :math:`|\mathcal{V}|` hyperedges.

        Args:
            ``graph`` (``Graph``): The graph to construct the hypergraph.
            ``k`` (``int``): The number of hop neighbors.
            ``only_kHop`` (``bool``): If set to ``True``, only the central vertex and its :math:`k`-th Hop neighbors are used to construct the hyperedges. By default, the constructed hyperedge will include the central vertex and its [ :math:`1`-th, :math:`2`-th, :math:`\cdots`, :math:`k`-th ] Hop neighbors. Defaults to ``False``.
            ``device`` (``torch.device``, optional): The device to store the hypergraph. Defaults to ``torch.device('cpu')``.
        """
        e_list = Hypergraph._e_list_from_graph_kHop(graph, k, only_kHop)
        hg = Hypergraph(graph.num_v, e_list, device=device)
        return hg

    @staticmethod
    def _e_list_from_bigraph(bigraph: "BiGraph", U_as_vertex: bool = True) -> List[tuple]:
        r"""Construct hyperedges from the bipartite graph.

        Args:
            ``bigraph`` (``BiGraph``): The bipartite graph to construct the hypergraph.
            ``U_as_vertex`` (``bool``, optional): If set to ``True``, vertices in set :math:`\mathcal{U}` and set :math:`\mathcal{V}`
             will be treated as vertices and hyperedges in the constructed hypergraph, respectively.
             If set to ``False``, vertices in set :math:`\mathcal{U}` and set :math:`\mathcal{V}`
             will be treated as hyperedges and vertices in the constructed hypergraph, respectively. Defaults to ``True``.
        """
        e_list = []
        if U_as_vertex:
            for v in range(bigraph.num_v):
                u_list = bigraph.nbr_u(v)
                if len(u_list) > 0:
                    e_list.append(u_list)
        else:
            for u in range(bigraph.num_u):
                v_list = bigraph.nbr_v(u)
                if len(v_list) > 0:
                    e_list.append(v_list)
        return e_list

    @staticmethod
    def from_bigraph(
        bigraph: "BiGraph", U_as_vertex: bool = True, device: torch.device = torch.device("cpu")
    ) -> "Hypergraph":
        r"""Construct the hypergraph from the bipartite graph.

        Args:
            ``bigraph`` (``BiGraph``): The bipartite graph to construct the hypergraph.
            ``U_as_vertex`` (``bool``, optional): If set to ``True``, vertices in set :math:`\mathcal{U}` and set :math:`\mathcal{V}`
             will be treated as vertices and hyperedges in the constructed hypergraph, respectively.
             If set to ``False``, vertices in set :math:`\mathcal{U}` and set :math:`\mathcal{V}`
             will be treated as hyperedges and vertices in the constructed hypergraph, respectively. Defaults to ``True``.
            ``device`` (``torch.device``, optional): The device to store the hypergraph. Defaults to ``torch.device('cpu')``.
        """
        e_list = Hypergraph._e_list_from_bigraph(bigraph, U_as_vertex)
        if U_as_vertex:
            hg = Hypergraph(bigraph.num_u, e_list, device=device)
        else:
            hg = Hypergraph(bigraph.num_v, e_list, device=device)
        return hg

    # =====================================================================================
    # some structure modification functions
    def add_hyperedges(
        self,
        e_list: Union[List[int], List[List[int]]],
        e_weight: Optional[Union[float, List[float]]] = None,
        merge_op: str = "mean",
        group_name: str = "main",
    ):
        r"""Add hyperedges to the hypergraph. If the ``group_name`` is not specified, the hyperedges will be added to the default ``main`` hyperedge group.

        Args:
            ``num_v`` (``int``): The number of vertices in the hypergraph.
            ``e_list`` (``Union[List[int], List[List[int]]]``): A list of hyperedges describes how the vertices point to the hyperedges.
            ``e_weight`` (``Union[float, List[float]]``, optional): A list of weights for hyperedges. If set to ``None``, the value ``1`` is used for all hyperedges. Defaults to ``None``.
            ``merge_op`` (``str``): The merge operation for the conflicting hyperedges. The possible values are ``"mean"``, ``"sum"``, and ``"max"``. Defaults to ``"mean"``.
            ``group_name`` (``str``, optional): The target hyperedge group to add these hyperedges. Defaults to the ``main`` hyperedge group.
        """
        e_list = self._format_e_list(e_list)
        if e_weight is None:
            e_weight = [1.0] * len(e_list)
        elif type(e_weight) in (int, float):
            e_weight = [e_weight]
        elif type(e_weight) is list:
            pass
        else:
            raise TypeError(f"The type of e_weight should be float or list, but got {type(e_weight)}")
        assert len(e_list) == len(e_weight), "The number of hyperedges and the number of weights are not equal."

        for _idx in range(len(e_list)):
            self._add_hyperedge(
                self._hyperedge_code(e_list[_idx], e_list[_idx]), {"w_e": float(e_weight[_idx])}, merge_op, group_name,
            )
        self._clear_cache(group_name)

    def add_hyperedges_from_feature_kNN(self, feature: torch.Tensor, k: int, group_name: str = "main"):
        r"""Add hyperedges from the feature matrix by k-NN. Each hyperedge is constructed by the central vertex and its :math:`k`-Nearest Neighbor vertices.

        Args:
            ``features`` (``torch.Tensor``): The feature matrix.
            ``k`` (``int``): The number of nearest neighbors.
            ``group_name`` (``str``, optional): The target hyperedge group to add these hyperedges. Defaults to the ``main`` hyperedge group.
        """
        assert (
            feature.shape[0] == self.num_v
        ), "The number of vertices in the feature matrix is not equal to the number of vertices in the hypergraph."
        e_list = Hypergraph._e_list_from_feature_kNN(feature, k)
        self.add_hyperedges(e_list, group_name=group_name)

    def add_hyperedges_from_graph(self, graph: "Graph", group_name: str = "main"):
        r"""Add hyperedges from edges in the graph. Each edge in the graph is treated as a hyperedge.

        Args:
            ``graph`` (``Graph``): The graph to join the hypergraph.
            ``group_name`` (``str``, optional): The target hyperedge group to add these hyperedges. Defaults to the ``main`` hyperedge group.
        """
        assert self.num_v == graph.num_v, "The number of vertices in the hypergraph and the graph are not equal."
        e_list, e_weight = graph.e
        self.add_hyperedges(e_list, e_weight=e_weight, group_name=group_name)

    def add_hyperedges_from_graph_kHop(
        self, graph: "Graph", k: int, only_kHop: bool = False, group_name: str = "main"
    ):
        r"""Add hyperedges from vertices and its k-Hop neighbors in the graph. Each hyperedge in the hypergraph is constructed by the central vertex and its :math:`k`-Hop neighbor vertices.

        .. note::
            If the graph have :math:`|\mathcal{V}|` vertices, the constructed hypergraph will have :math:`|\mathcal{V}|` vertices and equal to or less than :math:`|\mathcal{V}|` hyperedges.

        Args:
            ``graph`` (``Graph``): The graph to join the hypergraph.
            ``k`` (``int``): The number of hop neighbors.
            ``only_kHop`` (``bool``): If set to ``True``, only the central vertex and its :math:`k`-th Hop neighbors are used to construct the hyperedges. By default, the constructed hyperedge will include the central vertex and its [ :math:`1`-th, :math:`2`-th, :math:`\cdots`, :math:`k`-th ] Hop neighbors. Defaults to ``False``.
            ``group_name`` (``str``, optional): The target hyperedge group to add these hyperedges. Defaults to the ``main`` hyperedge group.
        """
        assert self.num_v == graph.num_v, "The number of vertices in the hypergraph and the graph are not equal."
        e_list = Hypergraph._e_list_from_graph_kHop(graph, k, only_kHop=only_kHop)
        self.add_hyperedges(e_list, group_name=group_name)

    def add_hyperedges_from_bigraph(self, bigraph: "BiGraph", U_as_vertex: bool = False, group_name: str = "main"):
        r"""Add hyperedges from the bipartite graph.

        Args:
            ``bigraph`` (``BiGraph``): The bigraph to join the hypergraph.
            ``U_as_vertex`` (``bool``): If set to ``True``, vertices in set :math:`\mathcal{U}` and set :math:`\mathcal{V}`
             will be treated as vertices and hyperedges in the constructed hypergraph, respectively.
             If set to ``False``, vertices in set :math:`\mathcal{U}` and set :math:`\mathcal{V}`
             will be treated as hyperedges and vertices in the constructed hypergraph, respectively. Defaults to ``True``.
            ``group_name`` (``str``, optional): The target hyperedge group to add these hyperedges. Defaults to the ``main`` hyperedge group.
        """
        if U_as_vertex:
            assert (
                self.num_v == bigraph.num_u
            ), "The number of vertices in the hypergraph and the number of vertices in set U of the bipartite graph are not equal."
        else:
            assert (
                self.num_v == bigraph.num_v
            ), "The number of vertices in the hypergraph and the number of vertices in set V of the bipartite graph are not equal."
        e_list = Hypergraph._e_list_from_bigraph(bigraph, U_as_vertex=U_as_vertex)
        self.add_hyperedges(e_list, group_name=group_name)

    def remove_hyperedges(
        self, e_list: Union[List[int], List[List[int]]], group_name: Optional[str] = None,
    ):
        r"""Remove the specified hyperedges from the hypergraph.

        Args:
            ``e_list`` (``Union[List[int], List[List[int]]]``): A list of hyperedges describes how the vertices point to the hyperedges.
            ``group_name`` (``str``, optional): Remove these hyperedges from the specified hyperedge group. If not specified, the function will
                remove those hyperedges from all hyperedge groups. Defaults to the ``None``.
        """
        assert (
            group_name is None or group_name in self.group_names
        ), "The specified group_name is not in existing hyperedge groups."
        e_list = self._format_e_list(e_list)
        if group_name is None:
            for _idx in range(len(e_list)):
                e_code = self._hyperedge_code(e_list[_idx], e_list[_idx])
                for name in self.group_names:
                    self._raw_groups[name].pop(e_code, None)
        else:
            for _idx in range(len(e_list)):
                e_code = self._hyperedge_code(e_list[_idx], e_list[_idx])
                self._raw_groups[group_name].pop(e_code, None)
        self._clear_cache(group_name)

    def remove_group(self, group_name: str):
        r"""Remove the specified hyperedge group from the hypergraph.

        Args:
            ``group_name`` (``str``): The name of the hyperedge group to remove.
        """
        self._raw_groups.pop(group_name, None)
        self._clear_cache(group_name)

    def drop_hyperedges(self, drop_rate: float, ord="uniform"):
        r"""Randomly drop hyperedges from the hypergraph. This function will return a new hypergraph with non-dropped hyperedges.

        Args:
            ``drop_rate`` (``float``): The drop rate of hyperedges.
            ``ord`` (``str``): The order of dropping edges. Currently, only ``'uniform'`` is supported. Defaults to ``uniform``.
        """
        if ord == "uniform":
            _raw_groups = {}
            for name in self.group_names:
                _raw_groups[name] = {k: v for k, v in self._raw_groups[name].items() if random.random() > drop_rate}
            state_dict = {
                "num_v": self.num_v,
                "raw_groups": _raw_groups,
            }
            _hg = Hypergraph.from_state_dict(state_dict)
            _hg = _hg.to(self.device)
        else:
            raise ValueError(f"Unkonwn drop order: {ord}.")
        return _hg

    def drop_hyperedges_of_group(self, group_name: str, drop_rate: float, ord="uniform"):
        r"""Randomly drop hyperedges from the specified hyperedge group. This function will return a new hypergraph with non-dropped hyperedges.

        Args:
            ``group_name`` (``str``): The name of the hyperedge group.
            ``drop_rate`` (``float``): The drop rate of hyperedges.
            ``ord`` (``str``): The order of dropping edges. Currently, only ``'uniform'`` is supported. Defaults to ``uniform``.
        """
        if ord == "uniform":
            _raw_groups = {}
            for name in self.group_names:
                if name == group_name:
                    _raw_groups[name] = {
                        k: v for k, v in self._raw_groups[name].items() if random.random() > drop_rate
                    }
                else:
                    _raw_groups[name] = self._raw_groups[name]
            state_dict = {
                "num_v": self.num_v,
                "raw_groups": _raw_groups,
            }
            _hg = Hypergraph.from_state_dict(state_dict)
            _hg = _hg.to(self.device)
        else:
            raise ValueError(f"Unkonwn drop order: {ord}.")
        return _hg

    # =====================================================================================
    # properties for representation
    @property
    def v(self) -> List[int]:
        r"""Return the list of vertices.
        """
        return super().v
    
    @property
    def v_weight(self) -> List[float]:
        r"""Return the list of vertex weights.
        """
        return self._v_weight

    @property
    def e(self) -> Tuple[List[List[int]], List[float]]:
        r"""Return all hyperedges and weights in the hypergraph.
        """
        if self.cache.get("e", None) is None:
            e_list, e_weight = [], []
            for name in self.group_names:
                _e = self.e_of_group(name)
                e_list.extend(_e[0])
                e_weight.extend(_e[1])
            self.cache["e"] = (e_list, e_weight)
        return self.cache["e"]

    def e_of_group(self, group_name: str) -> Tuple[List[List[int]], List[float]]:
        r"""Return all hyperedges and weights of the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("e", None) is None:
            e_list = [e_code[0] for e_code in self._raw_groups[group_name].keys()]
            e_weight = [e_content["w_e"] for e_content in self._raw_groups[group_name].values()]
            self.group_cache[group_name]["e"] = (e_list, e_weight)
        return self.group_cache[group_name]["e"]

    @property
    def num_v(self) -> int:
        r"""Return the number of vertices in the hypergraph.
        """
        return super().num_v

    @property
    def num_e(self) -> int:
        r"""Return the number of hyperedges in the hypergraph.
        """
        return super().num_e

    def num_e_of_group(self, group_name: str) -> int:
        r"""Return the number of hyperedges of the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        return super().num_e_of_group(group_name)

    @property
    def deg_v(self) -> List[int]:
        r"""Return the degree list of each vertex.
        """
        return self.D_v._values().cpu().view(-1).numpy().tolist()

    def deg_v_of_group(self, group_name: str) -> List[int]:
        r"""Return the degree list of each vertex of the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.D_v_of_group(group_name)._values().cpu().view(-1).numpy().tolist()

    @property
    def deg_e(self) -> List[int]:
        r"""Return the degree list of each hyperedge.
        """
        return self.D_e._values().cpu().view(-1).numpy().tolist()

    def deg_e_of_group(self, group_name: str) -> List[int]:
        r"""Return the degree list of each hyperedge of the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.D_e_of_group(group_name)._values().cpu().view(-1).numpy().tolist()

    def nbr_e(self, v_idx: int) -> List[int]:
        r"""Return the neighbor hyperedge list of the specified vertex.

        Args:
            ``v_idx`` (``int``): The index of the vertex.
        """
        return self.N_e(v_idx).cpu().numpy().tolist()

    def nbr_e_of_group(self, v_idx: int, group_name: str) -> List[int]:
        r"""Return the neighbor hyperedge list of the specified vertex of the specified hyperedge group.

        Args:
            ``v_idx`` (``int``): The index of the vertex.
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.N_e_of_group(v_idx, group_name).cpu().numpy().tolist()

    def nbr_v(self, e_idx: int) -> List[int]:
        r"""Return the neighbor vertex list of the specified hyperedge.

        Args:
            ``e_idx`` (``int``): The index of the hyperedge.
        """
        return self.N_v(e_idx).cpu().numpy().tolist()

    def nbr_v_of_group(self, e_idx: int, group_name: str) -> List[int]:
        r"""Return the neighbor vertex list of the specified hyperedge of the specified hyperedge group.

        Args:
            ``e_idx`` (``int``): The index of the hyperedge.
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.N_v_of_group(e_idx, group_name).cpu().numpy().tolist()

    @property
    def num_groups(self) -> int:
        r"""Return the number of hyperedge groups in the hypergraph.
        """
        return super().num_groups

    @property
    def group_names(self) -> List[str]:
        r"""Return the names of all hyperedge groups in the hypergraph.
        """
        return super().group_names

    # =====================================================================================
    # properties for deep learning
    @property
    def vars_for_DL(self) -> List[str]:
        r"""Return a name list of available variables for deep learning in the hypergraph including

        Sparse Matrices:

        .. math::
            \mathbf{H}, \mathbf{H}^\top, \mathcal{L}_{sym}, \mathcal{L}_{rw} \mathcal{L}_{HGNN},

        Sparse Diagnal Matrices:

        .. math::
            \mathbf{W}_v, \mathbf{W}_e, \mathbf{D}_v, \mathbf{D}_v^{-1}, \mathbf{D}_v^{-\frac{1}{2}}, \mathbf{D}_e, \mathbf{D}_e^{-1},

        Vectors:

        .. math::
            \overrightarrow{v2e}_{src}, \overrightarrow{v2e}_{dst}, \overrightarrow{v2e}_{weight},\\
            \overrightarrow{e2v}_{src}, \overrightarrow{e2v}_{dst}, \overrightarrow{e2v}_{weight}

        """
        return [
            "H",
            "H_T",
            "L_sym",
            "L_rw",
            "L_HGNN",
            "W_v",
            "W_e",
            "D_v",
            "D_v_neg_1",
            "D_v_neg_1_2",
            "D_e",
            "D_e_neg_1",
            "v2e_src",
            "v2e_dst",
            "v2e_weight" "e2v_src",
            "e2v_dst",
            "e2v_weight",
        ]

    @property
    def v2e_src(self) -> torch.Tensor:
        r"""Return the source vertex index vector :math:`\overrightarrow{v2e}_{src}` of the connections (vertices point to hyperedges) in the hypergraph.
        """
        return self.H_T._indices()[1].clone()

    def v2e_src_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the source vertex index vector :math:`\overrightarrow{v2e}_{src}` of the connections (vertices point to hyperedges) in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_T_of_group(group_name)._indices()[1].clone()

    @property
    def v2e_dst(self) -> torch.Tensor:
        r"""Return the destination hyperedge index vector :math:`\overrightarrow{v2e}_{dst}` of the connections (vertices point to hyperedges) in the hypergraph.
        """
        return self.H_T._indices()[0].clone()

    def v2e_dst_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the destination hyperedge index vector :math:`\overrightarrow{v2e}_{dst}` of the connections (vertices point to hyperedges) in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_T_of_group(group_name)._indices()[0].clone()

    @property
    def v2e_weight(self) -> torch.Tensor:
        r"""Return the weight vector :math:`\overrightarrow{v2e}_{weight}` of the connections (vertices point to hyperedges) in the hypergraph.
        """
        return self.H_T._values().clone()

    def v2e_weight_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the weight vector :math:`\overrightarrow{v2e}_{weight}` of the connections (vertices point to hyperedges) in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_T_of_group(group_name)._values().clone()

    @property
    def e2v_src(self) -> torch.Tensor:
        r"""Return the source hyperedge index vector :math:`\overrightarrow{e2v}_{src}` of the connections (hyperedges point to vertices) in the hypergraph.
        """
        return self.H._indices()[1].clone()

    def e2v_src_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the source hyperedge index vector :math:`\overrightarrow{e2v}_{src}` of the connections (hyperedges point to vertices) in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_of_group(group_name)._indices()[1].clone()

    @property
    def e2v_dst(self) -> torch.Tensor:
        r"""Return the destination vertex index vector :math:`\overrightarrow{e2v}_{dst}` of the connections (hyperedges point to vertices) in the hypergraph.
        """
        return self.H._indices()[0].clone()

    def e2v_dst_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the destination vertex index vector :math:`\overrightarrow{e2v}_{dst}` of the connections (hyperedges point to vertices) in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_of_group(group_name)._indices()[0].clone()

    @property
    def e2v_weight(self) -> torch.Tensor:
        r"""Return the weight vector :math:`\overrightarrow{e2v}_{weight}` of the connections (hyperedges point to vertices) in the hypergraph.
        """
        return self.H._values().clone()

    def e2v_weight_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the weight vector :math:`\overrightarrow{e2v}_{weight}` of the connections (hyperedges point to vertices) in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return self.H_of_group(group_name)._values().clone()

    @property
    def H(self) -> torch.Tensor:
        r"""Return the hypergraph incidence matrix :math:`\mathbf{H}` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("H") is None:
            self.cache["H"] = self.H_v2e
        return self.cache["H"]

    def H_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the hypergraph incidence matrix :math:`\mathbf{H}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("H") is None:
            self.group_cache[group_name]["H"] = self.H_v2e_of_group(group_name)
        return self.group_cache[group_name]["H"]

    @property
    def H_T(self) -> torch.Tensor:
        r"""Return the transpose of the hypergraph incidence matrix :math:`\mathbf{H}^\top` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("H_T") is None:
            self.cache["H_T"] = self.H.t()
        return self.cache["H_T"]

    def H_T_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the transpose of the hypergraph incidence matrix :math:`\mathbf{H}^\top` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("H_T") is None:
            self.group_cache[group_name]["H_T"] = self.H_of_group(group_name).t()
        return self.group_cache[group_name]["H_T"]
    
    @property
    def W_v(self) -> torch.Tensor:
        r"""Return the weight matrix :math:`\mathbf{W}_v` of vertices with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("W_v") is None:
            _tmp = torch.Tensor(self.v_weight)
            _num_v = _tmp.size(0)
            self.cache["W_v"] = torch.sparse_coo_tensor(
                torch.arange(0, _num_v, device=self.device).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([_num_v, _num_v]),
                device=self.device,
            ).coalesce()
        return self.cache["W_v"]

    @property
    def W_e(self) -> torch.Tensor:
        r"""Return the weight matrix :math:`\mathbf{W}_e` of hyperedges with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("W_e") is None:
            _tmp = [self.W_e_of_group(name)._values().clone() for name in self.group_names]
            _tmp = torch.cat(_tmp, dim=0).view(-1)
            _num_e = _tmp.size(0)
            self.cache["W_e"] = torch.sparse_coo_tensor(
                torch.arange(0, _num_e, device=self.device).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([_num_e, _num_e]),
                device=self.device,
            ).coalesce()
        return self.cache["W_e"]

    def W_e_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the weight matrix :math:`\mathbf{W}_e` of hyperedges of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("W_e") is None:
            _tmp = self._fetch_W_of_group(group_name).view(-1)
            _num_e = _tmp.size(0)
            self.group_cache[group_name]["W_e"] = torch.sparse_coo_tensor(
                torch.arange(0, _num_e, device=self.device).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([_num_e, _num_e]),
                device=self.device,
            ).coalesce()
        return self.group_cache[group_name]["W_e"]

    @property
    def D_v(self) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("D_v") is None:
            _tmp = [self.D_v_of_group(name)._values().clone() for name in self.group_names]
            _tmp = torch.vstack(_tmp).sum(dim=0).view(-1)
            self.cache["D_v"] = torch.sparse_coo_tensor(
                torch.arange(0, self.num_v, device=self.device).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([self.num_v, self.num_v]),
                device=self.device,
            ).coalesce()
        return self.cache["D_v"]

    def D_v_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("D_v") is None:
            H = self.H_of_group(group_name).clone()
            w_e = self.W_e_of_group(group_name)._values().clone()
            val = w_e[H._indices()[1]] * H._values()
            H_ = torch.sparse_coo_tensor(H._indices(), val, size=H.shape, device=self.device).coalesce()
            _tmp = torch.sparse.sum(H_, dim=1).to_dense().clone().view(-1)
            _num_v = _tmp.size(0)
            self.group_cache[group_name]["D_v"] = torch.sparse_coo_tensor(
                torch.arange(0, _num_v, device=self.device).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([_num_v, _num_v]),
                device=self.device,
            ).coalesce()
        return self.group_cache[group_name]["D_v"]

    @property
    def D_v_neg_1(self) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v^{-1}` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("D_v_neg_1") is None:
            _mat = self.D_v.clone()
            _val = _mat._values() ** -1
            _val[torch.isinf(_val)] = 0
            self.cache["D_v_neg_1"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.cache["D_v_neg_1"]

    def D_v_neg_1_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v^{-1}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("D_v_neg_1") is None:
            _mat = self.D_v_of_group(group_name).clone()
            _val = _mat._values() ** -1
            _val[torch.isinf(_val)] = 0
            self.group_cache[group_name]["D_v_neg_1"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.group_cache[group_name]["D_v_neg_1"]

    @property
    def D_v_neg_1_2(self) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v^{-\frac{1}{2}}` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("D_v_neg_1_2") is None:
            _mat = self.D_v.clone()
            _val = _mat._values() ** -0.5
            _val[torch.isinf(_val)] = 0
            self.cache["D_v_neg_1_2"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.cache["D_v_neg_1_2"]

    def D_v_neg_1_2_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the vertex degree matrix :math:`\mathbf{D}_v^{-\frac{1}{2}}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("D_v_neg_1_2") is None:
            _mat = self.D_v_of_group(group_name).clone()
            _val = _mat._values() ** -0.5
            _val[torch.isinf(_val)] = 0
            self.group_cache[group_name]["D_v_neg_1_2"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.group_cache[group_name]["D_v_neg_1_2"]

    @property
    def D_e(self) -> torch.Tensor:
        r"""Return the hyperedge degree matrix :math:`\mathbf{D}_e` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("D_e") is None:
            _tmp = [self.D_e_of_group(name)._values().clone() for name in self.group_names]
            _tmp = torch.cat(_tmp, dim=0).view(-1)
            _num_e = _tmp.size(0)
            self.cache["D_e"] = torch.sparse_coo_tensor(
                torch.arange(0, _num_e, device=self.device).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([_num_e, _num_e]),
                device=self.device,
            ).coalesce()
        return self.cache["D_e"]

    def D_e_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the hyperedge degree matrix :math:`\mathbf{D}_e` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("D_e") is None:
            _tmp = torch.sparse.sum(self.H_T_of_group(group_name), dim=1).to_dense().clone().view(-1)
            _num_e = _tmp.size(0)
            self.group_cache[group_name]["D_e"] = torch.sparse_coo_tensor(
                torch.arange(0, _num_e, device=self.device).view(1, -1).repeat(2, 1),
                _tmp,
                torch.Size([_num_e, _num_e]),
                device=self.device,
            ).coalesce()
        return self.group_cache[group_name]["D_e"]

    @property
    def D_e_neg_1(self) -> torch.Tensor:
        r"""Return the hyperedge degree matrix :math:`\mathbf{D}_e^{-1}` with ``torch.sparse_coo_tensor`` format.
        """
        if self.cache.get("D_e_neg_1") is None:
            _mat = self.D_e.clone()
            _val = _mat._values() ** -1
            _val[torch.isinf(_val)] = 0
            self.cache["D_e_neg_1"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.cache["D_e_neg_1"]

    def D_e_neg_1_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the hyperedge degree matrix :math:`\mathbf{D}_e^{-1}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("D_e_neg_1") is None:
            _mat = self.D_e_of_group(group_name).clone()
            _val = _mat._values() ** -1
            _val[torch.isinf(_val)] = 0
            self.group_cache[group_name]["D_e_neg_1"] = torch.sparse_coo_tensor(
                _mat._indices(), _val, _mat.size(), device=self.device
            ).coalesce()
        return self.group_cache[group_name]["D_e_neg_1"]

    def N_e(self, v_idx: int) -> torch.Tensor:
        r"""Return the neighbor hyperedges of the specified vertex with ``torch.Tensor`` format.

        .. note::
            The ``v_idx`` must be in the range of [0, :attr:`num_v`).

        Args:
            ``v_idx`` (``int``): The index of the vertex.
        """
        assert v_idx < self.num_v
        _tmp, e_bias = [], 0
        for name in self.group_names:
            _tmp.append(self.N_e_of_group(v_idx, name) + e_bias)
            e_bias += self.num_e_of_group(name)
        return torch.cat(_tmp, dim=0)

    def N_e_of_group(self, v_idx: int, group_name: str) -> torch.Tensor:
        r"""Return the neighbor hyperedges of the specified vertex of the specified hyperedge group with ``torch.Tensor`` format.

        .. note::
            The ``v_idx`` must be in the range of [0, :attr:`num_v`).

        Args:
            ``v_idx`` (``int``): The index of the vertex.
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        assert v_idx < self.num_v
        e_indices = self.H_of_group(group_name)[v_idx]._indices()[0]
        return e_indices.clone()

    def N_v(self, e_idx: int) -> torch.Tensor:
        r"""Return the neighbor vertices of the specified hyperedge with ``torch.Tensor`` format.

        .. note::
            The ``e_idx`` must be in the range of [0, :attr:`num_e`).

        Args:
            ``e_idx`` (``int``): The index of the hyperedge.
        """
        assert e_idx < self.num_e
        for name in self.group_names:
            if e_idx < self.num_e_of_group(name):
                return self.N_v_of_group(e_idx, name)
            else:
                e_idx -= self.num_e_of_group(name)

    def N_v_of_group(self, e_idx: int, group_name: str) -> torch.Tensor:
        r"""Return the neighbor vertices of the specified hyperedge of the specified hyperedge group with ``torch.Tensor`` format.

        .. note::
            The ``e_idx`` must be in the range of [0, :func:`num_e_of_group`).

        Args:
            ``e_idx`` (``int``): The index of the hyperedge.
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        assert e_idx < self.num_e_of_group(group_name)
        v_indices = self.H_T_of_group(group_name)[e_idx]._indices()[0]
        return v_indices.clone()

    # =====================================================================================
    # spectral-based convolution/smoothing
    def smoothing(self, X: torch.Tensor, L: torch.Tensor, lamb: float) -> torch.Tensor:
        return super().smoothing(X, L, lamb)

    @property
    def L_sym(self) -> torch.Tensor:
        r"""Return the symmetric Laplacian matrix :math:`\mathcal{L}_{sym}` of the hypergraph with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{sym} = \mathbf{I} - \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}}
        """
        if self.cache.get("L_sym") is None:
            L_HGNN = self.L_HGNN.clone()
            self.cache["L_sym"] = torch.sparse_coo_tensor(
                torch.hstack([torch.arange(0, self.num_v, device=self.device).view(1, -1).repeat(2, 1), L_HGNN._indices(),]),
                torch.hstack([torch.ones(self.num_v, device=self.device), -L_HGNN._values()]),
                torch.Size([self.num_v, self.num_v]),
                device=self.device,
            ).coalesce()
        return self.cache["L_sym"]

    def L_sym_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the symmetric Laplacian matrix :math:`\mathcal{L}_{sym}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{sym} = \mathbf{I} - \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}}

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("L_sym") is None:
            L_HGNN = self.L_HGNN_of_group(group_name).clone()
            self.group_cache[group_name]["L_sym"] = torch.sparse_coo_tensor(
                torch.hstack([torch.arange(0, self.num_v, device=self.device).view(1, -1).repeat(2, 1), L_HGNN._indices(),]),
                torch.hstack([torch.ones(self.num_v, device=self.device), -L_HGNN._values()]),
                torch.Size([self.num_v, self.num_v]),
                device=self.device,
            ).coalesce()
        return self.group_cache[group_name]["L_sym"]

    @property
    def L_rw(self) -> torch.Tensor:
        r"""Return the random walk Laplacian matrix :math:`\mathcal{L}_{rw}` of the hypergraph with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{rw} = \mathbf{I} - \mathbf{D}_v^{-1} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top
        """
        if self.cache.get("L_rw") is None:
            _tmp = self.D_v_neg_1.mm(self.H).mm(self.W_e).mm(self.D_e_neg_1).mm(self.H_T)
            self.cache["L_rw"] = (
                torch.sparse_coo_tensor(
                    torch.hstack([torch.arange(0, self.num_v, device=self.device).view(1, -1).repeat(2, 1), _tmp._indices(),]),
                    torch.hstack([torch.ones(self.num_v, device=self.device), -_tmp._values()]),
                    torch.Size([self.num_v, self.num_v]),
                    device=self.device,
                )
                .coalesce()
                .clone()
            )
        return self.cache["L_rw"]

    def L_rw_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the random walk Laplacian matrix :math:`\mathcal{L}_{rw}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{rw} = \mathbf{I} - \mathbf{D}_v^{-1} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("L_rw") is None:
            _tmp = (
                self.D_v_neg_1_of_group(group_name)
                .mm(self.H_of_group(group_name))
                .mm(self.W_e_of_group(group_name),)
                .mm(self.D_e_neg_1_of_group(group_name),)
                .mm(self.H_T_of_group(group_name),)
            )
            self.group_cache[group_name]["L_rw"] = (
                torch.sparse_coo_tensor(
                    torch.hstack([torch.arange(0, self.num_v, device=self.device).view(1, -1).repeat(2, 1), _tmp._indices(),]),
                    torch.hstack([torch.ones(self.num_v, device=self.device), -_tmp._values()]),
                    torch.Size([self.num_v, self.num_v]),
                    device=self.device,
                )
                .coalesce()
                .clone()
            )
        return self.group_cache[group_name]["L_rw"]

    ## HGNN Laplacian smoothing
    @property
    def L_HGNN(self) -> torch.Tensor:
        r"""Return the HGNN Laplacian matrix :math:`\mathcal{L}_{HGNN}` of the hypergraph with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{HGNN} = \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}}
        """
        if self.cache.get("L_HGNN") is None:
            _tmp = self.D_v_neg_1_2.mm(self.H).mm(self.W_e).mm(self.D_e_neg_1).mm(self.H_T,).mm(self.D_v_neg_1_2)
            self.cache["L_HGNN"] = _tmp.coalesce()
        return self.cache["L_HGNN"]

    def L_HGNN_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the HGNN Laplacian matrix :math:`\mathcal{L}_{HGNN}` of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        .. math::
            \mathcal{L}_{HGNN} = \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}}

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("L_HGNN") is None:
            _tmp = (
                self.D_v_neg_1_2_of_group(group_name)
                .mm(self.H_of_group(group_name))
                .mm(self.W_e_of_group(group_name))
                .mm(self.D_e_neg_1_of_group(group_name),)
                .mm(self.H_T_of_group(group_name),)
                .mm(self.D_v_neg_1_2_of_group(group_name),)
            )
            self.group_cache[group_name]["L_HGNN"] = _tmp.coalesce()
        return self.group_cache[group_name]["L_HGNN"]

    def smoothing_with_HGNN(self, X: torch.Tensor, drop_rate: float = 0.0) -> torch.Tensor:
        r"""Return the smoothed feature matrix with the HGNN Laplacian matrix :math:`\mathcal{L}_{HGNN}`.

            .. math::
                \mathbf{X} = \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}} \mathbf{X}

        Args:
            ``X`` (``torch.Tensor``): The feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
    """
        if self.device != X.device:
            X = X.to(self.device)
        if drop_rate > 0.0:
            L_HGNN = sparse_dropout(self.L_HGNN, drop_rate)
        else:
            L_HGNN = self.L_HGNN
        return L_HGNN.mm(X)

    def smoothing_with_HGNN_of_group(self, group_name: str, X: torch.Tensor, drop_rate: float = 0.0) -> torch.Tensor:
        r"""Return the smoothed feature matrix with the HGNN Laplacian matrix :math:`\mathcal{L}_{HGNN}`.

            .. math::
                \mathbf{X} = \mathbf{D}_v^{-\frac{1}{2}} \mathbf{H} \mathbf{W}_e \mathbf{D}_e^{-1} \mathbf{H}^\top \mathbf{D}_v^{-\frac{1}{2}} \mathbf{X}

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
            ``X`` (``torch.Tensor``): The feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.device != X.device:
            X = X.to(self.device)
        if drop_rate > 0.0:
            L_HGNN = sparse_dropout(self.L_HGNN_of_group(group_name), drop_rate)
        else:
            L_HGNN = self.L_HGNN_of_group(group_name)
        return L_HGNN.mm(X)

    # =====================================================================================
    # spatial-based convolution/message-passing
    ## general message passing functions
    def v2e_aggregation(
        self, X: torch.Tensor, aggr: str = "mean", v2e_weight: Optional[torch.Tensor] = None, drop_rate: float = 0.0
    ):
        r"""Message aggretation step of ``vertices to hyperedges``.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert aggr in ["mean", "sum", "softmax_then_sum"]
        if self.device != X.device:
            self.to(X.device)
        if v2e_weight is None:
            if drop_rate > 0.0:
                P = sparse_dropout(self.H_T, drop_rate)
            else:
                P = self.H_T
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                X = torch.sparse.mm(self.D_e_neg_1, X)
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method {aggr}.")
        else:
            # init message path
            assert (
                v2e_weight.shape[0] == self.v2e_weight.shape[0]
            ), "The size of v2e_weight must be equal to the size of self.v2e_weight."
            P = torch.sparse_coo_tensor(self.H_T._indices(), v2e_weight, self.H_T.shape, device=self.device)
            if drop_rate > 0.0:
                P = sparse_dropout(P, drop_rate)
            # message passing
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                D_e_neg_1 = torch.sparse.sum(P, dim=1).to_dense().view(-1, 1)
                D_e_neg_1[torch.isinf(D_e_neg_1)] = 0
                X = D_e_neg_1 * X
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method {aggr}.")
        return X

    def v2e_aggregation_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        v2e_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ):
        r"""Message aggregation step of ``vertices to hyperedges`` in specified hyperedge group.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        assert aggr in ["mean", "sum", "softmax_then_sum"]
        if self.device != X.device:
            self.to(X.device)
        if v2e_weight is None:
            if drop_rate > 0.0:
                P = sparse_dropout(self.H_T_of_group(group_name), drop_rate)
            else:
                P = self.H_T_of_group(group_name)
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                X = torch.sparse.mm(self.D_e_neg_1_of_group(group_name), X)
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method {aggr}.")
        else:
            # init message path
            assert (
                v2e_weight.shape[0] == self.v2e_weight_of_group(group_name).shape[0]
            ), f"The size of v2e_weight must be equal to the size of self.v2e_weight_of_group('{group_name}')."
            P = torch.sparse_coo_tensor(
                self.H_T_of_group(group_name)._indices(),
                v2e_weight,
                self.H_T_of_group(group_name).shape,
                device=self.device,
            )
            if drop_rate > 0.0:
                P = sparse_dropout(P, drop_rate)
            # message passing
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                D_e_neg_1 = torch.sparse.sum(P, dim=1).to_dense().view(-1, 1)
                D_e_neg_1[torch.isinf(D_e_neg_1)] = 0
                X = D_e_neg_1 * X
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method {aggr}.")
        return X

    def v2e_update(self, X: torch.Tensor, e_weight: Optional[torch.Tensor] = None):
        r"""Message update step of ``vertices to hyperedges``.

        Args:
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """
        if self.device != X.device:
            self.to(X.device)
        if e_weight is None:
            X = torch.sparse.mm(self.W_e, X)
        else:
            e_weight = e_weight.view(-1, 1)
            assert e_weight.shape[0] == self.num_e, "The size of e_weight must be equal to the size of self.num_e."
            X = e_weight * X
        return X

    def v2e_update_of_group(self, group_name: str, X: torch.Tensor, e_weight: Optional[torch.Tensor] = None):
        r"""Message update step of ``vertices to hyperedges`` in specified hyperedge group.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.device != X.device:
            self.to(X.device)
        if e_weight is None:
            X = torch.sparse.mm(self.W_e_of_group(group_name), X)
        else:
            e_weight = e_weight.view(-1, 1)
            assert e_weight.shape[0] == self.num_e_of_group(
                group_name
            ), f"The size of e_weight must be equal to the size of self.num_e_of_group('{group_name}')."
            X = e_weight * X
        return X

    def v2e(
        self,
        X: torch.Tensor,
        aggr: str = "mean",
        v2e_weight: Optional[torch.Tensor] = None,
        e_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ):
        r"""Message passing of ``vertices to hyperedges``. The combination of ``v2e_aggregation`` and ``v2e_update``.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        X = self.v2e_aggregation(X, aggr, v2e_weight, drop_rate=drop_rate)
        X = self.v2e_update(X, e_weight)
        return X

    def v2e_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        v2e_weight: Optional[torch.Tensor] = None,
        e_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ):
        r"""Message passing of ``vertices to hyperedges`` in specified hyperedge group. The combination of ``e2v_aggregation_of_group`` and ``e2v_update_of_group``.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        X = self.v2e_aggregation_of_group(group_name, X, aggr, v2e_weight, drop_rate=drop_rate)
        X = self.v2e_update_of_group(group_name, X, e_weight)
        return X

    def e2v_aggregation(
        self, X: torch.Tensor, aggr: str = "mean", e2v_weight: Optional[torch.Tensor] = None, drop_rate: float = 0.0
    ):
        r"""Message aggregation step of ``hyperedges to vertices``.

        Args:
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert aggr in ["mean", "sum", "softmax_then_sum"]
        if self.device != X.device:
            self.to(X.device)
        if e2v_weight is None:
            if drop_rate > 0.0:
                P = sparse_dropout(self.H, drop_rate)
            else:
                P = self.H
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                X = torch.sparse.mm(self.D_v_neg_1, X)
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method: {aggr}")
        else:
            # init message path
            assert (
                e2v_weight.shape[0] == self.e2v_weight.shape[0]
            ), "The size of e2v_weight must be equal to the size of self.e2v_weight."
            P = torch.sparse_coo_tensor(self.H._indices(), e2v_weight, self.H.shape, device=self.device)
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
                raise ValueError(f"Unknown aggregation method: {aggr}")
        return X

    def e2v_aggregation_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        e2v_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ):
        r"""Message aggregation step of ``hyperedges to vertices`` in specified hyperedge group.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        assert aggr in ["mean", "sum", "softmax_then_sum"]
        if self.device != X.device:
            self.to(X.device)
        if e2v_weight is None:
            if drop_rate > 0.0:
                P = sparse_dropout(self.H_of_group(group_name), drop_rate)
            else:
                P = self.H_of_group(group_name)
            if aggr == "mean":
                X = torch.sparse.mm(P, X)
                X = torch.sparse.mm(self.D_v_neg_1_of_group(group_name), X)
            elif aggr == "sum":
                X = torch.sparse.mm(P, X)
            elif aggr == "softmax_then_sum":
                P = torch.sparse.softmax(P, dim=1)
                X = torch.sparse.mm(P, X)
            else:
                raise ValueError(f"Unknown aggregation method: {aggr}")
        else:
            # init message path
            assert (
                e2v_weight.shape[0] == self.e2v_weight_of_group(group_name).shape[0]
            ), f"The size of e2v_weight must be equal to the size of self.e2v_weight_of_group('{group_name}')."
            P = torch.sparse_coo_tensor(
                self.H_of_group(group_name)._indices(),
                e2v_weight,
                self.H_of_group(group_name).shape,
                device=self.device,
            )
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
                raise ValueError(f"Unknown aggregation method: {aggr}")
        return X

    def e2v_update(self, X: torch.Tensor):
        r"""Message update step of ``hyperedges to vertices``.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
        """
        if self.device != X.device:
            self.to(X.device)
        return X

    def e2v_update_of_group(self, group_name: str, X: torch.Tensor):
        r"""Message update step of ``hyperedges to vertices`` in specified hyperedge group.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.device != X.device:
            self.to(X.device)
        return X

    def e2v(
        self, X: torch.Tensor, aggr: str = "mean", e2v_weight: Optional[torch.Tensor] = None, drop_rate: float = 0.0,
    ):
        r"""Message passing of ``hyperedges to vertices``. The combination of ``e2v_aggregation`` and ``e2v_update``.

        Args:
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        X = self.e2v_aggregation(X, aggr, e2v_weight, drop_rate=drop_rate)
        X = self.e2v_update(X)
        return X

    def e2v_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        e2v_weight: Optional[torch.Tensor] = None,
        drop_rate: float = 0.0,
    ):
        r"""Message passing of ``hyperedges to vertices`` in specified hyperedge group. The combination of ``e2v_aggregation_of_group`` and ``e2v_update_of_group``.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        X = self.e2v_aggregation_of_group(group_name, X, aggr, e2v_weight, drop_rate=drop_rate)
        X = self.e2v_update_of_group(group_name, X)
        return X

    def v2v(
        self,
        X: torch.Tensor,
        aggr: str = "mean",
        drop_rate: float = 0.0,
        v2e_aggr: Optional[str] = None,
        v2e_weight: Optional[torch.Tensor] = None,
        v2e_drop_rate: Optional[float] = None,
        e_weight: Optional[torch.Tensor] = None,
        e2v_aggr: Optional[str] = None,
        e2v_weight: Optional[torch.Tensor] = None,
        e2v_drop_rate: Optional[float] = None,
    ):
        r"""Message passing of ``vertices to vertices``. The combination of ``v2e`` and ``e2v``.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, this ``aggr`` will be used to both ``v2e`` and ``e2v``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
            ``v2e_aggr`` (``str``, optional): The aggregation method for hyperedges to vertices. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, it will override the ``aggr`` in ``e2v``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``v2e_drop_rate`` (``float``, optional): Dropout rate for hyperedges to vertices. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. If specified, it will override the ``drop_rate`` in ``e2v``. Default: ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e2v_aggr`` (``str``, optional): The aggregation method for vertices to hyperedges. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, it will override the ``aggr`` in ``v2e``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e2v_drop_rate`` (``float``, optional): Dropout rate for vertices to hyperedges. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. If specified, it will override the ``drop_rate`` in ``v2e``. Default: ``None``.
        """
        if v2e_aggr is None:
            v2e_aggr = aggr
        if e2v_aggr is None:
            e2v_aggr = aggr
        if v2e_drop_rate is None:
            v2e_drop_rate = drop_rate
        if e2v_drop_rate is None:
            e2v_drop_rate = drop_rate
        X = self.v2e(X, v2e_aggr, v2e_weight, e_weight, drop_rate=v2e_drop_rate)
        X = self.e2v(X, e2v_aggr, e2v_weight, drop_rate=e2v_drop_rate)
        return X

    def v2v_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        drop_rate: float = 0.0,
        v2e_aggr: Optional[str] = None,
        v2e_weight: Optional[torch.Tensor] = None,
        v2e_drop_rate: Optional[float] = None,
        e_weight: Optional[torch.Tensor] = None,
        e2v_aggr: Optional[str] = None,
        e2v_weight: Optional[torch.Tensor] = None,
        e2v_drop_rate: Optional[float] = None,
    ):
        r"""Message passing of ``vertices to vertices`` in specified hyperedge group. The combination of ``v2e_of_group`` and ``e2v_of_group``.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, this ``aggr`` will be used to both ``v2e_of_group`` and ``e2v_of_group``.
            ``drop_rate`` (``float``): Dropout rate. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. Default: ``0.0``.
            ``v2e_aggr`` (``str``, optional): The aggregation method for hyperedges to vertices. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, it will override the ``aggr`` in ``e2v_of_group``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``v2e_drop_rate`` (``float``, optional): Dropout rate for hyperedges to vertices. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. If specified, it will override the ``drop_rate`` in ``e2v_of_group``. Default: ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e2v_aggr`` (``str``, optional): The aggregation method for vertices to hyperedges. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, it will override the ``aggr`` in ``v2e_of_group``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e2v_drop_rate`` (``float``, optional): Dropout rate for vertices to hyperedges. Randomly dropout the connections in incidence matrix with probability ``drop_rate``. If specified, it will override the ``drop_rate`` in ``v2e_of_group``. Default: ``None``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if v2e_aggr is None:
            v2e_aggr = aggr
        if e2v_aggr is None:
            e2v_aggr = aggr
        if v2e_drop_rate is None:
            v2e_drop_rate = drop_rate
        if e2v_drop_rate is None:
            e2v_drop_rate = drop_rate
        X = self.v2e_of_group(group_name, X, v2e_aggr, v2e_weight, e_weight, drop_rate=v2e_drop_rate)
        X = self.e2v_of_group(group_name, X, e2v_aggr, e2v_weight, drop_rate=e2v_drop_rate)
        return X
