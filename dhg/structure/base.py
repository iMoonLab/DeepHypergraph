import abc
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Union, Optional, List, Tuple, Dict, Any

import torch
import numpy as np


def load_structure(file_path: Union[str, Path]):
    r"""Load a DHG's structure from a file. The supported structure includes: ``Graph``, ``DiGraph``, ``BiGraph``, ``Hypergraph``.
    
    Args: 
        ``file_path`` (``Union[str, Path]``): The file path to load the DHG's structure.
    """
    import dhg
    import pickle as pkl

    file_path = Path(file_path)
    assert file_path.exists(), f"{file_path} does not exist"
    with open(file_path, "rb") as f:
        data = pkl.load(f)
    class_name, state_dict = data["class"], data["state_dict"]
    structure_class = getattr(dhg, class_name)
    structure = structure_class.from_state_dict(state_dict)
    return structure


class BaseGraph:
    r"""The `BaseGraph` class is the base class for all graph structures.

        Args:
            ``num_v`` (``int``): The number of vertices.
            ``e_list`` (``Union[List[int], List[List[int]]], optional``): Edge list. Defaults to ``None``.
            ``e_weight`` (``Union[float, List[float]], optional``): A list of weights for edges. Defaults to ``None``.
            ``extra_selfloop`` (``bool``, optional): Whether to add extra self-loop to the graph. Defaults to ``False``.
            ``device`` (``torch.device``, optional): The device to store the graph. Defaults to ``torch.device('cpu')``.
    """

    def __init__(
        self,
        num_v: int,
        e_list: Optional[Union[List[int], List[List[int]]]] = None,
        e_weight: Optional[Union[float, List[float]]] = None,
        extra_selfloop: bool = False,
        device: torch.device = torch.device("cpu"),
    ):
        assert isinstance(num_v, int) and num_v > 0, "num_v should be a positive integer"
        self.clear()
        self.device = device
        self._num_v = num_v
        self._has_extra_selfloop = extra_selfloop

    @abc.abstractmethod
    def __repr__(self) -> str:
        r"""Print the graph information.
        """

    @property
    @abc.abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        r"""Get the state dict of the graph.
        """

    @abc.abstractmethod
    def save(self, file_path: Union[str, Path]):
        r"""Save the DHG's graph structure to a file.

        Args:
            ``file_path`` (``Union[str, Path]``): The file path to store the DHG's graph structure with DHG format.
        """

    @abc.abstractstaticmethod
    def load(file_path: Union[str, Path]):
        r"""Load the DHG's graph structure from a file.

        Args:
            ``file_path`` (``Union[str, Path]``): The file path to load the DHG's graph structure from.
        """

    @abc.abstractmethod
    def draw(self, **kwargs):
        r"""Draw the structure.
        """

    def clear(self):
        r"""Remove all edges and caches from the graph.
        """
        self._raw_e_dict = {}  # { (src, dst): w }
        self._raw_selfloop_dict = {}  # { src: w }
        self._has_extra_selfloop = False
        self._clear_cache()

    def _clear_cache(self):
        r"""Clear the cache.
        """
        self.cache = {}

    @abc.abstractmethod
    def clone(self):
        r"""Clone the graph.
        """

    def to(self, device: torch.device):
        r"""Move the graph to the specified device.

        Args:
            ``device`` (``torch.device``): The device to store the graph.
        """
        self.device = device
        for v in self.vars_for_DL:
            if self.cache.get(v, None) is not None:
                self.cache[v] = self.cache[v].to(device)
        return self

    # utils
    def _format_edges(
        self, e_list: Union[List[int], List[List[int]]], e_weight: Optional[Union[float, List[float]]] = None,
    ) -> Tuple[List[List[int]], List[float]]:
        r"""Check the format of input e_list, and convert raw edge list into edge tensor.

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
        assert np.array(e_list).max() < self._num_v, "Vertex index out of range"
        # complete the weight
        if e_weight is None:
            e_weight = [1.0] * len(e_list)
        return e_list, e_weight

    # =====================================================================================
    # some construction functions

    @staticmethod
    @abc.abstractmethod
    def from_state_dict(state_dict: dict):
        r"""Load the DHG's graph structure from the state dict.

        Args:
            ``state_dict`` (``dict``): The state dict to load the DHG's graph.
        """

    @staticmethod
    @abc.abstractmethod
    def from_adj_list(num_v: int, adj_list: List[List[int]], extra_selfloop: bool = False,) -> "BaseGraph":
        r"""Construct a graph from the adjacency list. Each line in the adjacency list has two components. The first element in each line is the source vertex index, and the rest elements are the target vertex indices that connected to the source vertex.

        .. note::
            This function can only construct unweighted graph.

        Args:
            ``num_v`` (``int``): The number of vertices.
            ``adj_list`` (``List[List[int]]``): Adjacency list.
            ``extra_selfloop`` (``bool``): Whether to add extra self-loop. Defaults to ``False``.

        Returns:
            ``BaseGraph``: The constructed graph.
        """

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
            ``e_weight`` (``Union[float, List[float]]``, optional): A list of weights for edges. Defaults to None.
            ``merge_op`` (``str``): The merge operation for the conflicting edges. If set to ``None``, the default merge operation specified in Graph Construction is used. Defaults to ``None``.
        """
        for (src, dst), w in zip(e_list, e_weight):
            self._add_edge(src, dst, w, merge_op)
        self._clear_cache()

    def _add_edge(self, src: int, dst: int, w: float, merge_op: str):
        r"""Add an edge to the graph.

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

        # self loop
        if src == dst:
            if src in self._raw_selfloop_dict:
                self._raw_selfloop_dict[src] = merge_func(self._raw_selfloop_dict[src], w)
            else:
                self._raw_selfloop_dict[src] = w
        else:
            if (src, dst) in self._raw_e_dict:
                self._raw_e_dict[(src, dst)] = merge_func(self._raw_e_dict[(src, dst)], w)
            else:
                self._raw_e_dict[(src, dst)] = w
        self._clear_cache()

    @abc.abstractmethod
    def remove_edges(self, e_list: Union[List[int], List[List[int]]]):
        r"""Remove edges from the graph.

        Args:
            ``e_list`` (``List[List[int]]``): Edge list to be removed.
        """

    def _remove_edge(self, v_a: int, v_b: int):
        r"""Reove the edge (v_a, v_b) from the graph.

        Args:
            ``v_a`` (``int``): The first vertex of the edge.
            ``v_b`` (``int``): The second vertex of the edge.
        """
        if v_a == v_b and v_a in self._raw_selfloop_dict:
            del self._raw_selfloop_dict[v_a]
        elif (v_a, v_b) in self._raw_e_dict:
            del self._raw_e_dict[(v_a, v_b)]
        self._clear_cache()

    def add_extra_selfloop(self):
        r"""Add extra selfloops to the graph.
        """
        self._has_extra_selfloop = True
        self._clear_cache()

    def remove_extra_selfloop(self):
        r"""Remove extra selfloops from the graph.
        """
        self._has_extra_selfloop = False
        self._clear_cache()

    def remove_selfloop(self):
        r"""Remove all selfloops from the graph.
        """
        self._raw_selfloop_dict.clear()
        self.remove_extra_selfloop()
        self._clear_cache()

    @abc.abstractmethod
    def drop_edges(self, drop_rate: float, ord: str = "uniform"):
        r"""Randomly drop edges from the graph. This function will return a new graph with non-dropped edges.

        Args:
            ``drop_rate`` (``float``): The drop rate of edges.
            ``ord`` (``str``): The order of dropping edges. Currently, only ``'uniform'`` is supported. Defaults to ``uniform``.
        """

    # =====================================================================================
    # properties for representation
    @property
    def v(self) -> List[int]:
        r"""Return vertices of the graph.
        """
        return list(range(self.num_v))

    @property
    def e(self) -> Tuple[List[List[int]], List[float]]:
        r"""Return the edge list and weight list in the graph.
        """
        if self.cache.get("e", None) is None:
            e_list = [(src_idx, dst_idx) for src_idx, dst_idx in self._raw_e_dict.keys()]
            w_list = list(self._raw_e_dict.values())
            e_list.extend([(v_idx, v_idx) for v_idx in self._raw_selfloop_dict.keys()])
            w_list.extend(list(self._raw_selfloop_dict.values()))
            if self._has_extra_selfloop:
                e_list.extend((v_idx, v_idx) for v_idx in range(self.num_v))
                w_list.extend([1.0] * self.num_v)
            self.cache["e"] = e_list, w_list
        return self.cache["e"]

    @property
    def num_v(self) -> int:
        r"""Return the number of vertices in the graph.
        """
        return self._num_v

    @property
    def num_e(self) -> int:
        r"""Return the number of edges in the graph.
        """
        cnt = 0
        cnt += len(self._raw_e_dict)
        cnt += len(self._raw_selfloop_dict)
        if self._has_extra_selfloop:
            cnt += self.num_v
        return cnt

    # =====================================================================================
    # properties for deep learning
    @property
    @abc.abstractmethod
    def vars_for_DL(self) -> List[str]:
        r"""Return a name list of available variables for deep learning in this type of graph.
        """

    @abc.abstractproperty
    def A(self) -> torch.Tensor:
        r"""Return the sparsed adjacency matrix of the graph. Format ``torch.sparse_coo``.
        """

    # spectral-based smoothing
    def smoothing(self, X: torch.Tensor, L: torch.Tensor, lamb: float) -> torch.Tensor:
        r"""Spectral-based smoothing.

        .. math::
            X_{smoothed} = X + \lambda \mathcal{L} X

        Args:
            ``X`` (``torch.Tensor``): The vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``L`` (``torch.Tensor``): The Laplacian matrix with ``torch.sparse_coo_tensor`` format. Size :math:`(|\mathcal{V}|, |\mathcal{V}|)`.
            ``lamb`` (``float``): :math:`\lambda`, the strength of smoothing.
        """
        return X + lamb * torch.sparse.mm(L, X)

    # message passing functions
    @abc.abstractmethod
    def v2v(
        self, X: torch.Tensor, aggr: str = "mean", e_weight: Optional[torch.Tensor] = None,
    ):
        r"""Message passing from vertex to vertex on the graph structure.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size: :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``, optional): Aggregation function for neighbor messages, which can be ``'mean'``, ``'sum'``, or ``'softmax_then_sum'``. Default: ``'mean'``.
            ``e_weight`` (``torch.Tensor``, optional): The edge weight vector. Size: :math:`(|\mathcal{E}|,)`. Defaults to ``None``.
        """


class BaseHypergraph:
    r"""The ``BaseHypergraph`` class is the base class for all hypergraph structures.

    Args:
        ``num_v`` (``int``): The number of vertices in the hypergraph.
        ``e_list_v2e`` (``Union[List[int], List[List[int]]]``, optional): A list of hyperedges describes how the vertices point to the hyperedges. Defaults to ``None``.
        ``e_list_e2v`` (``Union[List[int], List[List[int]]]``, optional): A list of hyperedges describes how the hyperedges point to the vertices. Defaults to ``None``.
        ``w_list_v2e`` (``Union[List[float], List[List[float]]]``, optional): The weights are attached to the connections from vertices to hyperedges, which has the same shape
            as ``e_list_v2e``. If set to ``None``, the value ``1`` is used for all connections. Defaults to ``None``.
        ``w_list_e2v`` (``Union[List[float], List[List[float]]]``, optional): The weights are attached to the connections from the hyperedges to the vertices, which has the
            same shape to ``e_list_e2v``. If set to ``None``, the value ``1`` is used for all connections. Defaults to ``None``.
        ``e_weight`` (``Union[float, List[float]]``, optional): A list of weights for hyperedges. If set to ``None``, the value ``1`` is used for all hyperedges. Defaults to ``None``.
        ``v_weight`` (``Union[float, List[float]]``, optional): Weights for vertices. If set to ``None``, the value ``1`` is used for all vertices. Defaults to ``None``.
        ``device`` (``torch.device``, optional): The deivce to store the hypergraph. Defaults to ``torch.device('cpu')``.
    """

    def __init__(
        self,
        num_v: int,
        e_list_v2e: Optional[Union[List[int], List[List[int]]]] = None,
        e_list_e2v: Optional[Union[List[int], List[List[int]]]] = None,
        w_list_v2e: Optional[Union[List[float], List[List[float]]]] = None,
        w_list_e2v: Optional[Union[List[float], List[List[float]]]] = None,
        e_weight: Optional[Union[float, List[float]]] = None,
        v_weight: Optional[List[float]] = None,
        device: torch.device = torch.device("cpu"),
    ):
        assert isinstance(num_v, int) and num_v > 0, "num_v should be a positive integer"
        self.clear()
        self._num_v = num_v
        self.device = device

    @abc.abstractmethod
    def __repr__(self) -> str:
        r"""Print the hypergraph information.
        """

    @property
    @abc.abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        r"""Get the state dict of the hypergraph.
        """

    @abc.abstractmethod
    def save(self, file_path: Union[str, Path]):
        r"""Save the DHG's hypergraph structure to a file.

        Args:
            ``file_path`` (``str``): The file_path to store the DHG's hypergraph structure.
        """

    @abc.abstractstaticmethod
    def load(file_path: Union[str, Path]):
        r"""Load the DHG's hypergraph structure from a file.

        Args:
            ``file_path`` (``str``): The file path to load the DHG's hypergraph structure.
        """

    @abc.abstractstaticmethod
    def from_state_dict(state_dict: dict):
        r"""Load the DHG's hypergraph structure from the state dict.

        Args:
            ``state_dict`` (``dict``): The state dict to load the DHG's hypergraph.
        """

    @abc.abstractmethod
    def draw(self, **kwargs):
        r"""Draw the structure.
        """

    def clear(self):
        r"""Remove all hyperedges and caches from the hypergraph.
        """
        self._clear_raw()
        self._clear_cache()

    def _clear_raw(self):
        self._v_weight = None
        self._raw_groups = {}

    def _clear_cache(self, group_name: Optional[str] = None):
        self.cache = {}
        if group_name is None:
            self.group_cache = defaultdict(dict)
        else:
            self.group_cache.pop(group_name, None)

    @abc.abstractmethod
    def clone(self) -> "BaseHypergraph":
        r"""Return a copy of this type of hypergraph.
        """

    def to(self, device: torch.device):
        r"""Move the hypergraph to the specified device.

        Args:
            ``device`` (``torch.device``): The device to store the hypergraph.
        """
        self.device = device
        for v in self.vars_for_DL:
            if v in self.cache and self.cache[v] is not None:
                self.cache[v] = self.cache[v].to(device)
            for name in self.group_names:
                if v in self.group_cache[name] and self.group_cache[name][v] is not None:
                    self.group_cache[name][v] = self.group_cache[name][v].to(device)
        return self

    # utils
    def _hyperedge_code(self, src_v_set: List[int], dst_v_set: List[int]) -> Tuple:
        r"""Generate the hyperedge code.

        Args:
            ``src_v_set`` (``List[int]``): The source vertex set.
            ``dst_v_set`` (``List[int]``): The destination vertex set.
        """
        return tuple([src_v_set, dst_v_set])

    def _merge_hyperedges(self, e1: dict, e2: dict, op: str = "mean"):
        assert op in ["mean", "sum", "max",], "Hyperedge merge operation must be one of ['mean', 'sum', 'max']"
        _func = {
            "mean": lambda x, y: (x + y) / 2,
            "sum": lambda x, y: x + y,
            "max": lambda x, y: max(x, y),
        }
        _e = {}
        if "w_v2e" in e1 and "w_v2e" in e2:
            for _idx in range(len(e1["w_v2e"])):
                _e["w_v2e"] = _func[op](e1["w_v2e"][_idx], e2["w_v2e"][_idx])
        if "w_e2v" in e1 and "w_e2v" in e2:
            for _idx in range(len(e1["w_e2v"])):
                _e["w_e2v"] = _func[op](e1["w_e2v"][_idx], e2["w_e2v"][_idx])
        _e["w_e"] = _func[op](e1["w_e"], e2["w_e"])
        return _e

    @staticmethod
    def _format_e_list(e_list: Union[List[int], List[List[int]]]) -> List[List[int]]:
        r"""Format the hyperedge list.

        Args:
            ``e_list`` (``List[int]`` or ``List[List[int]]``): The hyperedge list.
        """
        if type(e_list[0]) in (int, float):
            return [tuple(sorted(e_list))]
        elif type(e_list) == tuple:
            e_list = list(e_list)
        elif type(e_list) == list:
            pass
        else:
            raise TypeError("e_list must be List[int] or List[List[int]].")
        for _idx in range(len(e_list)):
            e_list[_idx] = tuple(sorted(e_list[_idx]))
        return e_list

    @staticmethod
    def _format_e_list_and_w_on_them(
        e_list: Union[List[int], List[List[int]]], w_list: Optional[Union[List[int], List[List[int]]]] = None,
    ):
        r"""Format ``e_list`` and ``w_list``.

        Args:
            ``e_list`` (Union[List[int], List[List[int]]]): Hyperedge list.
            ``w_list`` (Optional[Union[List[int], List[List[int]]]]): Weights on connections. Defaults to ``None``.
        """
        bad_connection_msg = (
            "The weight on connections between vertices and hyperedges must have the same size as the hyperedges."
        )
        if isinstance(e_list, tuple):
            e_list = list(e_list)
        if w_list is not None and isinstance(w_list, tuple):
            w_list = list(w_list)
        if isinstance(e_list[0], int) and w_list is None:
            w_list = [1] * len(e_list)
            e_list, w_list = [e_list], [w_list]
        elif isinstance(e_list[0], int) and w_list is not None:
            assert len(e_list) == len(w_list), bad_connection_msg
            e_list, w_list = [e_list], [w_list]
        elif isinstance(e_list[0], list) and w_list is None:
            w_list = [[1] * len(e) for e in e_list]
        assert len(e_list) == len(w_list), bad_connection_msg
        # TODO: this step can be speeded up
        for idx in range(len(e_list)):
            assert len(e_list[idx]) == len(w_list[idx]), bad_connection_msg
            cur_e, cur_w = np.array(e_list[idx]), np.array(w_list[idx])
            sorted_idx = np.argsort(cur_e)
            e_list[idx] = tuple(cur_e[sorted_idx].tolist())
            w_list[idx] = cur_w[sorted_idx].tolist()
        return e_list, w_list

    def _fetch_H_of_group(self, direction: str, group_name: str):
        r"""Fetch the H matrix of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``direction`` (``str``): The direction of hyperedges can be either ``'v2e'`` or ``'e2v'``.
            ``group_name`` (``str``): The name of the group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        assert direction in ["v2e", "e2v"], "direction must be one of ['v2e', 'e2v']"
        if direction == "v2e":
            select_idx = 0
        else:
            select_idx = 1
        num_e = len(self._raw_groups[group_name])
        e_idx, v_idx = [], []
        for _e_idx, e in enumerate(self._raw_groups[group_name].keys()):
            sub_e = e[select_idx]
            v_idx.extend(sub_e)
            e_idx.extend([_e_idx] * len(sub_e))
        H = torch.sparse_coo_tensor(
            torch.tensor([v_idx, e_idx], dtype=torch.long),
            torch.ones(len(v_idx)),
            torch.Size([self.num_v, num_e]),
            device=self.device,
        ).coalesce()
        return H

    def _fetch_R_of_group(self, direction: str, group_name: str):
        r"""Fetch the R matrix of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``direction`` (``str``): The direction of hyperedges can be either ``'v2e'`` or ``'e2v'``.
            ``group_name`` (``str``): The name of the group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        assert direction in ["v2e", "e2v"], "direction must be one of ['v2e', 'e2v']"
        if direction == "v2e":
            select_idx = 0
        else:
            select_idx = 1
        num_e = len(self._raw_groups[group_name])
        e_idx, v_idx, w_list = [], [], []
        for _e_idx, e in enumerate(self._raw_groups[group_name].keys()):
            sub_e = e[select_idx]
            v_idx.extend(sub_e)
            e_idx.extend([_e_idx] * len(sub_e))
            w_list.extend(self._raw_groups[group_name][e][f"w_{direction}"])
        R = torch.sparse_coo_tensor(
            torch.vstack([v_idx, e_idx]), torch.tensor(w_list), torch.Size([self.num_v, num_e]), device=self.device,
        ).coalesce()
        return R

    def _fetch_W_of_group(self, group_name: str):
        r"""Fetch the W matrix of the specified hyperedge group with ``torch.sparse_coo_tensor`` format.

        Args:
            ``group_name`` (``str``): The name of the group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        w_list = [content["w_e"] for content in self._raw_groups[group_name].values()]
        W = torch.tensor(w_list, device=self.device).view((-1, 1))
        return W

    # some structure modification functions
    def add_hyperedges(
        self,
        e_list_v2e: Union[List[int], List[List[int]]],
        e_list_e2v: Union[List[int], List[List[int]]],
        w_list_v2e: Optional[Union[List[float], List[List[float]]]] = None,
        w_list_e2v: Optional[Union[List[float], List[List[float]]]] = None,
        e_weight: Optional[Union[float, List[float]]] = None,
        merge_op: str = "mean",
        group_name: str = "main",
    ):
        r"""Add hyperedges to the hypergraph. If the ``group_name`` is not specified, the hyperedges will be added to the default ``main`` hyperedge group.

        Args:
            ``num_v`` (``int``): The number of vertices in the hypergraph.
            ``e_list_v2e`` (``Union[List[int], List[List[int]]]``): A list of hyperedges describes how the vertices point to the hyperedges.
            ``e_list_e2v`` (``Union[List[int], List[List[int]]]``): A list of hyperedges describes how the hyperedges point to the vertices.
            ``w_list_v2e`` (``Union[List[float], List[List[float]]]``, optional): The weights are attached to the connections from vertices to hyperedges, which has the same shape
                as ``e_list_v2e``. If set to ``None``, the value ``1`` is used for all connections. Defaults to ``None``.
            ``w_list_e2v`` (``Union[List[float], List[List[float]]]``, optional): The weights are attached to the connections from the hyperedges to the vertices, which has the
                same shape to ``e_list_e2v``. If set to ``None``, the value ``1`` is used for all connections. Defaults to ``None``.
            ``e_weight`` (``Union[float, List[float]]``, optional): A list of weights for hyperedges. If set to ``None``, the value ``1`` is used for all hyperedges. Defaults to ``None``.
            ``merge_op`` (``str``): The merge operation for the conflicting hyperedges. The possible values are ``mean``, ``sum``, ``max``, and ``min``. Defaults to ``mean``.
            ``group_name`` (``str``, optional): The target hyperedge group to add these hyperedges. Defaults to the ``main`` hyperedge group.
        """
        e_list_v2e, w_list_v2e = self._format_e_list_and_w_on_them(e_list_v2e, w_list_v2e)
        e_list_e2v, w_list_e2v = self._format_e_list_and_w_on_them(e_list_e2v, w_list_e2v)
        if e_weight is None:
            e_weight = [1.0] * len(e_list_v2e)
        assert len(e_list_v2e) == len(e_weight), "The number of hyperedges and the number of weights are not equal."
        assert len(e_list_v2e) == len(e_list_e2v), "Hyperedges of 'v2e' and 'e2v' must have the same size."
        for _idx in range(len(e_list_v2e)):
            self._add_hyperedge(
                self._hyperedge_code(e_list_v2e[_idx], e_list_e2v[_idx]),
                {"w_v2e": w_list_v2e[_idx], "w_e2v": w_list_e2v[_idx], "w_e": e_weight[_idx],},
                merge_op,
                group_name,
            )
        self._clear_cache(group_name)

    def _add_hyperedge(
        self, hyperedge_code: Tuple[List[int], List[int]], content: Dict[str, Any], merge_op: str, group_name: str,
    ):
        r"""Add a hyperedge to the specified hyperedge group.

        Args:
            ``hyperedge_code`` (``Tuple[List[int], List[int]]``): The hyperedge code.
            ``content`` (``Dict[str, Any]``): The content of the hyperedge.
            ``merge_op`` (``str``): The merge operation for the conflicting hyperedges.
            ``group_name`` (``str``): The target hyperedge group to add this hyperedge.
        """
        if group_name not in self.group_names:
            self._raw_groups[group_name] = {}
            self._raw_groups[group_name][hyperedge_code] = content
        else:
            if hyperedge_code not in self._raw_groups[group_name]:
                self._raw_groups[group_name][hyperedge_code] = content
            else:
                self._raw_groups[group_name][hyperedge_code] = self._merge_hyperedges(
                    self._raw_groups[group_name][hyperedge_code], content, merge_op
                )

    def remove_hyperedges(
        self,
        e_list_v2e: Union[List[int], List[List[int]]],
        e_list_e2v: Union[List[int], List[List[int]]],
        group_name: Optional[str] = None,
    ):
        r"""Remove the specified hyperedges from the hypergraph.

        Args:
            ``e_list_v2e`` (``Union[List[int], List[List[int]]]``): A list of hyperedges describes how the vertices point to the hyperedges.
            ``e_list_e2v`` (``Union[List[int], List[List[int]]]``): A list of hyperedges describes how the hyperedges point to the vertices.
            ``group_name`` (``str``, optional): Remove these hyperedges from the specified hyperedge group. If not specified, the function will
                remove those hyperedges from all hyperedge groups. Defaults to the ``None``.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        assert len(e_list_v2e) == len(e_list_e2v), "Hyperedges of 'v2e' and 'e2v' must have the same size."
        e_list_v2e = self._format_e_list(e_list_v2e)
        e_list_e2v = self._format_e_list(e_list_e2v)
        if group_name is None:
            for _idx in range(len(e_list_v2e)):
                e_code = self._hyperedge_code(e_list_v2e[_idx], e_list_e2v[_idx])
                for name in self.group_names:
                    self._raw_groups[name].pop(e_code, None)
        else:
            for _idx in range(len(e_list_v2e)):
                e_code = self._hyperedge_code(e_list_v2e[_idx], e_list_e2v[_idx])
                self._raw_groups[group_name].pop(e_code, None)
        self._clear_cache(group_name)

    @abc.abstractmethod
    def drop_hyperedges(self, drop_rate: float, ord="uniform"):
        r"""Randomly drop hyperedges from the hypergraph. This function will return a new hypergraph with non-dropped hyperedges.

        Args:
            ``drop_rate`` (``float``): The drop rate of hyperedges.
            ``ord`` (``str``): The order of dropping edges. Currently, only ``'uniform'`` is supported. Defaults to ``uniform``.
        """

    @abc.abstractmethod
    def drop_hyperedges_of_group(self, group_name: str, drop_rate: float, ord="uniform"):
        r"""Randomly drop hyperedges from the specified hyperedge group. This function will return a new hypergraph with non-dropped hyperedges.

        Args:
            ``group_name`` (``str``): The name of the hyperedge group.
            ``drop_rate`` (``float``): The drop rate of hyperedges.
            ``ord`` (``str``): The order of dropping edges. Currently, only ``'uniform'`` is supported. Defaults to ``uniform``.
        """

    # properties for the hypergraph
    @property
    def v(self) -> List[int]:
        r"""Return the list of vertices.
        """
        if self.cache.get("v") is None:
            self.cache["v"] = list(range(self.num_v))
        return self.cache["v"]

    @property
    def v_weight(self) -> List[float]:
        r"""Return the vertex weights of the hypergraph.
        """
        if self._v_weight is None:
            self._v_weight = [1.0] * self.num_v
        return self._v_weight

    @v_weight.setter
    def v_weight(self, v_weight: List[float]):
        r"""Set the vertex weights of the hypergraph.
        """
        assert len(v_weight) == self.num_v, "The length of vertex weights must be equal to the number of vertices."
        self._v_weight = v_weight
        self._clear_cache()

    @property
    @abc.abstractmethod
    def e(self) -> Tuple[List[List[int]], List[float]]:
        r"""Return all hyperedges and weights in the hypergraph.
        """

    @abc.abstractmethod
    def e_of_group(self, group_name: str) -> Tuple[List[List[int]], List[float]]:
        r"""Return all hyperedges and weights in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """

    @property
    def num_v(self) -> int:
        r"""Return the number of vertices in the hypergraph.
        """
        return self._num_v

    @property
    def num_e(self) -> int:
        r"""Return the number of hyperedges in the hypergraph.
        """
        _num_e = 0
        for name in self.group_names:
            _num_e += len(self._raw_groups[name])
        return _num_e

    def num_e_of_group(self, group_name: str) -> int:
        r"""Return the number of hyperedges in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        return len(self._raw_groups[group_name])

    @property
    def num_groups(self) -> int:
        r"""Return the number of hyperedge groups in the hypergraph.
        """
        return len(self._raw_groups)

    @property
    def group_names(self) -> List[str]:
        r"""Return the names of hyperedge groups in the hypergraph.
        """
        return list(self._raw_groups.keys())

    # properties for deep learning
    @property
    @abc.abstractmethod
    def vars_for_DL(self) -> List[str]:
        r"""Return a name list of available variables for deep learning in this type of hypergraph.
        """

    @property
    def W_v(self) -> torch.Tensor:
        r"""Return the vertex weight matrix of the hypergraph.
        """
        if self.cache["W_v"] is None:
            self.cache["W_v"] = torch.tensor(self.v_weight, dtype=torch.float, device=self.device).view(-1, 1)
        return self.cache["W_v"]

    @property
    def W_e(self) -> torch.Tensor:
        r"""Return the hyperedge weight matrix of the hypergraph.
        """
        if self.cache["W_e"] is None:
            _tmp = [self.W_e_of_group(name) for name in self.group_names]
            self.cache["W_e"] = torch.cat(_tmp, dim=0)
        return self.cache["W_e"]

    def W_e_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the hyperedge weight matrix of the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name]["W_e"] is None:
            self.group_cache[group_name]["W_e"] = self._fetch_W_of_group(group_name)
        return self.group_cache[group_name]["W_e"]

    @property
    @abc.abstractmethod
    def H(self) -> torch.Tensor:
        r"""Return the hypergraph incidence matrix.
        """

    @property
    @abc.abstractmethod
    def H_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the hypergraph incidence matrix in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """

    @property
    def H_v2e(self) -> torch.Tensor:
        r"""Return the hypergraph incidence matrix with ``sparse matrix`` format.
        """
        if self.cache.get("H_v2e") is None:
            _tmp = [self.H_v2e_of_group(name) for name in self.group_names]
            self.cache["H_v2e"] = torch.cat(_tmp, dim=1)
        return self.cache["H_v2e"]

    def H_v2e_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the hypergraph incidence matrix with ``sparse matrix`` format in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("H_v2e") is None:
            self.group_cache[group_name]["H_v2e"] = self._fetch_H_of_group("v2e", group_name)
        return self.group_cache[group_name]["H_v2e"]

    @property
    def H_e2v(self) -> torch.Tensor:
        r"""Return the hypergraph incidence matrix with ``sparse matrix`` format.
        """
        if self.cache.get("H_e2v") is None:
            _tmp = [self.H_e2v_of_group(name) for name in self.group_names]
            self.cache["H_e2v"] = torch.cat(_tmp, dim=1)
        return self.cache["H_e2v"]

    def H_e2v_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the hypergraph incidence matrix with ``sparse matrix`` format in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("H_e2v") is None:
            self.group_cache[group_name]["H_e2v"] = self._fetch_H_of_group("e2v", group_name)
        return self.group_cache[group_name]["H_e2v"]

    @property
    def R_v2e(self) -> torch.Tensor:
        r"""Return the weight matrix of connections (vertices point to hyperedges) with ``sparse matrix`` format.
        """
        if self.cache.get("R_v2e") is None:
            _tmp = [self.R_v2e_of_group(name) for name in self.group_names]
            self.cache["R_v2e"] = torch.cat(_tmp, dim=1)
        return self.cache["R_v2e"]

    def R_v2e_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the weight matrix of connections (vertices point to hyperedges) with ``sparse matrix`` format in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("R_v2e") is None:
            self.group_cache[group_name]["R_v2e"] = self._fetch_R_of_group("v2e", group_name)
        return self.group_cache[group_name]["R_v2e"]

    @property
    def R_e2v(self) -> torch.Tensor:
        r"""Return the weight matrix of connections (hyperedges point to vertices) with ``sparse matrix`` format.
        """
        if self.cache.get("R_e2v") is None:
            _tmp = [self.R_e2v_of_group(name) for name in self.group_names]
            self.cache["R_e2v"] = torch.cat(_tmp, dim=1)
        return self.cache["R_e2v"]

    def R_e2v_of_group(self, group_name: str) -> torch.Tensor:
        r"""Return the weight matrix of connections (hyperedges point to vertices) with ``sparse matrix`` format in the specified hyperedge group.

        Args:
            ``group_name`` (``str``): The name of the specified hyperedge group.
        """
        assert group_name in self.group_names, f"The specified {group_name} is not in existing hyperedge groups."
        if self.group_cache[group_name].get("R_e2v") is None:
            self.group_cache[group_name]["R_e2v"] = self._fetch_R_of_group("e2v", group_name)
        return self.group_cache[group_name]["R_e2v"]

    # spectral-based smoothing
    def smoothing(self, X: torch.Tensor, L: torch.Tensor, lamb: float) -> torch.Tensor:
        r"""Spectral-based smoothing.

        .. math::
            X_{smoothed} = X + \lambda \mathcal{L} X

        Args:
            ``X`` (``torch.Tensor``): The vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``L`` (``torch.Tensor``): The Laplacian matrix with ``torch.sparse_coo_tensor`` format. Size :math:`(|\mathcal{V}|, |\mathcal{V}|)`.
            ``lamb`` (``float``): :math:`\lambda`, the strength of smoothing.
        """
        return X + lamb * torch.sparse.mm(L, X)

    # message passing functions
    @abc.abstractmethod
    def v2e_aggregation(
        self, X: torch.Tensor, aggr: str = "mean", v2e_weight: Optional[torch.Tensor] = None,
    ):
        r"""Message aggretation step of ``vertices to hyperedges``.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """

    @abc.abstractmethod
    def v2e_aggregation_of_group(
        self, group_name: str, X: torch.Tensor, aggr: str = "mean", v2e_weight: Optional[torch.Tensor] = None,
    ):
        r"""Message aggregation step of ``vertices to hyperedges`` in specified hyperedge group.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """

    @abc.abstractmethod
    def v2e_update(self, X: torch.Tensor, e_weight: Optional[torch.Tensor] = None):
        r"""Message update step of ``vertices to hyperedges``.

        Args:
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """

    @abc.abstractmethod
    def v2e_update_of_group(self, group_name: str, X: torch.Tensor, e_weight: Optional[torch.Tensor] = None):
        r"""Message update step of ``vertices to hyperedges`` in specified hyperedge group.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """

    @abc.abstractmethod
    def v2e(
        self,
        X: torch.Tensor,
        aggr: str = "mean",
        v2e_weight: Optional[torch.Tensor] = None,
        e_weight: Optional[torch.Tensor] = None,
    ):
        r"""Message passing of ``vertices to hyperedges``. The combination of ``v2e_aggregation`` and ``v2e_update``.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """

    @abc.abstractmethod
    def v2e_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        v2e_weight: Optional[torch.Tensor] = None,
        e_weight: Optional[torch.Tensor] = None,
    ):
        r"""Message passing of ``vertices to hyperedges`` in specified hyperedge group. The combination of ``e2v_aggregation_of_group`` and ``e2v_update_of_group``.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """

    @abc.abstractmethod
    def e2v_aggregation(
        self, X: torch.Tensor, aggr: str = "mean", e2v_weight: Optional[torch.Tensor] = None,
    ):
        r"""Message aggregation step of ``hyperedges to vertices``.

        Args:
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """

    @abc.abstractmethod
    def e2v_aggregation_of_group(
        self, group_name: str, X: torch.Tensor, aggr: str = "mean", e2v_weight: Optional[torch.Tensor] = None,
    ):
        r"""Message aggregation step of ``hyperedges to vertices`` in specified hyperedge group.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """

    @abc.abstractmethod
    def e2v_update(self, X: torch.Tensor, v_weight: Optional[torch.Tensor] = None):
        r"""Message update step of ``hyperedges to vertices``.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``v_weight`` (``torch.Tensor``, optional): The vertex weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """

    @abc.abstractmethod
    def e2v_update_of_group(self, group_name: str, X: torch.Tensor, v_weight: Optional[torch.Tensor] = None):
        r"""Message update step of ``hyperedges to vertices`` in specified hyperedge group.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``v_weight`` (``torch.Tensor``, optional): The vertex weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """

    @abc.abstractmethod
    def e2v(
        self,
        X: torch.Tensor,
        aggr: str = "mean",
        e2v_weight: Optional[torch.Tensor] = None,
        v_weight: Optional[torch.Tensor] = None,
    ):
        r"""Message passing of ``hyperedges to vertices``. The combination of ``e2v_aggregation`` and ``e2v_update``.

        Args:
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``v_weight`` (``torch.Tensor``, optional): The vertex weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """

    @abc.abstractmethod
    def e2v_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        e2v_weight: Optional[torch.Tensor] = None,
        v_weight: Optional[torch.Tensor] = None,
    ):
        r"""Message passing of ``hyperedges to vertices`` in specified hyperedge group. The combination of ``e2v_aggregation_of_group`` and ``e2v_update_of_group``.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Hyperedge feature matrix. Size :math:`(|\mathcal{E}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``v_weight`` (``torch.Tensor``, optional): The vertex weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """

    @abc.abstractmethod
    def v2v(
        self,
        X: torch.Tensor,
        aggr: str = "mean",
        v2e_aggr: Optional[str] = None,
        v2e_weight: Optional[torch.Tensor] = None,
        e_weight: Optional[torch.Tensor] = None,
        e2v_aggr: Optional[str] = None,
        e2v_weight: Optional[torch.Tensor] = None,
        v_weight: Optional[torch.Tensor] = None,
    ):
        r"""Message passing of ``vertices to vertices``. The combination of ``v2e`` and ``e2v``.

        Args:
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, this ``aggr`` will be used to both ``v2e`` and ``e2v``.
            ``v2e_aggr`` (``str``, optional): The aggregation method for hyperedges to vertices. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, it will override the ``aggr`` in ``e2v``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e2v_aggr`` (``str``, optional): The aggregation method for vertices to hyperedges. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, it will override the ``aggr`` in ``v2e``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``v_weight`` (``torch.Tensor``, optional): The vertex weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """

    @abc.abstractmethod
    def v2v_of_group(
        self,
        group_name: str,
        X: torch.Tensor,
        aggr: str = "mean",
        v2e_aggr: Optional[str] = None,
        v2e_weight: Optional[torch.Tensor] = None,
        e_weight: Optional[torch.Tensor] = None,
        e2v_aggr: Optional[str] = None,
        e2v_weight: Optional[torch.Tensor] = None,
        v_weight: Optional[torch.Tensor] = None,
    ):
        r"""Message passing of ``vertices to vertices`` in specified hyperedge group. The combination of ``v2e_of_group`` and ``e2v_of_group``.

        Args:
            ``group_name`` (``str``): The specified hyperedge group.
            ``X`` (``torch.Tensor``): Vertex feature matrix. Size :math:`(|\mathcal{V}|, C)`.
            ``aggr`` (``str``): The aggregation method. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, this ``aggr`` will be used to both ``v2e_of_group`` and ``e2v_of_group``.
            ``v2e_aggr`` (``str``, optional): The aggregation method for hyperedges to vertices. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, it will override the ``aggr`` in ``e2v_of_group``.
            ``v2e_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (vertices point to hyepredges). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e_weight`` (``torch.Tensor``, optional): The hyperedge weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``e2v_aggr`` (``str``, optional): The aggregation method for vertices to hyperedges. Can be ``'mean'``, ``'sum'`` and ``'softmax_then_sum'``. If specified, it will override the ``aggr`` in ``v2e_of_group``.
            ``e2v_weight`` (``torch.Tensor``, optional): The weight vector attached to connections (hyperedges point to vertices). If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
            ``v_weight`` (``torch.Tensor``, optional): The vertex weight vector. If not specified, the function will use the weights specified in hypergraph construction. Defaults to ``None``.
        """
