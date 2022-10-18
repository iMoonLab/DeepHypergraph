from typing import Union, List, Tuple, Dict
from collections import defaultdict


def edge_list_to_adj_list(e_list: List[Tuple[int, int]]) -> List[List[int]]:
    r"""Convert edge list to adjacency list for low-order structures.

    .. note::
        Adjacency list can only represent low-order structures like graph, directed graph, and bipartite graph.

    Args:
        ``e_list`` (``List[Tuple[int, int]]``): Edge list.
    """
    adj_list = []
    adj_dict = edge_list_to_adj_dict(e_list)
    for src_idx, dst_idx_list in adj_dict.items():
        adj_list.append([src_idx] + dst_idx_list)
    return adj_list


def edge_list_to_adj_dict(e_list: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    r"""Convert edge list to adjacency dictionary for low-order structures.

    .. note::
        Adjacency list can only represent low-order structures like graph, directed graph, and bipartite graph.

    Args:
        ``e_list`` (``List[Tuple[int, int]]``): Edge list.
    """
    adj_dict = defaultdict(list)
    for src_idx, dst_idx in e_list:
        adj_dict[src_idx].append(dst_idx)
    return adj_dict


def adj_list_to_edge_list(adj_list: List[List[int]]) -> List[Tuple[int, int]]:
    r"""Convert adjacency list to edge list for low-order structures.

    .. note::
        Adjacency list can only represent low-order structures like graph, directed graph, and bipartite graph.

    Args:
        ``adj_list`` (``List[List[int]]``): Adjacency list.
    """
    e_list = []
    for line in adj_list:
        if len(line) <= 1:
            continue
        src_idx = line[0]
        for dst_idx in line[1:]:
            e_list.append((src_idx, dst_idx))
    return e_list


def remap_edge_list(
    e_list: List[tuple], bipartite_graph: bool = False, ret_map: bool = False
) -> Union[List[tuple], tuple]:
    r"""Remap the vertex markers to numbers of an ordered and continuous range.

    .. note::
        This function can support both low-order structures and high-order structures.

    Args:
        ``e_list`` (``List[tuple]``): Edge list of low-order structures or high-order structures.
        ``bipartite_graph`` (``bool``): Whether the structure is bipartite graph. Defaults to ``False``.
        ``ret_map`` (``bool``): Whether to return the map dictionary of raw marker to new index. Defaults to ``False``.
    """
    e_list = [[str(v) for v in e] for e in e_list]
    if bipartite_graph:
        u_set, v_set = set(), set()
        for u, v in e_list:
            u_set.add(u)
            v_set.add(v)
        u_list, v_list = sorted(u_set), sorted(v_set)
        u_map, v_map = (
            {raw_u: new_u for new_u, raw_u in enumerate(u_list)},
            {raw_v: new_v for new_v, raw_v in enumerate(v_list)},
        )
        e_list = [(u_map[u], v_map[v]) for u, v in e_list]
        if ret_map:
            return e_list, u_map, v_map
        else:
            return e_list
    else:
        v_set = set()
        for e in e_list:
            for v in e:
                v_set.add(v)
        v_list = sorted(v_set)
        v_map = {raw_v: new_v for new_v, raw_v in enumerate(v_list)}
        e_list = [tuple([v_map[v] for v in e]) for e in e_list]
        if ret_map:
            return e_list, v_map
        else:
            return e_list


def remap_edge_lists(
    *e_lists: List[List[tuple]], bipartite_graph: bool = False, ret_map: bool = False
) -> Union[List[List[tuple]], tuple]:
    r"""Remap the vertex markers to numbers of an ordered and continuous range for given multiple edge lists.

    .. note::
        This function can support both low-order structures and high-order structures.

    Args:
        ``e_lists`` (``List[List[tuple]]``): The list of edge list of low-order structures or high-order structures.
        ``bipartite_graph`` (``bool``): Whether the structure is bipartite graph. Defaults to ``False``.
        ``ret_map`` (``bool``): Whether to return the map dictionary of raw marker to new index. Defaults to ``False``.
    """
    e_lists = [[[str(v) for v in e] for e in e_list] for e_list in e_lists]
    if bipartite_graph:
        u_set, v_set = set(), set()
        for e_list in e_lists:
            for u, v in e_list:
                u_set.add(u)
                v_set.add(v)
        u_list, v_list = sorted(u_set), sorted(v_set)
        u_map, v_map = (
            {raw_u: new_u for new_u, raw_u in enumerate(u_list)},
            {raw_v: new_v for new_v, raw_v in enumerate(v_list)},
        )
        e_lists = [[(u_map[u], v_map[v]) for u, v in e_list] for e_list in e_lists]
        if ret_map:
            return e_lists, u_map, v_map
        else:
            return e_lists
    else:
        v_set = set()
        for e_list in e_lists:
            for e in e_list:
                for v in e:
                    v_set.add(v)
        v_list = sorted(v_set)
        v_map = {raw_v: new_v for new_v, raw_v in enumerate(v_list)}
        e_list = [[tuple([v_map[v] for v in e]) for e in e_list] for e_list in e_lists]
        if ret_map:
            return e_list, v_map
        else:
            return e_list


def remap_adj_list(
    adj_list: List[List[int]], bipartite_graph: bool = False, ret_map: bool = False
) -> Union[List[List[int]], tuple]:
    r"""Remap the vertex markers to numbers of an ordered and continuous range.

    .. note::
        This function can only support low-order structures like graph, directed graph, and bipartite graph.

    Args:
        ``adj_list`` (``List[List[int]]``): Adjacency list of low-order structures.
        ``bipartite_graph`` (``bool``): Whether the structure is bipartite graph. Defaults to ``False``.
        ``ret_map`` (``bool``): Whether to return the map dictionary of raw marker to new index. Defaults to ``False``.
    """
    adj_list = [[str(v) for v in line] for line in adj_list]
    if bipartite_graph:
        u_set, v_set = set(), set()
        for line in adj_list:
            if len(line) <= 1:
                continue
            u_set.add(line[0])
            for v in line[1:]:
                v_set.add(v)
        u_list, v_list = sorted(u_set), sorted(v_set)
        u_map, v_map = (
            {raw_u: new_u for new_u, raw_u in enumerate(u_list)},
            {raw_v: new_v for new_v, raw_v in enumerate(v_list)},
        )
        adj_list = [
            [u_map[line[0]]] + [v_map[v] for v in line[1:]] for line in adj_list
        ]
        if ret_map:
            return adj_list, u_map, v_map
        else:
            return adj_list
    else:
        v_set = set()
        for line in adj_list:
            if len(line) <= 1:
                continue
            for v in line:
                v_set.add(v)
        v_list = sorted(v_set)
        v_map = {raw_v: new_v for new_v, raw_v in enumerate(v_list)}
        adj_list = [[v_map[v] for v in line] for line in adj_list]
        if ret_map:
            return adj_list, v_map
        else:
            return adj_list


def remap_adj_lists(
    *adj_lists: List[List[List[int]]],
    bipartite_graph: bool = False,
    ret_map: bool = False
) -> Union[List[List[List[int]]], tuple]:
    r"""Remap the vertex markers to numbers of an ordered and continuous range for given multiple adjacency lists.

    .. note::
        This function can only support low-order structures like graph, directed graph, and bipartite graph.

    Args:
        ``adj_lists`` (``List[List[List[int]]]``): The list of adjacency list of low-order structures.
        ``bipartite_graph`` (``bool``): Whether the structure is bipartite graph. Defaults to ``False``.
        ``ret_map`` (``bool``): Whether to return the map dictionary of raw marker to new index. Defaults to ``False``.
    """
    adj_lists = [[[str(v) for v in line] for line in adj_list] for adj_list in adj_lists]
    if bipartite_graph:
        u_set, v_set = set(), set()
        for adj_list in adj_lists:
            for line in adj_list:
                if len(line) <= 1:
                    continue
                u_set.add(line[0])
                for v in line[1:]:
                    v_set.add(v)
        u_list, v_list = sorted(u_set), sorted(v_set)
        u_map, v_map = (
            {raw_u: new_u for new_u, raw_u in enumerate(u_list)},
            {raw_v: new_v for new_v, raw_v in enumerate(v_list)},
        )
        adj_lists = [
            [[u_map[line[0]]] + [v_map[v] for v in line[1:]] for line in adj_list]
            for adj_list in adj_lists
        ]
        if ret_map:
            return adj_lists, u_map, v_map
        else:
            return adj_lists
    else:
        v_set = set()
        for adj_list in adj_lists:
            for line in adj_list:
                if len(line) <= 1:
                    continue
                for v in line:
                    v_set.add(v)
        v_list = sorted(v_set)
        v_map = {raw_v: new_v for new_v, raw_v in enumerate(v_list)}
        adj_lists = [
            [[v_map[v] for v in line] for line in adj_list] for adj_list in adj_lists
        ]
        if ret_map:
            return adj_lists, v_map
        else:
            return adj_lists
