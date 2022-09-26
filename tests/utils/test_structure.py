import pytest
from dhg import utils


def test_remap_edge_list():
    e_list = [(1, 3), ('A', 100), (4.5, 'A')]

    e1, m = utils.remap_edge_list(e_list, ret_map=True)
    for i, (u, v) in enumerate(e_list):
        assert e1[i][0] == m[u]
        assert e1[i][1] == m[v]

    e2, m = utils.remap_edge_list(e_list, ret_map=True, bipartite_graph=True)
    for i, (u, v) in enumerate(e_list):
        assert e2[i][0] == m[u]
        assert e2[i][1] == m[v]


def test_remap_edge_lists():
    e_list = [[(1, 3), ('A', 100), (4.5, 'A')], [(1, 5), ('B', 101), (4.1, 'A')]]

    e1, m = utils.remap_edge_lists(e_list, ret_map=True)
    for i, ee in enumerate(e_list):
        for j, (u, v) in enumerate(ee):
            assert e1[i][j][0] == m[u]
            assert e1[i][j][1] == m[v]

    e2, m = utils.remap_edge_list(e_list, ret_map=True, bipartite_graph=True)
    for i, ee in enumerate(e_list):
        for j, (u, v) in enumerate(ee):
            assert e2[i][j][0] == m[u]
            assert e2[i][j][1] == m[v]


def test_remap_adj_list():
    adj_list = [[0, 'A', 1.5], ['A', 1, 2], [1.5, 1, 0]]
    e1, m = utils.remap_adj_list(adj_list, ret_map=True)
    for i, adj in enumerate(adj_list):
        for j, a in enumerate(adj):
            assert e1[i][j] == m[a]

    e2, m = utils.remap_adj_list(adj_list, ret_map=True, bipartite_graph=True)
    for i, adj in enumerate(adj_list):
        for j, a in enumerate(adj):
            assert e2[i][j] == m[a]


def test_remap_adj_lists():
    adj_lists = [[[0, 'A', 1.5], ['A', 1, 2], [1.5, 1, 0]], [[0, 3, 'A', 'B'], [1, 2, 'A', 'B'], ['A', 'B', 0, 1]]]
    e1, m = utils.remap_adj_lists(adj_lists, ret_map=True)
    for i, adj_list in enumerate(adj_lists):
        for j, adj in enumerate(adj_list):
            for k, a in enumerate(adj):
                assert e1[i][j][k] == m[a]

    e2, m = utils.remap_adj_list(adj_lists, ret_map=True, bipartite_graph=True)
    for i, adj_list in enumerate(adj_lists):
        for j, adj in enumerate(adj_list):
            for k, a in enumerate(adj):
                assert e2[i][j][k] == m[a]



def test_edge_list_to_adj_list():
    e_list = [(0, 3), (1, 2), (1, 3), (2, 3), (3, 4), (3, 2), (3, 1)]
    adj_list = utils.edge_list_to_adj_list(e_list)
    for adj in adj_list:
        if adj[0] == 0:
            assert len(adj) == 2 and adj[1] == 3
        if adj[0] == 1:
            assert len(adj) == 3 and 2 in adj and 3 in adj
        if adj[0] == 2:
            assert len(adj) == 2 and adj[1] == 3
        if adj[0] == 3:
            assert len(adj) == 4 and 4 in adj and 2 in adj and 1 in adj


def test_edge_list_to_adj_dict():
    e_list = [(0, 3), (1, 2), (1, 3), (2, 3), (3, 4), (3, 2), (3, 1)]
    adj_dict = utils.edge_list_to_adj_dict(e_list)
    assert len(adj_dict) == 4
    assert len(adj_dict[0]) == 1 and adj_dict[0][0] == 3
    assert len(adj_dict[1]) == 2 and 2 in adj_dict[1] and 3 in adj_dict[1]
    assert len(adj_dict[2]) == 1 and adj_dict[2][0] == 3
    assert len(adj_dict[3]) == 3 and 4 in adj_dict[3] and 2 in adj_dict[3] and 1 in adj_dict[3]


def test_adj_list_to_edge_list():
    adj_list = [[0, 3], [3, 2, 1], [2, 3], [1, 2, 3, 4], [1, 3]]
    e_list = utils.adj_list_to_edge_list(adj_list)
    assert (0, 3) in e_list
    assert (3, 2) in e_list
    assert (3, 1) in e_list
    assert (1, 2) in e_list
    assert (1, 3) in e_list
    assert (1, 4) in e_list
    assert (2, 3) in e_list
    assert len(e_list) == 7   # 8 ???
