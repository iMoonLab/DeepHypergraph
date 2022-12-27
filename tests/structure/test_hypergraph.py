from copy import deepcopy
import torch
import pytest
import numpy as np
import scipy.spatial
from dhg import Graph, BiGraph, Hypergraph
from dhg.random import graph_Gnm, bigraph_Gnm

# from dhg.random.hypergraph import hypergraph_Gnm


@pytest.fixture()
def g1():
    e_list = [(0, 1, 2, 5), (0, 1), (2, 3, 4), (3, 2, 4)]
    g = Hypergraph(6, e_list)
    return g


@pytest.fixture()
def g2():
    e_list = [(1, 2, 3), (0, 1, 3), (0, 1), (2, 4, 3), (2, 3)]
    e_weight = [0.5, 1, 0.5, 1, 0.5]
    g = Hypergraph(5, e_list, e_weight)
    return g


def test_save(g1, tmp_path):
    from dhg import load_structure

    g1.save(tmp_path / "g1")
    g2 = load_structure(tmp_path / "g1")

    for e1, e2 in zip(g1.e[0], g2.e[0]):
        assert e1 == e2
    for w1, w2 in zip(g1.e[1], g2.e[1]):
        assert w1 == w2


# test construction
def test_from_feature_kNN():
    ft = np.random.rand(32, 8)
    cdist = scipy.spatial.distance.cdist(ft, ft)
    tk_mat = np.argsort(cdist, axis=1)[:, :3]
    hg = Hypergraph.from_feature_kNN(torch.tensor(ft), k=3)
    assert tuple(sorted(tk_mat[0].tolist())) in hg.e[0]
    assert tuple(sorted(tk_mat[8].tolist())) in hg.e[0]
    assert tuple(sorted(tk_mat[13].tolist())) in hg.e[0]
    assert tuple(sorted(tk_mat[26].tolist())) in hg.e[0]


def test_from_graph():
    g = Graph(5, [(0, 1), (0, 3), (1, 4), (2, 3), (3, 4)])
    hg = Hypergraph.from_graph(g)
    assert hg.num_e == 5
    assert (0, 1) in hg.e[0]
    assert (1, 4) in hg.e[0]


def test_from_graph_kHop():
    g = Graph(5, [(0, 1), (0, 3), (1, 4), (2, 3)])
    hg = Hypergraph.from_graph_kHop(g, k=1)
    assert hg.num_e == 5
    assert (0, 1, 3) in hg.e[0]
    assert (0, 1, 4) in hg.e[0]
    assert (1, 4) in hg.e[0]
    assert (2, 3) in hg.e[0]
    assert (0, 2, 3) in hg.e[0]
    hg = Hypergraph.from_graph_kHop(g, k=2)
    assert hg.num_e == 5
    assert (0, 1, 3, 4) in hg.e[0]
    hg = Hypergraph.from_graph_kHop(g, k=2, only_kHop=True)
    assert hg.num_e == 4
    assert (1, 3) in hg.e[0]


def test_from_bigraph():
    g = BiGraph(3, 4, [(0, 1), (0, 2), (1, 2), (2, 3)])
    hg = Hypergraph.from_bigraph(g, U_as_vertex=True)
    assert hg.num_v == 3 and hg.num_e == 3
    assert (0,) in hg.e[0]
    assert (0, 1) in hg.e[0]
    assert (2,) in hg.e[0]
    hg = Hypergraph.from_bigraph(g, U_as_vertex=False)
    assert hg.num_v == 4 and hg.num_e == 3
    assert (1, 2) in hg.e[0]
    assert (2,) in hg.e[0]
    assert (3,) in hg.e[0]


# test representation
def test_empty():
    g = Hypergraph(10)
    assert g.num_v == 10
    assert g.e == ([], [])


def test_init(g1, g2):
    assert g1.num_v == 6
    assert g1.num_e == 3
    assert g1.e[0] == [(0, 1, 2, 5), (0, 1), (2, 3, 4)]
    assert g1.e[1] == [1, 1, 1]
    assert g2.num_v == 5
    assert g2.num_e == 5
    assert g2.e[0] == [(1, 2, 3), (0, 1, 3), (0, 1), (2, 3, 4), (2, 3)]
    assert g2.e[1] == [0.5, 1, 0.5, 1, 0.5]


def test_clear(g1):
    assert g1.num_e == 3
    g1.clear()
    assert g1.num_e == 0
    assert g1.e == ([], [])


def test_add_and_merge_hyperedges(g1):
    assert g1.e[1] == [1, 1, 1]
    g1.add_hyperedges([0, 1], 3, merge_op="mean")
    assert g1.e[1] == [1, 2, 1]
    assert g1.e[0] == [(0, 1, 2, 5), (0, 1), (2, 3, 4)]
    g1.add_hyperedges([(2, 4, 3), (1, 0), (3, 4)], [1, 1, 1], merge_op="sum")
    assert g1.e[0] == [(0, 1, 2, 5), (0, 1), (2, 3, 4), (3, 4)]
    assert g1.e[1] == [1, 3, 2, 1]


def test_add_hyperedges_from_feature_kNN(g1):

    origin_e = deepcopy(g1.e[0])
    ft = np.random.rand(6, 8)
    cdist = scipy.spatial.distance.cdist(ft, ft)
    tk_mat = np.argsort(cdist, axis=1)[:, :3]

    g1.add_hyperedges_from_feature_kNN(torch.tensor(ft), k=3, group_name="knn")
    assert tuple(sorted(tk_mat[0].tolist())) in g1.e_of_group("knn")[0]
    assert tuple(sorted(tk_mat[3].tolist())) in g1.e_of_group("knn")[0]
    assert tuple(sorted(tk_mat[4].tolist())) in g1.e_of_group("knn")[0]
    assert tuple(sorted(tk_mat[5].tolist())) in g1.e_of_group("knn")[0]

    for e in origin_e:
        assert e in g1.e_of_group("main")[0]

    for e in g1.e_of_group("main")[0]:
        assert e in origin_e


def test_add_hyperedges_from_graph(g1):
    g = graph_Gnm(6, 3)

    origin_e = deepcopy(g1.e[0])

    g1.add_hyperedges_from_graph(g, group_name="graph")
    g_e = g.e[0]
    g1_e = g1.e_of_group("graph")[0]

    for e in g_e:
        assert e in g1_e

    for e in origin_e:
        assert e in g1.e_of_group("main")[0]

    for e in g1.e[0]:
        assert e in origin_e or e in g_e


def test_add_hyperedges_from_graph_kHop(g1):
    g = graph_Gnm(6, 5)

    origin_e = deepcopy(g1.e[0])
    for k in range(1, 3):
        gg1 = deepcopy(g1)
        gg1.add_hyperedges_from_graph_kHop(g, k=k, group_name="kHop")

        khop = [[] for _ in range(6)]
        for kk in range(k):
            for v in range(6):
                if kk == 0:
                    khop[v] = g.nbr_v(v)
                else:
                    kk_hop_v = []
                    for nbr in khop[v]:
                        kk_hop_v += g.nbr_v(nbr)
                    khop[v] += kk_hop_v
                khop[v] = list(set(khop[v]))

        for v in range(6):
            edge = [v] + khop[v]
            edge = tuple(set(sorted(edge)))
            assert edge in gg1.e_of_group("kHop")[0]

        gg2 = deepcopy(g1)
        gg2.add_hyperedges_from_graph_kHop(g, k=k, group_name="kHop", only_kHop=True)

        khop = [[] for _ in range(6)]
        for kk in range(k):
            for v in range(6):
                if len(khop[v]) == 0:
                    khop[v] = g.nbr_v(v)
                else:
                    kk_hop_v = []
                    for nbr in khop[v]:
                        kk_hop_v += g.nbr_v(nbr)
                    khop[v] = kk_hop_v
                khop[v] = list(set(khop[v]))

        for v in range(6):
            edge = [v] + khop[v]
            edge = tuple(set(sorted(edge)))
            assert edge in gg2.e_of_group("kHop")[0]

        for e in origin_e:
            assert e in gg1.e_of_group("main")[0]
            assert e in gg2.e_of_group("main")[0]

        for e in gg1.e_of_group("main")[0]:
            assert e in origin_e
        for e in gg2.e_of_group("main")[0]:
            assert e in origin_e


def test_add_hyperedges_from_bigraph():
    g = BiGraph(4, 3, [[0, 0], [1, 0], [2, 0], [3, 0], [0, 1], [1, 1], [2, 2], [3, 2]])
    hg = Hypergraph(3)
    hg.add_hyperedges_from_bigraph(g, group_name="bigraph")
    assert hg.num_e == 2
    assert (0, 1) in hg.e_of_group("bigraph")[0]
    assert (0, 2) in hg.e_of_group("bigraph")[0]

    hg = Hypergraph(4)
    hg.add_hyperedges_from_bigraph(g, group_name="bigraph-u", U_as_vertex=True)
    assert hg.num_e == 3
    assert (0, 1, 2, 3) in hg.e_of_group("bigraph-u")[0]
    assert (0, 1) in hg.e_of_group("bigraph-u")[0]
    assert (2, 3) in hg.e_of_group("bigraph-u")[0]


def test_remove_hyperedges(g1):
    assert g1.e[0] == [(0, 1, 2, 5), (0, 1), (2, 3, 4)]
    assert g1.e[1] == [1, 1, 1]
    g1.remove_hyperedges([0, 1])
    assert (0, 1) not in g1.e[0]
    assert (0, 1, 5) not in g1.e[0]
    g1.add_hyperedges([[0, 1, 5], [2, 3, 4]])
    assert (0, 1, 5) in g1.e[0]
    g1.remove_hyperedges([[0, 1, 5], (0, 1, 2, 5)])
    assert (0, 1, 5) not in g1.e[0]
    assert (0, 1, 2, 5) not in g1.e[0]
    g1.clear()
    assert g1.num_e == 0
    assert g1.e == ([], [])


def test_remove_group(g1):
    origin_e = deepcopy(g1.e[0])

    g1.add_hyperedges(([0, 1, 2, 5], [0, 1]), group_name="test")
    for e in origin_e:
        assert e in g1.e_of_group("main")[0]
    for e in g1.e_of_group("main")[0]:
        assert e in origin_e

    g1.remove_group("none")

    g1.remove_group("test")
    assert "test" not in g1.group_names

    for e in origin_e:
        assert e in g1.e_of_group("main")[0]
    for e in g1.e_of_group("main")[0]:
        assert e in origin_e

    g1.remove_group("main")

    assert len(g1.e[0]) == 0
    assert len(g1.e[1]) == 0


def test_add_and_remove_group(g1):
    assert g1.group_names == ["main"]
    g1.add_hyperedges([0, 2, 3], group_name="knn")
    assert len(g1.group_names) == 2
    assert "main" in g1.group_names
    assert "knn" in g1.group_names
    assert (0, 2, 3) in g1.e[0]
    assert (0, 2, 3) in g1.e_of_group("knn")[0]
    assert (0, 2, 3) not in g1.e_of_group("main")[0]
    g1.remove_hyperedges([0, 2, 3], group_name="knn")
    assert (0, 2, 3) not in g1.e[0]
    assert (0, 2, 3) not in g1.e_of_group("knn")[0]


def test_deg(g1, g2):
    assert g1.deg_v == [2, 2, 2, 1, 1, 1]
    assert g1.deg_e == [4, 2, 3]
    assert g2.deg_v == [1.5, 2, 2, 3, 1]
    assert g2.deg_e == [3, 3, 2, 3, 2]


def test_deg_group(g1):
    assert g1.deg_v == [2, 2, 2, 1, 1, 1]
    assert g1.deg_e == [4, 2, 3]
    g1.add_hyperedges([0, 2], 1, group_name="knn")
    assert g1.deg_v == [3, 2, 3, 1, 1, 1]
    assert g1.deg_e == [4, 2, 3, 2]
    assert g1.deg_v_of_group("main") == [2, 2, 2, 1, 1, 1]
    assert g1.deg_e_of_group("main") == [4, 2, 3]
    assert g1.deg_v_of_group("knn") == [1, 0, 1, 0, 0, 0]
    assert g1.deg_e_of_group("knn") == [2]


def test_nbr(g1, g2):
    assert g1.nbr_v(0) == [0, 1, 2, 5]
    assert g1.nbr_e(1) == [0, 1]
    assert g2.nbr_v(2) == [0, 1]
    assert g2.nbr_e(4) == [3]


def test_nbr_group(g1):
    assert g1.nbr_v(1) == [0, 1]
    assert g1.nbr_e(0) == [0, 1]
    g1.add_hyperedges([[0, 1]], group_name="knn")
    assert g1.nbr_v(1) == [0, 1]
    assert g1.nbr_e(1) == [0, 1, 3]
    assert g1.nbr_v_of_group(1, "main") == [0, 1]
    assert g1.nbr_e_of_group(2, "main") == [0, 2]
    assert g1.nbr_v_of_group(0, "knn") == [0, 1]
    assert g1.nbr_e_of_group(1, "knn") == [0]


def test_clone(g1):
    assert g1.num_v == 6
    assert g1.num_e == 3
    g1_clone = g1.clone()
    g1_clone.add_hyperedges([0, 2], 1, group_name="knn")
    assert g1.num_e == 3
    assert g1_clone.num_e == 4


# test deep learning
def test_v2e_index(g1):
    v2e_src = g1.v2e_src.view(-1, 1)
    v2e_dst = g1.v2e_dst.view(-1, 1)

    index = torch.cat((v2e_src, v2e_dst), dim=1)
    index = index.numpy().tolist()
    index = list(map(lambda x: tuple(x), index))

    assert (0, 0) in index
    assert (1, 0) in index
    assert (2, 0) in index
    assert (5, 0) in index
    assert (0, 1) in index
    assert (1, 1) in index
    assert (2, 2) in index
    assert (3, 2) in index
    assert (4, 2) in index


def test_v2e_index_group(g1):
    v2e_src = g1.v2e_src_of_group("main").view(-1, 1)
    v2e_dst = g1.v2e_dst_of_group("main").view(-1, 1)

    index = torch.cat((v2e_src, v2e_dst), dim=1)
    index = index.numpy().tolist()
    index = list(map(lambda x: tuple(x), index))

    assert (0, 0) in index
    assert (1, 0) in index
    assert (2, 0) in index
    assert (5, 0) in index
    assert (0, 1) in index
    assert (1, 1) in index
    assert (2, 2) in index
    assert (3, 2) in index
    assert (4, 2) in index


def test_e2v_index(g1):
    e2v_src = g1.e2v_src.view(-1, 1)
    e2v_dst = g1.e2v_dst.view(-1, 1)

    index = torch.cat((e2v_src, e2v_dst), dim=1)
    index = index.numpy().tolist()
    index = list(map(lambda x: tuple(x), index))

    assert (0, 0) in index
    assert (0, 1) in index
    assert (0, 2) in index
    assert (0, 5) in index
    assert (1, 0) in index
    assert (1, 1) in index
    assert (2, 2) in index
    assert (2, 3) in index
    assert (2, 4) in index


def test_e2v_index_group(g1):
    e2v_src = g1.e2v_src_of_group("main").view(-1, 1)
    e2v_dst = g1.e2v_dst_of_group("main").view(-1, 1)

    index = torch.cat((e2v_src, e2v_dst), dim=1)
    index = index.numpy().tolist()
    index = list(map(lambda x: tuple(x), index))

    assert (0, 0) in index
    assert (0, 1) in index
    assert (0, 2) in index
    assert (0, 5) in index
    assert (1, 0) in index
    assert (1, 1) in index
    assert (2, 2) in index
    assert (2, 3) in index
    assert (2, 4) in index


def test_H(g1):
    assert (
        g1.H.to_dense().cpu() == torch.tensor([[1, 1, 0], [1, 1, 0], [1, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0]])
    ).all()


def test_H_group(g1):
    g1.add_hyperedges([0, 4, 5], group_name="knn")
    assert (
        g1.H.to_dense().cpu()
        == torch.tensor([[1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1],])
    ).all()
    assert (
        g1.H_of_group("main").to_dense().cpu()
        == torch.tensor([[1, 1, 0], [1, 1, 0], [1, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0]])
    ).all()
    assert (g1.H_of_group("knn").to_dense().cpu() == torch.tensor([[1], [0], [0], [0], [1], [1]])).all()


def test_H_T(g1):
    assert (
        g1.H_T.to_dense().cpu() == torch.tensor([[1, 1, 0], [1, 1, 0], [1, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0]]).t()
    ).all()


def test_H_T_group(g1):
    g1.add_hyperedges([0, 4, 5], group_name="knn")
    assert (
        g1.H_T.to_dense().cpu()
        == torch.tensor([[1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1],]).t()
    ).all()
    assert (
        g1.H_T_of_group("main").to_dense().cpu()
        == torch.tensor([[1, 1, 0], [1, 1, 0], [1, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0]]).t()
    ).all()
    assert (g1.H_T_of_group("knn").to_dense().cpu() == torch.tensor([[1, 0, 0, 0, 1, 1]])).all()


def test_W_v(g2):
    assert (g2.W_v.cpu()._values() == torch.tensor([1, 1, 1, 1, 1])).all()
    hg = Hypergraph(5, [[1, 2], [0, 2, 3, 4]], v_weight=[0.1, 1, 2, 1, 1])
    assert (hg.W_v.cpu()._values() == torch.tensor([0.1, 1, 2, 1, 1])).all()


def test_W_e(g2):
    assert (g2.W_e.cpu()._values() == torch.tensor([0.5, 1, 0.5, 1, 0.5])).all()


def test_W_e_group(g2):
    g2.add_hyperedges([0, 4, 5], group_name="knn")
    assert (g2.W_e.cpu()._values() == torch.tensor([0.5, 1, 0.5, 1, 0.5, 1])).all()
    assert (g2.W_e_of_group("main").cpu()._values() == torch.tensor([0.5, 1, 0.5, 1, 0.5])).all()
    assert (g2.W_e_of_group("knn").cpu()._values() == torch.tensor([1])).all()


def test_D(g1, g2):
    assert (g1.D_v.cpu()._values() == torch.tensor([2, 2, 2, 1, 1, 1])).all()
    assert (g1.D_e.cpu()._values() == torch.tensor([4, 2, 3])).all()
    assert (g2.D_v.cpu()._values() == torch.tensor([1.5, 2, 2, 3, 1])).all()
    assert (g2.D_e.cpu()._values() == torch.tensor([3, 3, 2, 3, 2])).all()


def test_D_group(g1):
    assert (g1.D_v.cpu()._values() == torch.tensor([2, 2, 2, 1, 1, 1])).all()
    assert (g1.D_e.cpu()._values() == torch.tensor([4, 2, 3])).all()
    g1.add_hyperedges([[0, 2], [1, 2, 3]], group_name="knn")
    assert (g1.D_v.cpu()._values() == torch.tensor([3, 3, 4, 2, 1, 1])).all()
    assert (g1.D_e.cpu()._values() == torch.tensor([4, 2, 3, 2, 3])).all()
    assert (g1.D_v_of_group("main").cpu()._values() == torch.tensor([2, 2, 2, 1, 1, 1])).all()
    assert (g1.D_e_of_group("main").cpu()._values() == torch.tensor([4, 2, 3])).all()
    assert (g1.D_v_of_group("knn").cpu()._values() == torch.tensor([1, 1, 2, 1, 0, 0])).all()
    assert (g1.D_e_of_group("knn").cpu()._values() == torch.tensor([2, 3])).all()


def test_D_neg(g1, g2):
    # -1
    assert (g1.D_v_neg_1.cpu()._values() == torch.tensor([2, 2, 2, 1, 1, 1]) ** (-1.0)).all()
    assert (g1.D_e_neg_1.cpu()._values() == torch.tensor([4, 2, 3]) ** (-1.0)).all()
    assert (g2.D_v_neg_1.cpu()._values() == torch.tensor([1.5, 2, 2, 3, 1]) ** (-1.0)).all()
    assert (g2.D_e_neg_1.cpu()._values() == torch.tensor([3, 3, 2, 3, 2]) ** (-1.0)).all()
    # -1/2
    assert (g1.D_v_neg_1_2.cpu()._values() == torch.tensor([2, 2, 2, 1, 1, 1]) ** (-0.5)).all()
    assert (g2.D_v_neg_1_2.cpu()._values() == torch.tensor([1.5, 2, 2, 3, 1]) ** (-0.5)).all()
    # isolated vertex
    g3 = Hypergraph(3, [0, 1])
    assert (g3.D_v_neg_1.cpu()._values() == torch.tensor([1, 1, 0])).all()


def test_D_neg_group(g1):
    # -1
    assert (g1.D_v_neg_1.cpu()._values() == torch.tensor([2, 2, 2, 1, 1, 1]) ** (-1.0)).all()
    assert (g1.D_e_neg_1.cpu()._values() == torch.tensor([4, 2, 3]) ** (-1.0)).all()
    g1.add_hyperedges([[0, 2], [1, 2, 3]], group_name="knn")
    assert (g1.D_v_neg_1.cpu()._values() == torch.tensor([3, 3, 4, 2, 1, 1]) ** (-1.0)).all()
    assert (g1.D_e_neg_1.cpu()._values() == torch.tensor([4, 2, 3, 2, 3]) ** (-1.0)).all()
    assert (g1.D_v_neg_1_of_group("main").cpu()._values() == torch.tensor([2, 2, 2, 1, 1, 1]) ** (-1.0)).all()
    assert (g1.D_e_neg_1_of_group("main").cpu()._values() == torch.tensor([4, 2, 3]) ** (-1.0)).all()
    assert (g1.D_v_neg_1_of_group("knn").cpu()._values() == torch.tensor([1 / 1, 1 / 1, 1 / 2, 1 / 1, 0, 0])).all()
    assert (g1.D_e_neg_1_of_group("knn").cpu()._values() == torch.tensor([2, 3]) ** (-1.0)).all()
    # -1/2
    assert (g1.D_v_neg_1_2.cpu()._values() == torch.tensor([3, 3, 4, 2, 1, 1]) ** (-0.5)).all()
    assert (g1.D_v_neg_1_2_of_group("main").cpu()._values() == torch.tensor([2, 2, 2, 1, 1, 1]) ** (-0.5)).all()
    assert (
        g1.D_v_neg_1_2_of_group("knn").cpu()._values()
        == torch.tensor([1 ** (-0.5), 1 ** (-0.5), 2 ** (-0.5), 1 ** (-0.5), 0, 0])
    ).all()


def test_N(g1, g2):
    assert (g1.N_v(0).cpu() == torch.tensor([0, 1, 2, 5])).all()
    assert (g1.N_e(2).cpu() == torch.tensor([0, 2])).all()
    assert (g2.N_v(1).cpu() == torch.tensor([0, 1, 3])).all()
    assert (g2.N_e(3).cpu() == torch.tensor([0, 1, 3, 4])).all()


def test_N_group(g1):
    assert (g1.N_v(1).cpu() == torch.tensor([0, 1])).all()
    assert (g1.N_e(1).cpu() == torch.tensor([0, 1])).all()
    g1.add_hyperedges([[0, 1], [1, 2]], group_name="knn")
    assert (g1.N_v(1).cpu() == torch.tensor([0, 1])).all()
    assert (g1.N_e(1).cpu() == torch.tensor([0, 1, 3, 4])).all()
    assert (g1.N_v_of_group(1, "main").cpu() == torch.tensor([0, 1])).all()
    assert (g1.N_e_of_group(2, "main").cpu() == torch.tensor([0, 2])).all()
    assert (g1.N_v_of_group(1, "knn").cpu() == torch.tensor([1, 2])).all()
    assert (g1.N_e_of_group(1, "knn").cpu() == torch.tensor([0, 1])).all()


def test_L_HGNN(g1):
    H = g1.H.to_dense().cpu()
    D_v_neg_1_2 = torch.diag(H.sum(dim=1).view(-1) ** (-0.5))
    D_e_neg_1 = torch.diag(H.sum(dim=0).view(-1) ** (-1))
    W_e = g1.W_e.to_dense()
    L_HGNN = D_v_neg_1_2 @ H @ W_e @ D_e_neg_1 @ H.t() @ D_v_neg_1_2
    assert (L_HGNN == g1.L_HGNN.to_dense().cpu()).all()


def test_L_HGNN_group(g1):
    g1.add_hyperedges([[0, 1]], group_name="knn")
    # all
    H = g1.H.to_dense().cpu()
    D_v_neg_1_2 = torch.diag(H.sum(dim=1).view(-1) ** (-0.5))
    D_e_neg_1 = torch.diag(H.sum(dim=0).view(-1) ** (-1))
    W_e = g1.W_e.to_dense()
    L_HGNN = D_v_neg_1_2 @ H @ W_e @ D_e_neg_1 @ H.t() @ D_v_neg_1_2
    assert (L_HGNN == g1.L_HGNN.to_dense().cpu()).all()
    # main group
    H = g1.H_of_group("main").to_dense().cpu()
    D_v_neg_1_2 = torch.diag(H.sum(dim=1).view(-1) ** (-0.5))
    D_e_neg_1 = torch.diag(H.sum(dim=0).view(-1) ** (-1))
    W_e = g1.W_e_of_group("main").to_dense()
    L_HGNN = D_v_neg_1_2 @ H @ W_e @ D_e_neg_1 @ H.t() @ D_v_neg_1_2
    assert (L_HGNN == g1.L_HGNN_of_group("main").to_dense().cpu()).all()
    # knn group
    H = g1.H_of_group("knn").to_dense().cpu()
    D_v_neg_1_2 = H.sum(dim=1).view(-1) ** (-0.5)
    D_v_neg_1_2[torch.isinf(D_v_neg_1_2)] = 0
    D_v_neg_1_2 = torch.diag(D_v_neg_1_2)
    D_e_neg_1 = torch.diag(H.sum(dim=0).view(-1) ** (-1))
    W_e = g1.W_e_of_group("knn").to_dense()
    L_HGNN = D_v_neg_1_2 @ H @ W_e @ D_e_neg_1 @ H.t() @ D_v_neg_1_2
    assert (L_HGNN == g1.L_HGNN_of_group("knn").to_dense().cpu()).all()


def test_smoothing():
    x = torch.rand(10, 5)
    L = torch.rand(10, 10)
    g = Hypergraph(10)
    lbd = 0.1
    assert pytest.approx(g.smoothing(x, L, lbd)) == x + lbd * L @ x


def test_L_sym(g1):
    H = g1.H.to_dense().cpu()
    D_v_neg_1_2 = torch.diag(H.sum(dim=1).view(-1) ** (-0.5))
    D_e_neg_1 = torch.diag(H.sum(dim=0).view(-1) ** (-1))
    W_e = g1.W_e.to_dense()
    L_sym = torch.eye(H.shape[0]) - D_v_neg_1_2 @ H @ W_e @ D_e_neg_1 @ H.t() @ D_v_neg_1_2
    assert (L_sym == g1.L_sym.to_dense().cpu()).all()


def test_L_sym_group(g1):
    g1.add_hyperedges([[0, 1]], group_name="knn")
    # all
    H = g1.H.to_dense().cpu()
    D_v_neg_1_2 = torch.diag(H.sum(dim=1).view(-1) ** (-0.5))
    D_e_neg_1 = torch.diag(H.sum(dim=0).view(-1) ** (-1))
    W_e = g1.W_e.to_dense()
    L_sym = torch.eye(H.shape[0]) - D_v_neg_1_2 @ H @ W_e @ D_e_neg_1 @ H.t() @ D_v_neg_1_2
    assert (L_sym == g1.L_sym.to_dense().cpu()).all()
    # main group
    H = g1.H_of_group("main").to_dense().cpu()
    D_v_neg_1_2 = torch.diag(H.sum(dim=1).view(-1) ** (-0.5))
    D_e_neg_1 = torch.diag(H.sum(dim=0).view(-1) ** (-1))
    W_e = g1.W_e_of_group("main").to_dense()
    L_sym = torch.eye(H.shape[0]) - D_v_neg_1_2 @ H @ W_e @ D_e_neg_1 @ H.t() @ D_v_neg_1_2
    assert (L_sym == g1.L_sym_of_group("main").to_dense().cpu()).all()
    # knn group
    H = g1.H_of_group("knn").to_dense().cpu()
    D_v_neg_1_2 = H.sum(dim=1).view(-1) ** (-0.5)
    D_v_neg_1_2[torch.isinf(D_v_neg_1_2)] = 0
    D_v_neg_1_2 = torch.diag(D_v_neg_1_2)
    D_e_neg_1 = torch.diag(H.sum(dim=0).view(-1) ** (-1))
    W_e = g1.W_e_of_group("knn").to_dense()
    L_sym = torch.eye(H.shape[0]) - D_v_neg_1_2 @ H @ W_e @ D_e_neg_1 @ H.t() @ D_v_neg_1_2
    assert (L_sym == g1.L_sym_of_group("knn").to_dense().cpu()).all()


def test_L_rw(g1):
    H = g1.H.to_dense().cpu()
    D_v_neg_1 = torch.diag(H.sum(dim=1).view(-1) ** (-1))
    D_e_neg_1 = torch.diag(H.sum(dim=0).view(-1) ** (-1))
    W_e = g1.W_e.to_dense()
    L_rw = torch.eye(H.shape[0]) - D_v_neg_1 @ H @ W_e @ D_e_neg_1 @ H.t()
    assert (L_rw == g1.L_rw.to_dense().cpu()).all()


def test_L_rw_group(g1):
    g1.add_hyperedges([[0, 1]], group_name="knn")
    # all
    H = g1.H.to_dense().cpu()
    D_v_neg_1 = torch.diag(H.sum(dim=1).view(-1) ** (-1))
    D_e_neg_1 = torch.diag(H.sum(dim=0).view(-1) ** (-1))
    W_e = g1.W_e.to_dense()
    L_rw = torch.eye(H.shape[0]) - D_v_neg_1 @ H @ W_e @ D_e_neg_1 @ H.t()
    assert (L_rw == g1.L_rw.to_dense().cpu()).all()
    # main group
    H = g1.H_of_group("main").to_dense().cpu()
    D_v_neg_1 = torch.diag(H.sum(dim=1).view(-1) ** (-1))
    D_e_neg_1 = torch.diag(H.sum(dim=0).view(-1) ** (-1))
    W_e = g1.W_e_of_group("main").to_dense()
    L_rw = torch.eye(H.shape[0]) - D_v_neg_1 @ H @ W_e @ D_e_neg_1 @ H.t()
    assert (L_rw == g1.L_rw_of_group("main").to_dense().cpu()).all()
    # knn group
    H = g1.H_of_group("knn").to_dense().cpu()
    D_v_neg_1 = H.sum(dim=1).view(-1) ** (-1)
    D_v_neg_1[torch.isinf(D_v_neg_1)] = 0
    D_v_neg_1 = torch.diag(D_v_neg_1)
    D_e_neg_1 = torch.diag(H.sum(dim=0).view(-1) ** (-1))
    W_e = g1.W_e_of_group("knn").to_dense()
    L_rw = torch.eye(H.shape[0]) - D_v_neg_1 @ H @ W_e @ D_e_neg_1 @ H.t()
    assert (L_rw == g1.L_rw_of_group("knn").to_dense().cpu()).all()


def test_smoothing_with_HGNN(g1):
    H = torch.tensor([[1, 1, 0], [1, 1, 0], [1, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0]], dtype=torch.float32,)
    D_v_inv_1_2 = H.sum(1).view(-1) ** (-0.5)
    D_v_inv_1_2[torch.isinf(D_v_inv_1_2)] = 0
    D_v_inv_1_2 = torch.diag(D_v_inv_1_2)

    D_e_inv = H.sum(0).view(-1) ** (-1)
    D_e_inv[torch.isinf(D_e_inv)] = 0
    D_e_inv = torch.diag(D_e_inv)

    x = torch.rand(H.shape[0], 8)

    gt = D_v_inv_1_2 @ H @ D_e_inv @ H.t() @ D_v_inv_1_2 @ x

    res = g1.smoothing_with_HGNN(x)

    assert pytest.approx(gt, rel=1e-6) == res.cpu()


def test_smoothing_with_HGNN_group(g1):
    H = torch.tensor([[1, 1, 0], [1, 1, 0], [1, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0]], dtype=torch.float32,)
    D_v_inv_1_2 = H.sum(1).view(-1) ** (-0.5)
    D_v_inv_1_2[torch.isinf(D_v_inv_1_2)] = 0
    D_v_inv_1_2 = torch.diag(D_v_inv_1_2)

    D_e_inv = H.sum(0).view(-1) ** (-1)
    D_e_inv[torch.isinf(D_e_inv)] = 0
    D_e_inv = torch.diag(D_e_inv)

    x = torch.rand(H.shape[0], 8)

    gt = D_v_inv_1_2 @ H @ D_e_inv @ H.t() @ D_v_inv_1_2 @ x

    res = g1.smoothing_with_HGNN_of_group("main", x)

    assert pytest.approx(gt, rel=1e-6) == res.cpu()


def test_v2e_message_passing(g1):
    H = torch.tensor([[1, 1, 0], [1, 1, 0], [1, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0]], dtype=torch.float32,)

    x = torch.rand(H.shape[0], 8)

    gt_sum = H.t() @ x
    res_sum = g1.v2e(x, aggr="sum")
    assert pytest.approx(gt_sum, rel=1e-6) == res_sum.cpu()

    D_e_inv = H.sum(0).view(-1) ** (-1)
    D_e_inv[torch.isinf(D_e_inv)] = 0
    D_e_inv = torch.diag(D_e_inv)

    gt_mean = D_e_inv @ gt_sum
    res_mean = g1.v2e(x, aggr="mean")
    assert pytest.approx(gt_mean, rel=1e-6) == res_mean.cpu()


def test_e2v_message_passing(g1):
    H = torch.tensor([[1, 1, 0], [1, 1, 0], [1, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0]], dtype=torch.float32,)

    x = torch.rand(3, 8)

    gt_sum = H @ x
    res_sum = g1.e2v(x, aggr="sum")
    assert pytest.approx(gt_sum, rel=1e-6) == res_sum.cpu()

    D_v_inv = H.sum(1).view(-1) ** (-1)
    D_v_inv[torch.isinf(D_v_inv)] = 0
    D_v_inv = torch.diag(D_v_inv)

    gt_mean = D_v_inv @ gt_sum
    res_mean = g1.e2v(x, aggr="mean")
    assert pytest.approx(gt_mean, rel=1e-6) == res_mean.cpu()


def test_v2v_message_passing(g1):
    H = torch.tensor([[1, 1, 0], [1, 1, 0], [1, 0, 1], [0, 0, 1], [0, 0, 1], [1, 0, 0]], dtype=torch.float32,)

    x = torch.rand(6, 8)

    gt_sum = H @ H.t() @ x
    res_sum = g1.v2v(x, aggr="sum")
    assert pytest.approx(gt_sum, rel=1e-6) == res_sum.cpu()

    D_v_inv = H.sum(1).view(-1) ** (-1)
    D_v_inv[torch.isinf(D_v_inv)] = 0
    D_v_inv = torch.diag(D_v_inv)

    D_e_inv = H.sum(0).view(-1) ** (-1)
    D_e_inv[torch.isinf(D_e_inv)] = 0
    D_e_inv = torch.diag(D_e_inv)

    gt_mean = D_v_inv @ H @ D_e_inv @ H.t() @ x
    res_mean = g1.v2v(x, aggr="mean")
    assert pytest.approx(gt_mean, rel=1e-6) == res_mean.cpu()


def test_graph_and_hypergraph():
    g = Graph(4, [[0, 1], [0, 2], [1, 3]])
    hg = Hypergraph.from_graph(g)
    _mm = torch.sparse.mm
    est_A = _mm(_mm(g.D_v_neg_1_2, g.A), g.D_v_neg_1_2) + torch.eye(4).to_sparse()
    assert pytest.approx(est_A.to_dense() / 2) == hg.L_HGNN.to_dense()
