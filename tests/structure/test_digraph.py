import random
from random import randrange
import numpy as np
import scipy
import torch
import pytest
from dhg import DiGraph
from dhg.random import digraph_Gnm


@pytest.fixture()
def g1():
    e_list = [(0, 1), (0, 2)]
    g = DiGraph(4, e_list)
    return g


@pytest.fixture()
def g2():
    e_list = [(0, 1), (0, 2), (0, 3)]
    e_weight = [0.5, 1, 0.5]
    g = DiGraph(4, e_list, e_weight)
    return g


def test_save(g1):
    from dhg import load_structure
    from pathlib import Path

    save_path = Path("./tmp/test_save.dhg")
    if not save_path.parent.exists():
        save_path.parent.mkdir()
    g1.save(save_path)
    g2 = load_structure(save_path)

    for e1, e2 in zip(g1.e[0], g2.e[0]):
        assert e1 == e2
    for w1, w2 in zip(g1.e[1], g2.e[1]):
        assert w1 == w2
    save_path.unlink()


def test_empty():
    g = DiGraph(5)
    assert g.num_v == 5
    assert g.e == ([], [])


def test_init(g1, g2):
    assert g1.num_v == 4
    assert g1.num_e == 2
    assert (0, 1) in g1.e[0]
    assert g1.A[0, 1] == 1
    assert (1, 0) not in g1.e[0]
    assert g2.num_v == 4
    assert g2.num_e == 3
    assert (0, 3) in g2.e[0]
    assert g2.A[0, 3] == 0.5
    assert (0, 2) in g2.e[0]
    assert g2.A[0, 2] == 1
    assert (3, 0) not in g2.e[0]
    assert (2, 0) not in g2.e[0]


# test construction
def test_from_adj_list():
    num_v = 6
    adj_list = [[0, 1, 2], [1, 3, 4, 5], [2, 2, 1], [3, 2, 2], [], []]
    g = DiGraph.from_adj_list(num_v, adj_list)
    e = g.e[0]
    assert (0, 1) in e
    assert (0, 2) in e
    assert (1, 3) in e
    assert (1, 4) in e
    assert (1, 5) in e
    assert (2, 2) in e
    assert (2, 1) in e
    assert (3, 2) in e

    assert len(e) == 8


def test_from_feature_kNN():
    ft = np.random.rand(32, 8)
    cdist = scipy.spatial.distance.cdist(ft, ft)
    tk_mat = np.argsort(cdist, axis=1)[:, :3].tolist()
    g = DiGraph.from_feature_kNN(torch.tensor(ft), k=3, include_center=True)
    for src, dst_set in enumerate(tk_mat):
        assert all((src, dst) in g.e[0] for dst in dst_set)

    tk_mat = np.argsort(cdist, axis=1)[:, 1:4].tolist()
    g = DiGraph.from_feature_kNN(torch.tensor(ft), k=3, include_center=False)
    for src, dst_set in enumerate(tk_mat):
        assert all((src, dst) in g.e[0] for dst in dst_set)


# test modification
def test_add_edges(g1, g2):
    assert g1.num_e == 2
    g1.add_edges((3, 2))
    assert g1.num_e == 3
    assert (3, 2) in g1.e[0]
    assert g1.A[3, 2] == 1
    assert g2.num_e == 3
    g2.add_edges(((1, 2), (1, 3)))
    assert g2.num_e == 5
    assert (1, 2) in g2.e[0]
    assert g2.A[1, 2] == 1
    assert (2, 1) not in g2.e[0]
    g2.add_edges(((3, 2), (3, 1)))
    assert g2.num_e == 7
    assert (3, 2) in g2.e[0]
    assert g2.A[3, 2] == 1
    assert (2, 3) not in g2.e[0]


def test_remove_edges(g1):
    assert (0, 1) in g1.e[0]
    assert g1.A[0, 1] == 1
    g1.remove_edges((0, 1))
    assert (0, 1) not in g1.e[0]


def test_reverse_direction(g1):
    g1.reverse_direction()
    assert g1.num_e == 2
    assert (1, 0) in g1.e[0]
    assert (2, 0) in g1.e[0]


# test properties
def test_pred(g1):
    assert g1.nbr_v_in(0) == []
    assert g1.nbr_v_in(1) == [0]
    assert g1.nbr_v_in(2) == [0]
    assert g1.nbr_v_in(3) == []
    g1.add_edges((3, 0))
    assert g1.nbr_v_in(0) == [3]
    g1.remove_edges((0, 2))
    assert g1.nbr_v_in(2) == []


def test_succ(g1):
    assert g1.nbr_v_out(0) == [1, 2]
    assert g1.nbr_v_out(1) == []
    assert g1.nbr_v_out(2) == []
    assert g1.nbr_v_out(3) == []
    g1.add_edges((3, 0))
    assert g1.nbr_v_out(0) == [1, 2]
    g1.remove_edges((0, 2))
    assert g1.nbr_v_out(0) == [1]


def test_deg_in(g1):
    assert g1.deg_v_in[0] == 0
    assert g1.deg_v_in[1] == 1
    assert g1.D_v_in_neg_1._values()[1] == 1
    assert pytest.approx(g1.D_v_in_neg_1._values()[3]) == 0
    g1.add_extra_selfloop()
    assert g1.deg_v_in[0] == 1
    assert g1.deg_v_in[1] == 2
    assert g1.D_v_in_neg_1._values()[1] == 0.5
    assert g1.D_v_in_neg_1._values()[3] == 1


def test_deg_out(g1):
    assert g1.deg_v_out[0] == 2
    assert g1.deg_v_out[1] == 0
    assert pytest.approx(g1.D_v_out_neg_1._values()[1]) == 0
    assert pytest.approx(g1.D_v_out_neg_1._values()[3]) == 0
    g1.add_extra_selfloop()
    assert g1.deg_v_out[0] == 3
    assert g1.deg_v_out[1] == 1
    assert g1.D_v_out_neg_1._values()[1] == 1
    assert g1.D_v_out_neg_1._values()[3] == 1


def test_nbr_v_in(g1):
    assert g1.nbr_v_in(0) == []
    assert g1.nbr_v_in(1) == [0]
    assert g1.nbr_v_in(2) == [0]
    assert g1.nbr_v_in(3) == []


def test_nbr_v_out(g1):
    assert g1.nbr_v_out(0) == [1, 2]
    assert g1.nbr_v_out(1) == []
    assert g1.nbr_v_out(2) == []
    assert g1.nbr_v_out(3) == []


# test deep learning
def test_clone(g1):
    assert g1.num_e == 2
    graph1_copy = g1.clone()
    graph1_copy.add_edges((3, 2))
    assert graph1_copy.num_e == 3
    assert g1.num_e == 2


def test_A():
    for _ in range(10):
        num_v = 100
        num_e = 500

        gt_A = torch.zeros((num_v, num_v))
        g = DiGraph(num_v)

        while g.num_e < num_e:
            src, dst = random.randrange(num_v), random.randrange(num_v)

            g.add_edges((src, dst))
            gt_A[src, dst] = 1

        A = g.A.to_dense()
        assert torch.all(A == gt_A)


def test_D():
    for _ in range(10):
        num_v = 100
        num_e = 500

        gt_A = torch.zeros((num_v, num_v))
        g = DiGraph(num_v)

        while g.num_e < num_e:
            src, dst = random.randrange(num_v), random.randrange(num_v)

            g.add_edges((src, dst))
            gt_A[src, dst] = 1

        D_v_in = g.D_v_in.to_dense()
        D_v_out = g.D_v_out.to_dense()
        D_v_in_neg_1 = g.D_v_in_neg_1.to_dense()
        D_v_out_neg_1 = g.D_v_out_neg_1.to_dense()

        gt_d_v_in = torch.diag(gt_A.sum(0))
        gt_d_v_out = torch.diag(gt_A.sum(1))
        gt_d_v_in_neg_1 = 1 / (gt_d_v_in)
        gt_d_v_in_neg_1[torch.isinf(gt_d_v_in_neg_1)] = 0
        gt_d_v_out_neg_1 = 1 / (gt_d_v_out)
        gt_d_v_out_neg_1[torch.isinf(gt_d_v_out_neg_1)] = 0

        assert torch.all(D_v_in == gt_d_v_in)
        assert torch.all(D_v_out == gt_d_v_out)
        assert pytest.approx(D_v_in_neg_1) == gt_d_v_in_neg_1
        assert pytest.approx(D_v_out_neg_1) == gt_d_v_out_neg_1


def test_N():
    for _ in range(10):
        num_v = 20
        num_e = 40

        gt_A = torch.zeros((num_v, num_v))
        e_list = []
        for _ in range(num_e):
            src, dst = random.randrange(num_v), random.randrange(num_v)
            e_list.append((src, dst))
            gt_A[src, dst] = 1
        g = DiGraph(num_v, e_list)

        for v_idx in range(num_v):
            gt_in = np.where(gt_A[:, v_idx].view(-1).long().numpy())[0].tolist()
            gt_in = tuple(sorted(gt_in))

            n_v_in = g.N_v_in(v_idx).view(-1).numpy().tolist()
            n_v_in = tuple(sorted(n_v_in))

            assert gt_in == n_v_in

            gt_out = np.where(gt_A[v_idx, :].view(-1).long().numpy())[0].tolist()
            gt_out = tuple(sorted(gt_out))

            n_v_out = g.N_v_out(v_idx).view(-1).numpy().tolist()
            n_v_out = tuple(sorted(n_v_out))

            assert gt_out == n_v_out


def test_smoothing():
    num_v = 200
    num_e = 500
    x = torch.rand((num_v, 10))
    for _ in range(3):
        e_list = []
        # A = torch.zeros((num_u + num_v, num_u + num_v))
        for i in range(num_e):
            u, v = randrange(num_v), randrange(num_v)
            e_list.append((u, v))
            # A[u, v + num_u] = 1
            # A[v + num_u, u] = 1


        g = DiGraph(num_v)
        g.add_edges(e_list)

        L = g.A
        lbd = 0.1

        assert pytest.approx(x + lbd * L @ x) == g.smoothing(x, L, lbd).to_dense()


# test message passing
def test_v2v(g1):
    x = torch.rand(4, 8)

    y = g1.v2v(x, direction="src2dst")
    assert y.shape == (4, 8)

    y = g1.v2v(x)
    assert y.shape == (4, 8)
    y = g1.v2v(x, "mean")
    assert y.shape == (4, 8)

    w = torch.rand_like(g1.e_weight)
    y = g1.v2v(x, e_weight=w)
    assert y.shape == (4, 8)
    y = g1.v2v(x, "mean", w)
    assert y.shape == (4, 8)

    y = g1.v2v(x)
    assert y.shape == (4, 8)
    y = g1.v2v(x, "sum")
    assert y.shape == (4, 8)

    w = torch.rand_like(g1.e_weight)
    y = g1.v2v(x, e_weight=w)
    assert y.shape == (4, 8)
    y = g1.v2v(x, "sum", w)
    assert y.shape == (4, 8)

    y = g1.v2v(x)
    assert y.shape == (4, 8)
    y = g1.v2v(x, "softmax_then_sum")
    assert y.shape == (4, 8)

    w = torch.rand_like(g1.e_weight)
    y = g1.v2v(x, e_weight=w)
    assert y.shape == (4, 8)
    y = g1.v2v(x, "softmax_then_sum", w)
    assert y.shape == (4, 8)


def test_drop_edges():
    g = digraph_Gnm(100, 200)
    gg = g.drop_edges(0.1)
    assert pytest.approx(gg.num_e, rel=0.1) == 180
