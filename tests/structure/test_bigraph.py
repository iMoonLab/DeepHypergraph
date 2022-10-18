from random import choices, random, randrange, choice
import torch
import pytest
from dhg import BiGraph
from dhg.random import bigraph_Gnm


@pytest.fixture()
def g1():
    e_list = [(0, 1), (0, 2), (1, 2), (2, 2), (3, 4)]
    g = BiGraph(4, 5, e_list)
    return g


@pytest.fixture()
def g2():
    e_list = [(0, 3), (0, 1), (1, 3), (1, 2)]
    e_weight = [0.5, 1, 0.5, 0.5]
    g = BiGraph(3, 4, e_list, e_weight)
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
    g = BiGraph(5, 7)
    assert g.num_u == 5
    assert g.num_v == 7
    assert g.e == ([], [])


def test_init(g1, g2):
    assert g1.num_u == 4
    assert g1.num_v == 5

    for e in [(0, 1), (0, 2), (1, 2), (2, 2), (3, 4)]:
        assert e in g1.e[0]

    assert g2.num_u == 3
    assert g2.num_v == 4

    for e in [(0, 3), (0, 1), (1, 3), (1, 2)]:
        assert e in g2.e[0]

    assert g2.e[1] == [0.5, 1, 0.5, 0.5]


# test construction
def test_from_adj_list():
    num_u, num_v = 100, 200
    num_e = 500
    for _ in range(3):
        adj_list = [[] for _ in range(num_u)]
        e_list = []
        for i in range(num_e):
            u, v = randrange(num_u), randrange(num_v)
            e_list.append((u, v))
            if adj_list[u] == []:
                adj_list[u].append(u)
            adj_list[u].append(v)

        g = BiGraph.from_adj_list(num_u, num_v, adj_list)

        for e in e_list:
            assert e in g.e[0]
        for e in g.e[0]:
            assert e in e_list


def test_from_hypergraph():
    from dhg.random import hypergraph_Gnm

    for _ in range(3):
        hg = hypergraph_Gnm(100, 200)
        bg = BiGraph.from_hypergraph(hg)
        assert bg.num_u == hg.num_v
        assert bg.num_v == hg.num_e

        for e_id, e in enumerate(hg.e[0]):
            for v in e:
                assert (v, e_id) in bg.e[0]


# test modification
def test_add_edges():
    num_u, num_v = 100, 200
    num_e = 500
    for _ in range(3):
        g = BiGraph(num_u, num_v)
        e_list = []
        for i in range(num_e):
            u, v = randrange(num_u), randrange(num_v)
            e_list.append((u, v))
            g.add_edges((u, v))

        for e in e_list:
            assert e in g.e[0]
        for e in g.e[0]:
            assert e in e_list


def test_remove_edges():
    num_u, num_v = 100, 200
    num_e = 500
    for _ in range(3):
        g = BiGraph(num_u, num_v)
        e_list = []
        for i in range(num_e):
            u, v = randrange(num_u), randrange(num_v)
            e_list.append((u, v))
            g.add_edges((u, v))

        removed = choices(e_list, k=num_e // 2)
        for re in removed:
            g.remove_edges(re)

        e_list = list(set(e_list) - set(removed))

        for e in e_list:
            assert e in g.e[0]
        for e in g.e[0]:
            assert e in e_list


def test_switch_uv():
    num_u, num_v = 10, 20
    num_e = 10
    for _ in range(3):
        e_list = []
        for i in range(num_e):
            u, v = randrange(num_u), randrange(num_v)
            e_list.append((u, v))

        g = BiGraph(num_u, num_v)
        g.add_edges(e_list)
        gg = g.switch_uv()

        assert g.num_u == gg.num_v
        assert g.num_v == gg.num_u

        for (u, v) in g.e[0]:
            assert (v, u) in gg.e[0]
        for (u, v) in gg.e[0]:
            assert (v, u) in g.e[0]


# test properties
def test_deg_u():
    num_u, num_v = 100, 200
    num_e = 500
    for _ in range(3):
        e_list = []
        A = torch.zeros((num_u, num_v))
        for i in range(num_e):
            u, v = randrange(num_u), randrange(num_v)
            e_list.append((u, v))
            A[u, v] = 1

        g = BiGraph(num_u, num_v)
        g.add_edges(e_list)

        gt = A.sum(1).numpy().tolist()
        assert all(gt_deg == deg for gt_deg, deg in zip(gt, g.deg_u))


def test_deg_v():
    num_u, num_v = 100, 200
    num_e = 500
    for _ in range(3):
        e_list = []
        A = torch.zeros((num_u, num_v))
        for i in range(num_e):
            u, v = randrange(num_u), randrange(num_v)
            e_list.append((u, v))
            A[u, v] = 1

        g = BiGraph(num_u, num_v)
        g.add_edges(e_list)

        gt = A.sum(0).numpy().tolist()
        assert all(gt_deg == deg for gt_deg, deg in zip(gt, g.deg_v))


def test_nbr_u():
    num_u, num_v = 100, 200
    num_e = 500
    for _ in range(3):
        e_list = []
        A = torch.zeros((num_u, num_v))
        for i in range(num_e):
            u, v = randrange(num_u), randrange(num_v)
            e_list.append((u, v))
            A[u, v] = 1

        g = BiGraph(num_u, num_v)
        g.add_edges(e_list)

        for v_idx in range(num_v):
            gt_v_nbr = A[:, v_idx].nonzero().view(-1).numpy().tolist()
            assert all(gt_v_n == v_n for gt_v_n, v_n in zip(gt_v_nbr, g.nbr_u(v_idx)))


def test_nbr_v():
    num_u, num_v = 100, 200
    num_e = 500
    for _ in range(3):
        e_list = []
        A = torch.zeros((num_u, num_v))
        for i in range(num_e):
            u, v = randrange(num_u), randrange(num_v)
            e_list.append((u, v))
            A[u, v] = 1

        g = BiGraph(num_u, num_v)
        g.add_edges(e_list)

        for u_idx in range(num_u):
            gt_u_nbr = A[u_idx].nonzero().view(-1).numpy().tolist()
            assert all(gt_u_n == u_n for gt_u_n, u_n in zip(gt_u_nbr, g.nbr_v(u_idx)))


# test deep learning
def test_clone(g1):
    g2 = g1.clone()
    assert g1.num_u == g2.num_u
    assert g1.num_v == g2.num_v
    assert all(e1 == e2 for e1, e2 in zip(g1.e[0], g2.e[0]))
    assert all(w1 == w2 for w1, w2 in zip(g1.e[1], g2.e[1]))


def test_A():
    num_u, num_v = 100, 200
    num_e = 500
    for _ in range(3):
        e_list = []
        A = torch.zeros((num_u + num_v, num_u + num_v))
        for i in range(num_e):
            u, v = randrange(num_u), randrange(num_v)
            e_list.append((u, v))
            A[u, v + num_u] = 1
            A[v + num_u, u] = 1

        g = BiGraph(num_u, num_v)
        g.add_edges(e_list)

        assert pytest.approx(A) == g.A.to_dense()


def test_B():
    num_u, num_v = 100, 200
    num_e = 500
    for _ in range(3):
        e_list = []
        B = torch.zeros((num_u, num_v))
        for i in range(num_e):
            u, v = randrange(num_u), randrange(num_v)
            e_list.append((u, v))
            B[u, v] = 1

        g = BiGraph(num_u, num_v)
        g.add_edges(e_list)

        assert pytest.approx(B) == g.B.to_dense()
        assert pytest.approx(B.t()) == g.B_T.to_dense()


def test_D():
    num_u, num_v = 100, 200
    num_e = 500
    for _ in range(3):
        e_list = []
        B = torch.zeros((num_u, num_v))
        for i in range(num_e):
            u, v = randrange(num_u), randrange(num_v)
            e_list.append((u, v))
            B[u, v] = 1

        g = BiGraph(num_u, num_v)
        g.add_edges(e_list)

        assert pytest.approx(torch.diag(B.sum(1))) == g.D_u.to_dense()
        assert pytest.approx(torch.diag(B.sum(0))) == g.D_v.to_dense()


def test_D_neg_1(g1):
    num_u, num_v = 100, 200
    num_e = 500
    for _ in range(3):
        e_list = []
        B = torch.zeros((num_u, num_v))
        for i in range(num_e):
            u, v = randrange(num_u), randrange(num_v)
            e_list.append((u, v))
            B[u, v] = 1

        g = BiGraph(num_u, num_v)
        g.add_edges(e_list)

        d_u_neg_1 = 1 / B.sum(1)
        d_u_neg_1[torch.isinf(d_u_neg_1)] = 0
        d_u_neg_1 = torch.diag(d_u_neg_1)

        d_v_neg_1 = 1 / B.sum(0)
        d_v_neg_1[torch.isinf(d_v_neg_1)] = 0
        d_v_neg_1 = torch.diag(d_v_neg_1)

        assert pytest.approx(d_u_neg_1) == g.D_u_neg_1.to_dense()
        assert pytest.approx(d_v_neg_1) == g.D_v_neg_1.to_dense()


def test_N():
    num_u, num_v = 100, 200
    num_e = 500
    for _ in range(3):
        e_list = []
        B = torch.zeros((num_u, num_v))
        for i in range(num_e):
            u, v = randrange(num_u), randrange(num_v)
            e_list.append((u, v))
            B[u, v] = 1

        g = BiGraph(num_u, num_v)
        g.add_edges(e_list)

        assert all(torch.all(B[u_idx].nonzero().view(-1) == g.N_v(u_idx)) for u_idx in range(num_u))
        assert all(torch.all(B[:, v_idx].nonzero().view(-1) == g.N_u(v_idx)) for v_idx in range(num_v))


def test_smoothing():
    num_u, num_v = 100, 200
    num_e = 500
    x = torch.rand((num_u + num_v, 10))
    for _ in range(3):
        e_list = []
        # A = torch.zeros((num_u + num_v, num_u + num_v))
        for i in range(num_e):
            u, v = randrange(num_u), randrange(num_v)
            e_list.append((u, v))
            # A[u, v + num_u] = 1
            # A[v + num_u, u] = 1


        g = BiGraph(num_u, num_v)
        g.add_edges(e_list)

        L = g.L_GCN
        lbd = 0.1

        assert pytest.approx(x + lbd * L @ x) == g.smoothing(x, L, lbd).to_dense()


# test spactral-based smoothing matrix
def test_L_GCN():
    num_u, num_v = 100, 200
    num_e = 500
    for _ in range(3):
        e_list = []
        A = torch.zeros((num_u + num_v, num_u + num_v))
        for i in range(num_e):
            u, v = randrange(num_u), randrange(num_v)
            e_list.append((u, v))
            A[u, v + num_u] = 1
            A[v + num_u, u] = 1

        for idx in range(num_u + num_v):
            A[idx, idx] += 1

        g = BiGraph(num_u, num_v)
        g.add_edges(e_list)

        D = A.sum(0)
        D_inv_1_2 = D ** -0.5
        D_inv_1_2[torch.isinf(D_inv_1_2)] = 0
        D_inv_1_2 = torch.diag(D_inv_1_2.view(-1))
        L = D_inv_1_2 @ A @ D_inv_1_2

        assert pytest.approx(L) == g.L_GCN.to_dense()


# test message passing
def test_u2v():
    num_u, num_v = 100, 200
    num_e = 500
    x = torch.rand((num_u, 8))
    for _ in range(3):
        e_list = []
        B = torch.zeros((num_u, num_v))
        for i in range(num_e):
            u, v = randrange(num_u), randrange(num_v)
            e_list.append((u, v))
            B[u, v] = 1

        g = BiGraph(num_u, num_v)
        g.add_edges(e_list)

        D_v = B.sum(0)
        D_v_neg_1 = 1 / D_v
        D_v_neg_1[torch.isinf(D_v_neg_1)] = 0
        D_v_neg_1 = torch.diag(D_v_neg_1)

        sum_u2v = B.t() @ x
        mean_u2v = D_v_neg_1 @ sum_u2v

        assert pytest.approx(sum_u2v) == g.u2v(x, "sum")
        assert pytest.approx(mean_u2v) == g.u2v(x, "mean")


def test_v2u():
    num_u, num_v = 100, 200
    num_e = 500
    x = torch.rand((num_v, 8))
    for _ in range(3):
        e_list = []
        B = torch.zeros((num_u, num_v))
        for i in range(num_e):
            u, v = randrange(num_u), randrange(num_v)
            e_list.append((u, v))
            B[u, v] = 1

        g = BiGraph(num_u, num_v)
        g.add_edges(e_list)

        D_u = B.sum(1)
        D_u_neg_1 = 1 / D_u
        D_u_neg_1[torch.isinf(D_u_neg_1)] = 0
        D_u_neg_1 = torch.diag(D_u_neg_1)

        sum_v2u = B @ x
        mean_v2u = D_u_neg_1 @ sum_v2u

        assert pytest.approx(sum_v2u) == g.v2u(x, "sum")
        assert pytest.approx(mean_v2u) == g.v2u(x, "mean")


def test_drop_edges():
    g = bigraph_Gnm(100, 200, 500)
    gg = g.drop_edges(0.1)
    assert pytest.approx(gg.num_e, rel=0.1) == 450
