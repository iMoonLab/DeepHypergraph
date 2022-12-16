import torch
import pytest
import random

from dhg import Graph
from dhg.random import graph_Gnm


@pytest.fixture()
def g1():
    e_list = [(0, 1), (0, 2)]
    g = Graph(4, e_list)
    return g


@pytest.fixture()
def g2():
    e_list = [(0, 1), (0, 2), (0, 3)]
    e_weight = [0.5, 1, 0.5]
    g = Graph(4, e_list, e_weight)
    return g


def test_save(g1, tmp_path):
    from dhg import load_structure

    g1.save(tmp_path / "g1")
    g2 = load_structure(tmp_path / "g1")

    for e1, e2 in zip(g1.e[0], g2.e[0]):
        assert e1 == e2
    for w1, w2 in zip(g1.e[1], g2.e[1]):
        assert w1 == w2


# test representation
def test_empty():
    g = Graph(5)
    assert g.num_v == 5
    assert g.e == ([], [])


def test_init(g1, g2):
    assert g1.num_v == 4
    assert g1.num_e == 2
    assert (0, 1) in g1.e[0]
    assert g1.A[0, 1] == 1
    assert (1, 0) in g1.e_both_side[0]
    assert g1.A[1, 0] == 1
    assert g2.num_v == 4
    assert g2.num_e == 3
    assert (0, 3) in g2.e[0]
    assert g2.A[0, 3] == 0.5
    assert (0, 2) in g2.e[0]
    assert g2.A[0, 2] == 1
    assert (3, 0) in g2.e_both_side[0]
    assert g2.A[3, 0] == 0.5
    assert (2, 0) in g2.e_both_side[0]
    assert g2.A[2, 0] == 1


def test_clear(g1):
    g1.clear()
    assert g1.num_e == 0
    assert all([deg == 0 for deg in g1.deg_v])
    e_list, w_list = g1.e
    assert len(e_list) == 0 and len(w_list) == 0
    assert g1.e_dst.shape[0] == 0
    assert g1.e_src.shape[0] == 0
    assert g1.e_weight.shape[0] == 0


def test_add_edges(g1, g2):
    assert g1.num_e == 2
    g1.add_edges((3, 2))
    assert g1.num_e == 3
    assert (2, 3) in g1.e[0]
    assert (3, 2) not in g1.e[0]
    assert g1.A[3, 2] == 1
    assert g2.num_e == 3
    g2.add_edges(((1, 2), (1, 3)))
    assert g2.num_e == 5
    assert (1, 2) in g2.e[0]
    assert g2.A[1, 2] == 1
    assert (2, 1) in g2.e_both_side[0]
    assert g2.A[2, 1] == 1
    g2.add_edges(((3, 2), (3, 1)))
    assert g2.num_e == 6
    assert (2, 3) in g2.e[0]
    assert g2.A[3, 2] == 1
    assert (2, 3) in g2.e_both_side[0]
    assert g2.A[2, 3] == 1
    g2.add_edges(((3, 2), (3, 1)), merge_op="sum")
    assert g2.num_e == 6
    assert g2.A[3, 2] == 2
    assert g2.A[2, 3] == 2


def test_add_edges_sum(g1, g2):
    assert g1.num_e == 2
    g1.add_edges((3, 2), e_weight=0.5, merge_op="sum")
    assert g1.num_e == 3
    assert (2, 3) in g1.e[0]
    assert (3, 2) not in g1.e[0]
    assert g1.A[3, 2] == 0.5

    assert g2.num_e == 3
    g2.add_edges(((1, 2), (1, 3)), e_weight=[0.1, 0.2], merge_op="sum")
    assert g2.num_e == 5
    assert (1, 2) in g2.e[0]
    assert g2.A[1, 2] == 0.1
    assert (2, 1) in g2.e_both_side[0]
    assert g2.A[2, 1] == 0.1
    g2.add_edges(((3, 2), (3, 1)), e_weight=[1.1, 2.1], merge_op="sum")
    assert g2.num_e == 6
    assert (2, 3) in g2.e[0]
    assert g2.A[3, 2] == 1.1
    assert (2, 3) in g2.e_both_side[0]
    assert g2.A[2, 3] == 1.1
    assert g2.A[1, 3] == 2.3


def test_add_edges_max(g1, g2):
    assert g1.num_e == 2
    g1.add_edges((3, 2), e_weight=0.5, merge_op="max")
    assert g1.num_e == 3
    assert (2, 3) in g1.e[0]
    assert (3, 2) not in g1.e[0]
    assert g1.A[3, 2] == 0.5

    assert g2.num_e == 3
    g2.add_edges(((1, 2), (1, 3)), e_weight=[0.1, 0.2], merge_op="max")
    assert g2.num_e == 5
    assert (1, 2) in g2.e[0]
    assert g2.A[1, 2] == 0.1
    assert (2, 1) in g2.e_both_side[0]
    assert g2.A[2, 1] == 0.1
    g2.add_edges(((3, 2), (3, 1)), e_weight=[1.1, 2.1], merge_op="max")
    assert g2.num_e == 6
    assert (2, 3) in g2.e[0]
    assert g2.A[3, 2] == 1.1
    assert (2, 3) in g2.e_both_side[0]
    assert g2.A[2, 3] == 1.1
    assert g2.A[1, 3] == 2.1


def test_remove_edges(g1):
    assert (0, 1) in g1.e[0]
    assert g1.A[0, 1] == 1
    g1.remove_edges((0, 1))
    assert (0, 1) not in g1.e[0]
    assert (1, 0) not in g1.e_both_side[0]


def test_deg(g1):
    assert g1.deg_v[0] == 2
    assert g1.deg_v[1] == 1
    assert g1.deg_v[2] == 1
    assert g1.deg_v[3] == 0
    g1.add_edges((3, 0))
    assert g1.deg_v[0] == 3
    assert g1.deg_v[3] == 1
    g1.remove_edges((0, 2))
    assert g1.deg_v[2] == 0
    assert g1.deg_v[0] == 2


def test_nbr(g1):
    assert g1.nbr_v(0) == [1, 2]
    assert g1.nbr_v(1) == [0]
    assert g1.nbr_v(2) == [0]
    assert g1.nbr_v(3) == []
    g1.add_edges((3, 0))
    assert g1.nbr_v(0) == [1, 2, 3]
    g1.remove_edges((0, 2))
    assert g1.nbr_v(2) == []
    # hop k
    g3 = Graph(5, [(0, 1), (0, 3), (1, 4), (2, 3)])
    assert sorted(g3.nbr_v(3, 1)) == [0, 2]
    assert sorted(g3.nbr_v(3, 2)) == [1, 3]
    assert sorted(g3.nbr_v(3, 3)) == [0, 2, 4]


# test deep learning
def test_clone(g1):
    assert g1.num_e == 2
    g1_copy = g1.clone()
    g1_copy.add_edges((3, 2))
    assert g1_copy.num_e == 3
    assert g1.num_e == 2


def test_A():
    num_v = 20
    num_e = 50
    import random

    for _ in range(3):
        g = Graph(num_v)
        A = torch.zeros((num_v, num_v))
        for _ in range(num_e):
            s = random.randrange(num_v)
            d = random.randrange(num_v)
            g.add_edges((s, d))
            A[s, d] = 1
            A[d, s] = 1
        assert torch.all(g.A.to_dense() == A)

    for _ in range(3):
        g = Graph(num_v)
        A = torch.zeros((num_v, num_v))
        for _ in range(num_e):
            s = random.randrange(num_v)
            d = random.randrange(num_v)
            g.add_edges((s, d), merge_op="sum")
            if s == d:
                A[s, d] += 1
            else:
                A[s, d] += 1
                A[d, s] += 1
        assert torch.all(g.A.to_dense() == A)


def test_D(g1):
    assert g1.D_v[0, 0].item() == 2
    assert g1.D_v[1, 1].item() == 1
    assert g1.D_v_neg_1[1, 1].item() == 1
    assert pytest.approx(g1.D_v_neg_1[3, 3].item()) == 0
    assert g1.D_v_neg_1_2[1, 1].item() == 1
    assert pytest.approx(g1.D_v_neg_1_2[3, 3].item()) == 0
    g1.add_extra_selfloop()
    assert g1.D_v[0, 0].item() == 3
    assert g1.D_v[1, 1].item() == 2
    assert g1.D_v_neg_1[1, 1].item() == 0.5
    assert g1.D_v_neg_1[3, 3].item() == 1
    assert pytest.approx(g1.D_v_neg_1_2[1, 1].item()) == 0.7071067690849304
    assert g1.D_v_neg_1_2[3, 3].item() == 1


def test_N(g1):
    assert g1.N_v(0).tolist() == [1, 2]
    assert g1.N_v(1).tolist() == [0]
    assert g1.N_v(2).tolist() == [0]
    assert g1.N_v(3).tolist() == []
    g1.add_edges((3, 0))
    assert g1.N_v(0).tolist() == [1, 2, 3]
    g1.remove_edges((0, 2))
    assert g1.N_v(2).tolist() == []


def test_smoothing():
    num_v = 200
    num_e = 500
    x = torch.rand((num_v, 10))

    g = graph_Gnm(num_v, num_e)
    L = g.L_GCN
    lbd = 0.1

    assert pytest.approx(x + lbd * L @ x) == g.smoothing(x, L, lbd).to_dense()



def test_laplacian_GCN():
    num_v = 20
    num_e = 50

    for _ in range(3):
        g = Graph(num_v)
        A = torch.zeros((num_v, num_v))
        for _ in range(num_e):
            s = random.randrange(num_v)
            d = random.randrange(num_v)
            g.add_edges((s, d))
            A[s, d] = 1
            A[d, s] = 1
        # add the extra selfloop
        for idx in range(num_v):
            A[idx, idx] += 1

        D = A.sum(0)
        D_inv_1_2 = D ** -0.5
        D_inv_1_2[torch.isinf(D_inv_1_2)] = 0
        D_inv_1_2 = torch.diag(D_inv_1_2.view(-1))
        L = D_inv_1_2 @ A @ D_inv_1_2

        assert pytest.approx(g.L_GCN.to_dense()) == L


def test_laplacian_symmetric():
    num_v = 20
    num_e = 50

    for _ in range(3):
        g = Graph(num_v)
        A = torch.zeros((num_v, num_v))
        for _ in range(num_e):
            s = random.randrange(num_v)
            d = random.randrange(num_v)
            if s == d:
                continue
            g.add_edges((s, d))
            A[s, d] = 1
            A[d, s] = 1
        # add the extra selfloop
        # for idx in range(num_v):
        #     A[idx, idx] += 1

        D = A.sum(0)
        D_inv_1_2 = D ** -0.5
        D_inv_1_2[torch.isinf(D_inv_1_2)] = 0
        D_inv_1_2 = torch.diag(D_inv_1_2.view(-1))
        L = torch.eye(num_v) - D_inv_1_2 @ A @ D_inv_1_2

        assert pytest.approx(g.L_sym.to_dense()) == L


def test_laplacian_random_walk():
    num_v = 20
    num_e = 50

    for _ in range(3):
        g = Graph(num_v)
        A = torch.zeros((num_v, num_v))
        for _ in range(num_e):
            s = random.randrange(num_v)
            d = random.randrange(num_v)
            if s == d:
                continue
            g.add_edges((s, d))
            A[s, d] = 1
            A[d, s] = 1
        # add the extra selfloop
        # for idx in range(num_v):
        #     A[idx, idx] += 1

        D = A.sum(0)
        D_inv_1 = D ** -1
        D_inv_1[torch.isinf(D_inv_1)] = 0
        D_inv_1 = torch.diag(D_inv_1.view(-1))
        L = torch.eye(num_v) - D_inv_1 @ A

        assert pytest.approx(g.L_rw.to_dense()) == L


def test_v2v(g1):
    num_v = 20
    num_e = 50
    x = torch.rand(4, 8)  # .cuda()
    y = g1.v2v(x)
    assert y.shape == (4, 8)
    y = g1.v2v(x, "sum")
    assert y.shape == (4, 8)
    w = torch.rand(g1.e_weight.shape[0])
    y = g1.v2v(x, e_weight=w)
    assert y.shape == (4, 8)
    y = g1.v2v(x, "sum", w)
    assert y.shape == (4, 8)


    for _ in range(1):
        g = Graph(num_v)  # .cuda()
        x = torch.rand(num_v, 4)  # .cuda()
        A = torch.zeros((num_v, num_v))  # .cuda()
        for _ in range(num_e):
            s = random.randrange(num_v)
            d = random.randrange(num_v)
            g.add_edges((s, d))
            A[s, d] = 1
            A[d, s] = 1

        D = A.sum(0)
        D_inv = 1 / D
        D_inv[torch.isinf(D_inv)] = 0
        D_inv = torch.diag(D_inv.view(-1))
        sum_v2v = A @ x
        mean_v2v = D_inv @ sum_v2v

        assert pytest.approx(g.v2v(x, "sum").cpu()) == sum_v2v.cpu()
        assert pytest.approx(g.v2v(x, "mean").cpu()) == mean_v2v.cpu()


def test_drop_edges(g1):
    g = graph_Gnm(100, 200)
    gg = g.drop_edges(0.1)
    assert pytest.approx(gg.num_e, rel=0.1) == 180
