import pytest

from dhg.random import bigraph_Gnp, bigraph_Gnm

def test_graph_gnp():
    n_u = 50
    n_v = 100

    g = bigraph_Gnp(n_u, n_v, 0.5)
    max_n_e = n_v * n_u
    assert pytest.approx(g.num_e / max_n_e, 0.05) == 0.5


def test_graph_gnm():
    n_u, n_v, n_e = 50, 100, 500

    g = bigraph_Gnm(n_u, n_v, n_e)
    assert g.num_v == n_v
    assert g.num_e == n_e