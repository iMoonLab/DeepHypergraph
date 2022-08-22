import pytest

from dhg.random import graph_Gnp, graph_Gnm, graph_Gnp_fast

def test_graph_gnp():
    n_v = 100

    g = graph_Gnp(n_v, 0.5)
    max_n_e = n_v * (n_v - 1) // 2
    assert pytest.approx(g.num_e / max_n_e, 0.05) == 0.5


def test_graph_gnp_fast():
    n_v = 100

    g = graph_Gnp_fast(n_v, 0.5)
    max_n_e = n_v * (n_v - 1) // 2
    assert pytest.approx(g.num_e / max_n_e, 0.05) == 0.5

def test_graph_gnm():
    n_v, n_e = 100, 500

    g = graph_Gnm(n_v, n_e)
    assert g.num_v == n_v
    assert g.num_e == n_e
