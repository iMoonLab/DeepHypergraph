import pytest

from dhg.random import digraph_Gnp, digraph_Gnm, digraph_Gnp_fast

def test_digraph_gnp():
    n_v = 100

    g = digraph_Gnp(n_v, 0.5)
    max_n_e = n_v * (n_v - 1)
    assert pytest.approx(g.num_e / max_n_e, 0.05) == 0.5


def test_digraph_gnp_fast():
    n_v = 100

    g = digraph_Gnp_fast(n_v, 0.5)
    max_n_e = n_v * (n_v - 1)
    assert pytest.approx(g.num_e / max_n_e, 0.05) == 0.5


def test_digraph_gnm():
    n_v, n_e = 100, 500

    g = digraph_Gnm(n_v, n_e)
    assert g.num_v == n_v
    assert g.num_e == n_e
