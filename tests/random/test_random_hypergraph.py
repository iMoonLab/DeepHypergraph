import pytest

from dhg.random import uniform_hypergraph_Gnm, uniform_hypergraph_Gnp, hypergraph_Gnm


def C(n, m):
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = 1
    for i in range(1, n + 1):
        for j in range(1, min(m + 1, i + 1)):
            dp[i][j] = dp[i - 1][j] + dp[i - 1][j - 1]
    return dp[-1][-1]


def test_uniform_hypergraph_Gnp():
    n_v = 10
    k = 5
    prob = 0.01

    g = uniform_hypergraph_Gnp(k, n_v, prob)
    edges = g.e[0]

    assert all(map(lambda e: len(e) == k, edges))

    max_n_e = C(n_v, k)
    assert pytest.approx(g.num_e / max_n_e, 1) == prob


def test_uniform_hypergraph_Gnm():
    n_v = 100
    n_e = 500
    k = 10

    g = uniform_hypergraph_Gnm(k, n_v, n_e)
    edges = g.e[0]

    assert all(map(lambda e: len(e) == k, edges))
    assert g.num_e == n_e


def test_hypergraph_Gnm():
    n_v = 100
    n_e = 500

    g = hypergraph_Gnm(n_v, n_e)

    assert g.num_e == n_e
