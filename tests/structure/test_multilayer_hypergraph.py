import torch
import pytest
import numpy as np
from dhg import Hypergraph, MultilayerHypergraph
import matplotlib.pyplot as plt

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


# test adjacency matrix
def test_adjacency_matrix(g1):
    adj = g1.A
    assert adj.shape == (6, 6)
    assert np.allclose(adj[0, 1], 1)
    assert np.allclose(adj[0, 2], 1)
    assert np.allclose(adj[0, 3], 0)
    assert np.allclose(adj[0, 4], 0)
    assert np.allclose(adj[0, 5], 1)
# test construction
def test_construct_multilayer_hypergraph():
    g1 = g1()
    g2 = g2()
    
    mhg = MultilayerHypergraph(11, 2, [g1, g2])
    assert mhg.num_layers == 2
    assert mhg.layers_list[0] == g1
    assert mhg.layers_list[1] == g2
    assert mhg.num_v == 11
    assert mhg.prob_inner_layer_connect == 0
    assert mhg.device == torch.device('cpu')

def test_draw(mhg):
    mhg.draw()
    plt.show()

