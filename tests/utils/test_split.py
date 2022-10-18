import dhg
import numpy as np
import pytest

from dhg.utils import split_by_num, split_by_ratio, split_by_num_for_UI_bigraph, split_by_ratio_for_UI_bigraph
from dhg.utils import edge_list_to_adj_list


def test_split_by_num():
    n = 1000
    v_label = np.random.randint(0, 10, n)
    n_train, n_val, n_test = 50, 20, 10
    m_train, m_val, m_test = split_by_num(n, v_label, n_train, n_val, n_test)
    assert m_train.sum() == 500
    assert m_val.sum() == 200
    assert m_test.sum() == 100

    m_train, m_val, m_test = split_by_num(n, v_label, n_train, n_val)
    assert m_train.sum() == 500
    assert m_val.sum() == 200
    assert m_test.sum() == 300


def test_split_by_ratio():
    n = 1000
    v_label = np.random.randint(0, 10, n)
    r_train, r_val, r_test = 0.5, 0.2, 0.1
    m_train, m_val, m_test = split_by_ratio(n, v_label, r_train, r_val, r_test)
    assert pytest.approx(m_train.sum(), 5) == 500
    assert pytest.approx(m_val.sum(), 5) == 200
    assert pytest.approx(m_test.sum(), 5) == 100

    m_train, m_val, m_test = split_by_ratio(n, v_label, r_train, r_val)
    assert pytest.approx(m_train.sum(), 5) == 500
    assert pytest.approx(m_val.sum(), 5) == 200
    assert pytest.approx(m_test.sum(), 5) == 300


def test_split_by_num_for_UI_bigraph():
    e_list = [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 4],
        [2, 2],
        [2, 3],
        [2, 4],
        [3, 3],
        [3, 4],
        [4, 4],
    ]
    g = dhg.BiGraph(5, 5, e_list)
    train_num = 3
    train_adj, test_adj = split_by_num_for_UI_bigraph(g, train_num)
    assert len(train_adj) == 5
    assert len(test_adj) == 2
    assert len(train_adj[0]) == 4
    assert len(train_adj[1]) == 4
    assert len(train_adj[2]) == 4
    assert len(train_adj[3]) == 3
    assert len(train_adj[4]) == 2
    assert len(test_adj[0]) == 3
    assert len(test_adj[1]) == 2


def test_split_by_ratio_for_UI_bigraph():
    e_list = []
    for idx in range(100):
        e_list.append((0, idx))
        e_list.append((1, idx))
    g = dhg.BiGraph(2, 100, e_list)
    train_ratio = 0.6
    train_adj, test_adj = split_by_ratio_for_UI_bigraph(g, train_ratio)
    assert (len(train_adj[0]) - 1) / len(g.nbr_v(0)) == pytest.approx(train_ratio, 0.1)
    assert (len(train_adj[1]) - 1) / len(g.nbr_v(1)) == pytest.approx(train_ratio, 0.1)
    assert (len(test_adj[0]) - 1) / len(g.nbr_v(0)) == pytest.approx(1 - train_ratio, 0.1)
    assert (len(test_adj[1]) - 1) / len(g.nbr_v(1)) == pytest.approx(1 - train_ratio, 0.1)
