import torch
import pytest

from dhg import Hypergraph, MultilayerHypergraph


@pytest.fixture
def g1():
    return Hypergraph(6, [(0, 1, 2, 5), (0, 1), (2, 3, 4), (3, 2, 4)])


@pytest.fixture
def g2():
    return Hypergraph(5, [(1, 2, 3), (0, 1, 3), (0, 1), (2, 4, 3), (2, 3)])


def test_construct_multilayer_hypergraph(g1, g2):
    mhg = MultilayerHypergraph(11, 2, [g1, g2])

    assert mhg.num_layers == 2
    assert mhg.layers_list == [g1, g2]
    assert mhg.num_v == 11
    assert mhg.prob_inner_layer_connect == 0
    assert mhg.device == torch.device("cpu")

    combined = mhg.construct_multi_layer_hypergraph()
    assert type(combined) is Hypergraph
    assert combined.num_v == 11
    assert combined.num_e == g1.num_e + g2.num_e


def test_offsets_are_local_and_repeatable(g1, g2):
    mhg = MultilayerHypergraph(11, 2, [g1, g2])

    assert mhg.node_layer_list() == [list(g1.v), [v + 6 for v in g2.v]]
    first = mhg._inter_layer_connect()
    second = mhg._inter_layer_connect()
    assert first == second
    assert mhg.construct_multi_layer_hypergraph().num_e == g1.num_e + g2.num_e


def test_select_nodes_accepts_range(monkeypatch):
    monkeypatch.setattr(
        "dhg.structure.hypergraphs.multilayer_hypergraph.random.random",
        lambda: 0,
    )
    assert MultilayerHypergraph.select_nodes(range(3), 1.0) == [0, 1, 2]


def test_mapping_save_writes_once(g1, g2, tmp_path):
    source_a = tmp_path / "a.txt"
    source_b = tmp_path / "b.txt"
    source_a.write_text("geneA\t1\ngeneB\t2\n", encoding="utf-8")
    source_b.write_text("geneA\t7\ngeneB\t8\n", encoding="utf-8")
    output = tmp_path / "merged.txt"
    mhg = MultilayerHypergraph(11, 2, [g1, g2])

    assert mhg.load_interlayer_mapping(
        [str(source_a), str(source_b)], True, output
    ) == [(1, 7), (2, 8)]
    assert output.read_text(encoding="utf-8").splitlines() == [
        "geneA [1, 7]",
        "geneB [2, 8]",
    ]