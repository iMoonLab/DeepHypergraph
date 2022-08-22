import random
import itertools
from typing import Optional, List

from dhg.utils import C
from dhg.structure import Hypergraph


def uniform_hypergraph_Gnp(k: int, num_v: int, prob: float):
    r"""Return a random ``k``-uniform hypergraph with ``num_v`` vertices and probability ``prob`` of choicing a hyperedge.

    Args:
        ``num_v`` (``int``): The Number of vertices.
        ``k`` (``int``): The Number of vertices in each hyperedge.
        ``prob`` (``float``): Probability of choosing a hyperedge.
    """
    # similar to BinomialRandomUniform in sagemath, https://doc.sagemath.org/html/en/reference/graphs/sage/graphs/hypergraph_generators.html

    assert num_v > 1, "num_v must be greater than 1"
    assert k > 1, "k must be greater than 1"  # TODO ?
    assert 0 <= prob <= 1, "prob must be between 0 and 1"

    edges = itertools.combinations(range(num_v), k)
    edges = [e for e in edges if random.random() < prob]

    return Hypergraph(num_v, edges)


def uniform_hypergraph_Gnm(k: int, num_v: int, num_e: int):
    r"""Return a random ``k``-uniform hypergraph with ``num_v`` vertices and ``num_e`` hyperedges.

    Args:
        ``k`` (``int``): The Number of vertices in each hyperedge.
        ``num_v`` (``int``): The Number of vertices.
        ``num_e`` (``int``): The Number of hyperedges.
    """
    # similar to UniformRandomUniform in sagemath, https://doc.sagemath.org/html/en/reference/graphs/sage/graphs/hypergraph_generators.html

    assert k > 1, "k must be greater than 1"  # TODO ?
    assert num_v > 1, "num_v must be greater than 1"
    assert num_e > 0, "num_e must be greater than 0"

    edges = set()
    while len(edges) < num_e:
        e = random.sample(range(num_v), k)
        e = tuple(sorted(e))
        if e not in edges:
            edges.add(e)

    return Hypergraph(num_v, list(edges))


def hypergraph_Gnm(num_v: int, num_e: int, prob_k_list: Optional[List[float]] = None):
    r"""Return a random hypergraph with ``num_v`` vertices and ``num_e`` hyperedges.

    Args:
        ``num_v`` (``int``): The Number of vertices.
        ``num_e`` (``int``): The Number of hyperedges.
    """
    # similar to nauty in sagemath, https://doc.sagemath.org/html/en/reference/graphs/sage/graphs/hypergraph_generators.html

    assert num_v > 1, "num_v must be greater than 1"
    assert num_e > 0, "num_e must be greater than 0"

    if prob_k_list is None:
        # prob_k_list = [1 / (num_v - 1)] * (num_v - 1)
        prob_k_list = [C(num_v, k) / (2 ** num_v - 1) for k in range(2, num_v + 1)]

    edges = set()
    while len(edges) < num_e:
        k = random.choices(range(2, num_v + 1), weights=prob_k_list)[0]
        e = random.sample(range(num_v), k)
        e = tuple(sorted(e))
        if e not in edges:
            edges.add(e)

    return Hypergraph(num_v, list(edges))
