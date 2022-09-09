import random
import itertools
from typing import Optional, List

from dhg.utils import C
from dhg.structure import Hypergraph


def uniform_hypergraph_Gnp(k: int, num_v: int, prob: float):
    r"""Return a random ``k``-uniform hypergraph with ``num_v`` vertices and probability ``prob`` of choosing a hyperedge.

    Args:
        ``num_v`` (``int``): The Number of vertices.
        ``k`` (``int``): The Number of vertices in each hyperedge.
        ``prob`` (``float``): Probability of choosing a hyperedge.

    Examples:
        >>> import dhg.random as random
        >>> hg = random.uniform_hypergraph_Gnp(3, 5, 0.5)
        >>> hg.e
        ([(0, 1, 3), (0, 1, 4), (0, 2, 4), (1, 3, 4), (2, 3, 4)], [1.0, 1.0, 1.0, 1.0, 1.0])
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

    Examples:
        >>> import dhg.random as random
        >>> hg = random.uniform_hypergraph_Gnm(3, 5, 4)
        >>> hg.e
        ([(0, 1, 2), (0, 1, 3), (0, 3, 4), (2, 3, 4)], [1.0, 1.0, 1.0, 1.0])
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


def hypergraph_Gnm(num_v: int, num_e: int, method:str="low_order_first", prob_k_list: Optional[List[float]] = None):
    r"""Return a random hypergraph with ``num_v`` vertices and ``num_e`` hyperedges. The ``method`` argument determines the distribution of the hyperedge degree.
    The ``method`` can be one of ``"uniform"``, ``"low_order_first"``, ``"high_order_first"``.

    - If set to ``"uniform"``, the number of hyperedges with the same degree will approximately to the capacity of each hyperedge degree.
      For example, the ``num_v`` is :math:`10`. The capacity of hyperedges with degree  :math:`2` is :math:`C^2_{10} = 45`.
    - If set to ``"low_order_first"``, the generated hyperedges will tend to have low degrees.
    - If set to ``"high_order_first"``, the generated hyperedges will tend to have high degrees.

    Args:
        ``num_v`` (``int``): The Number of vertices.
        ``num_e`` (``int``): The Number of hyperedges.
        ``method`` (``str``): The method to generate hyperedges must be one of ``"uniform"``, ``"low_order_first"``, ``"high_order_first"``. Defaults to ``"uniform"``.
    Examples:
        >>> import dhg.random as random
        >>> hg = random.hypergraph_Gnm(5, 4)
        >>> hg.e
        ([(0, 1, 3, 4), (0, 2, 3, 4), (0, 2, 3), (0, 2, 4)], [1.0, 1.0, 1.0, 1.0])
    """
    # similar to nauty in sagemath, https://doc.sagemath.org/html/en/reference/graphs/sage/graphs/hypergraph_generators.html

    assert num_v > 1, "num_v must be greater than 1"
    assert num_e > 0, "num_e must be greater than 0"
    assert method in ("uniform", "low_order_first", "high_order_first"), "method must be one of 'uniform', 'low_order_first', 'high_order_first'"
    deg_e_list = list(range(2, num_v + 1))
    if method == "uniform":
        prob_k_list = [C(num_v, k) / (2 ** num_v - 1) for k in deg_e_list]
    elif method == "low_order_first":
        prob_k_list = [3 ** (-k) for k in range(len(deg_e_list))]
        sum_of_prob_k_list = sum(prob_k_list)
        prob_k_list = [prob_k / sum_of_prob_k_list for prob_k in prob_k_list]
    elif method == "high_order_first":
        prob_k_list = [3 ** (-k) for k in range(len(deg_e_list))].reverse()
        sum_of_prob_k_list = sum(prob_k_list)
        prob_k_list = [prob_k / sum_of_prob_k_list for prob_k in prob_k_list]
    else:
        raise ValueError(f"Unknown method: {method}")

    edges = set()
    while len(edges) < num_e:
        k = random.choices(deg_e_list, weights=prob_k_list)[0]
        e = random.sample(range(num_v), k)
        e = tuple(sorted(e))
        if e not in edges:
            edges.add(e)

    return Hypergraph(num_v, list(edges))
