import math
import random
import itertools

from dhg.structure import Graph


def graph_Gnp(num_v: int, prob: float):
    r"""Return a random graph with ``num_v`` vertices and probability ``prob`` of choosing an edge. 

    Args:
        ``num_v`` (``int``): The Number of vertices.
        ``prob`` (``float``): Probability of choosing an edge.

    Examples:
        >>> import dhg.random as random
        >>> g = random.graph_Gnp(4, 0.5)
        >>> g.e
        ([(0, 1), (0, 2), (0, 3)], [1.0, 1.0, 1.0])
    """
    assert num_v > 1, "num_v must be greater than 1"
    assert prob >= 0 and prob <= 1, "prob must be between 0 and 1"

    all_e_list = itertools.permutations(range(num_v), 2)
    e_list = [e for e in all_e_list if random.random() < prob and e[0] < e[1]]
    g = Graph(num_v, e_list)
    return g


def graph_Gnp_fast(num_v: int, prob: float):
    r"""Return a random graph with ``num_v`` vertices and probability ``prob`` of choosing an edge. This function is an implementation of `Efficient generation of large random networks <http://vlado.fmf.uni-lj.si/pub/networks/doc/ms/rndgen.pdf>`_ paper.

    Args:
        ``num_v`` (``int``): The Number of vertices.
        ``prob`` (``float``): Probability of choosing an edge.


    Examples:
        >>> import dhg.random as random
        >>> g = random.graph_Gnp_fast(4, 0.8)
        >>> g.e
        ([(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    """
    assert num_v > 1, "num_v must be greater than 1"
    assert prob >= 0 and prob <= 1, "prob must be between 0 and 1"

    e_list = []
    lp = math.log(1.0 - prob)
    v, w = 1, -1
    while v < num_v:
        lr = math.log(1.0 - random.random())
        w = w + 1 + int(lr / lp)
        while w >= v and v < num_v:
            w = w - v
            v = v + 1
        if v < num_v:
            e_list.append((v, w))
    g = Graph(num_v, e_list)
    return g


def graph_Gnm(num_v: int, num_e: int):
    r"""Return a random graph with ``num_v`` verteices and ``num_e`` edges. Edges are drawn uniformly from the set of possible edges.

    Args:
        ``num_v`` (``int``): The Number of vertices.
        ``num_e`` (``int``): The Number of edges.

    Examples:
        >>> import dhg.random as random
        >>> g = random.graph_Gnm(4, 5)
        >>> g.e
        ([(1, 2), (0, 3), (2, 3), (0, 2), (1, 3)], [1.0, 1.0, 1.0, 1.0, 1.0])
    """
    assert num_v > 1, "num_v must be greater than 1"
    assert (
        num_e < num_v * (num_v - 1) // 2
    ), "the specified num_e is larger than the possible number of edges"

    v_list = list(range(num_v))
    cur_num_e, e_set = 0, set()
    while cur_num_e < num_e:
        v = random.choice(v_list)
        w = random.choice(v_list)
        if v > w:
            v, w = w, v
        if v == w or (v, w) in e_set:
            continue
        e_set.add((v, w))
        cur_num_e += 1
    g = Graph(num_v, list(e_set))
    return g
