import math
import random
import itertools

from dhg.structure import BiGraph


def bigraph_Gnp(num_u: int, num_v: int, prob: float):
    r"""Return a random bipartite graph with ``num_u`` vertices in set :math:`\mathcal{U}` and ``num_v`` vertices in set :math:`\mathcal{V}` and probability ``prob`` of choosing an edge.

    Args:
        ``num_u`` (``int``): The Number of vertices in set :math:`\mathcal{U}`.
        ``num_v`` (``int``): The Number of vertices in set :math:`\mathcal{V}`.
        ``prob`` (``float``): Probability of choosing an edge.

    Examples:
        >>> import dhg.random as random
        >>> g = random.bigraph_Gnp(2, 3, 0.6)
        >>> g.e
        ([(0, 1), (1, 0), (1, 2)], [1.0, 1.0, 1.0])
    """
    assert num_v > 1, "num_v must be greater than 1"
    assert num_u > 1, "num_u must be greater than 1"
    assert prob >= 0 and prob <= 1, "prob must be between 0 and 1"

    all_e_list = itertools.product(range(num_u), range(num_v))
    e_list = [e for e in all_e_list if random.random() < prob]
    g = BiGraph(num_u, num_v, e_list)
    return g


# def bigraph_Gnp_fast(num_v: int, prob: float):
#     r"""Return a random bipartite graph with ``num_u`` vertices in set :math:`\mathcal{U}` and ``num_v`` vertices in set :math:`\mathcal{V}` and probability ``prob`` of choosing an edge. This function is an implementation of `Efficient generation of large random networks <http://vlado.fmf.uni-lj.si/pub/networks/doc/ms/rndgen.pdf>`_ paper.

#     Args:
#         ``num_v`` (``int``): The Number of vertices.
#         ``prob`` (``float``): Probability of choosing an edge.
#     """
#     assert num_v > 1, "num_v must be greater than 1"
#     assert prob >= 0 and prob <= 1, "prob must be between 0 and 1"

#     e_list = []
#     lp = math.log(1.0 - prob)
#     v, w = 1, -1
#     while v < num_v:
#         lr = math.log(1.0 - random.random())
#         w = w + 1 + int(lr / lp)
#         while w >= v and v < num_v:
#             w = w - v
#             v = v + 1
#         if v < num_v:
#             e_list.append((v, w))
#     g = Graph(num_v, e_list)
#     return g


def bigraph_Gnm(num_u: int, num_v: int, num_e: int):
    r"""Return a random bipartite graph with ``num_u`` vertices in set :math:`\mathcal{U}` and ``num_v`` vertices in set :math:`\mathcal{V}` and ``num_e`` edges. Edges are drawn uniformly from the set of possible edges.

    Args:
        ``num_u`` (``int``): The Number of vertices in set :math:`\mathcal{U}`.
        ``num_v`` (``int``): The Number of vertices in set :math:`\mathcal{V}`.
        ``num_e`` (``int``): The Number of edges.

    Examples:
        >>> import dhg.random as random
        >>> g = random.bigraph_Gnm(3, 3, 5)
        >>> g.e
        ([(1, 2), (2, 1), (1, 1), (2, 0), (1, 0)], [1.0, 1.0, 1.0, 1.0, 1.0])
    """
    assert num_u > 1, "num_u must be greater than 1"
    assert num_v > 1, "num_v must be greater than 1"
    assert (
        num_e <= num_v * num_u
    ), "the specified num_e is larger than the possible number of edges"

    u_list = list(range(num_u))
    v_list = list(range(num_v))
    cur_num_e, e_set = 0, set()
    while cur_num_e < num_e:
        u = random.choice(u_list)
        v = random.choice(v_list)
        if (u, v) in e_set:
            continue
        e_set.add((u, v))
        cur_num_e += 1
    g = BiGraph(num_u, num_v, list(e_set))
    return g
