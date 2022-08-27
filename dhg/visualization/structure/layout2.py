from typing import Optional, List

import numpy as np

from .simulator import Simulator
from .utils2 import edge_list_to_incidence_matrix


def force_layout(
    num_v: int,
    e_list: List[tuple],
    push_v_strength: float,
    push_e_strength: float,
    pull_e_strength: float,
    pull_center_strength: float,
):
    v_coor = np.random.rand(num_v, 2) * 10 - 5.0
    assert v_coor.max() <= 5.0 and v_coor.min() >= -5.0
    sim = Simulator(
        nums=num_v,
        forces={
            Simulator.NODE_ATTRACTION: pull_e_strength,
            Simulator.NODE_REPULSION: push_v_strength,
            Simulator.EDGE_REPULSION: push_e_strength,
            Simulator.CENTER_GRAVITY: pull_center_strength,
        },
        n_centers=1,
    )
    v_coor = sim.simulate(v_coor, edge_list_to_incidence_matrix(num_v, e_list))
    # v_coor = (v_coor - np.mean(v_coor, axis=0)) / np.std(v_coor, axis=0)
    v_coor = (v_coor - v_coor.min(0)) / (v_coor.max(0) - v_coor.min(0)) * 0.8 + 0.1
    return v_coor


def bipartite_force_layout(
    num_u: int,
    num_v: int,
    e_list: List[tuple],
    push_u_strength: float,
    push_v_strength: float,
    push_e_strength: float,
    pull_e_strength: float,
    pull_u_center_strength: float,
    pull_v_center_strength: float,
):
    pos = np.random.rand(num_v + num_u, 2)
    sim = Simulator(
        nums=num_v,
        forces={
            Simulator.NODE_ATTRACTION: pull_e_strength,
            Simulator.NODE_REPULSION: [push_u_strength, push_v_strength],
            Simulator.EDGE_REPULSION: push_e_strength,
            Simulator.CENTER_GRAVITY: [pull_u_center_strength, pull_v_center_strength],
        },
        n_centers=2,
    )
    pos = sim.simulate(pos, edge_list_to_incidence_matrix(num_v + num_u, e_list))
    # pos = (pos - np.mean(pos, axis=0)) / np.std(pos, axis=0)
    pos = (pos - pos.min(0)) / (pos.max(0) - pos.min(0)) * 0.8 + 0.1
    return pos[:num_u], pos[num_u:]
