from typing import Optional, List

import numpy as np

from .simulator import Simulator
from .utils import edge_list_to_incidence_matrix, init_pos


def force_layout(
    num_v: int,
    e_list: List[tuple],
    push_v_strength: float,
    push_e_strength: float,
    pull_e_strength: float,
    pull_center_strength: float,
):
    v_coor = init_pos(num_v, scale=5)
    assert v_coor.max() <= 5.0 and v_coor.min() >= -5.0
    centers = [np.array([0, 0])]
    sim = Simulator(
        nums=num_v,
        forces={
            Simulator.NODE_ATTRACTION: pull_e_strength,
            Simulator.NODE_REPULSION: push_v_strength,
            Simulator.EDGE_REPULSION: push_e_strength,
            Simulator.CENTER_GRAVITY: pull_center_strength,
        },
        centers=centers,
    )
    v_coor = sim.simulate(v_coor, edge_list_to_incidence_matrix(num_v, e_list))
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
    pos_u = init_pos(num_u, center=(5, 0), scale=4.5)
    pos_v = init_pos(num_v, center=(-5, 0), scale=4.5)
    centers = [np.array([5, 0]), np.array([-5, 0])]
    pos = np.vstack((pos_u, pos_v))
    sim = Simulator(
        nums=num_u,
        forces={
            Simulator.NODE_ATTRACTION: pull_e_strength,
            Simulator.NODE_REPULSION: [push_u_strength, push_v_strength],
            Simulator.EDGE_REPULSION: push_e_strength,
            Simulator.CENTER_GRAVITY: [pull_u_center_strength, pull_v_center_strength],
        },
        centers=centers,
    )
    pos = sim.simulate(pos, edge_list_to_incidence_matrix(num_v + num_u, e_list))
    # pos = (pos - np.mean(pos, axis=0)) / np.std(pos, axis=0)
    pos = (pos - pos.min(0)) / (pos.max(0) - pos.min(0)) * 0.8 + 0.1
    return pos[:num_u], pos[num_u:]
