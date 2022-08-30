from typing import Optional, List

import numpy as np
from scipy.spatial import ConvexHull

from .simulator import Simulator
from .utils import edge_list_to_incidence_matrix, init_pos
from .geometry import radian_from_atan, vlen, common_tangent_radian, polar_position


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

def hull_layout(n_v, e_list, pos, init_radius=0.015, radius_increment=0.005):

    # paths = []
    line_paths = []
    arc_paths = []

    polygons_vertices_index = []
    vertices_radius = np.zeros(n_v) + init_radius

    for edge in e_list:

        line_path_for_e = []
        arc_path_for_e = []

        pos_in_edge = pos[edge]
        if len(edge) == 2:
            vertices_index = np.array((0, 1), dtype=np.int64)
        else:
            hull = ConvexHull(pos_in_edge)
            vertices_index = hull.vertices

        n_vertices = vertices_index.shape[0]

        vertices_index = np.append(vertices_index, vertices_index[0]) # close the loop

        thetas = []

        for i in range(n_vertices):
            # line
            i1 = edge[vertices_index[i]]
            i2 = edge[vertices_index[i + 1]]

            r1 = vertices_radius[i1]
            r2 = vertices_radius[i2]

            p1 = pos[i1]
            p2 = pos[i2]

            dp = p2 - p1
            dp_len = vlen(dp)

            beta = radian_from_atan(dp[0], dp[1])
            alpha = common_tangent_radian(r1, r2, dp_len)

            theta = beta - alpha
            start_point = polar_position(r1, theta, p1)
            end_point = polar_position(r2, theta, p2)

            # paths.append((start_point, end_point))
            line_path_for_e.append((start_point, end_point))
            thetas.append(theta)

        thetas.append(thetas[0])

        for i in range(n_vertices + 1):
            # arcs
            theta_1 = thetas[i - 1]
            theta_2 = thetas[i]

            arc_center = pos[edge[vertices_index[i]]]
            radius = vertices_radius[edge[vertices_index[i]]]

            # paths.append((arc_center, theta_1, theta_2, radius))
            arc_path_for_e.append((arc_center, theta_1, theta_2, radius))

        vertices_radius[edge] += radius_increment

        polygons_vertices_index.append(vertices_index.copy())

        line_paths.append(line_path_for_e)
        arc_paths.append(arc_path_for_e)

    return line_paths, arc_paths, polygons_vertices_index
