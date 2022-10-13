from itertools import chain
from typing import Optional, List, Tuple

import matplotlib
import numpy as np
from matplotlib.path import Path
from scipy.spatial import ConvexHull
from matplotlib.patches import Circle, PathPatch
from matplotlib.collections import PathCollection, PatchCollection

from .geometry import radian_from_atan, vlen, common_tangent_radian, polar_position, rad_2_deg


def safe_div(a: np.ndarray, b: np.ndarray, jitter_scale: float = 0.000001):
    mask = b == 0
    b[mask] = 1
    inv_b = 1.0 / b
    res = a * inv_b
    if mask.sum() > 0:
        res[mask.repeat(2, 2)] = np.random.randn(mask.sum() * 2) * jitter_scale
    return res


def init_pos(num_v: int, center: Tuple[float, float] = (0, 0), scale: float = 1.0):
    return (np.random.rand(num_v, 2) * 2 - 1) * scale + center


def draw_line_edge(
    ax: matplotlib.axes.Axes,
    v_coor: np.array,
    v_size: list,
    e_list: List[Tuple[int, int]],
    show_arrow: bool,
    e_color: list,
    e_line_width: list,
):
    arrow_head_width = [0.015 * w for w in e_line_width] if show_arrow else [0] * len(e_list)

    for eidx, e in enumerate(e_list):
        start_pos = v_coor[e[0]]
        end_pos = v_coor[e[1]]

        dir = end_pos - start_pos
        dir = dir / np.linalg.norm(dir)

        start_pos = start_pos + dir * v_size[e[0]]
        end_pos = end_pos - dir * v_size[e[1]]

        x, y = start_pos[0], start_pos[1]
        dx, dy = end_pos[0] - x, end_pos[1] - y

        ax.arrow(
            x,
            y,
            dx,
            dy,
            head_width=arrow_head_width[eidx],
            color=e_color[eidx],
            linewidth=e_line_width[eidx],
            length_includes_head=True,
        )


def draw_circle_edge(
    ax: matplotlib.axes.Axes,
    v_coor: List[Tuple[float, float]],
    v_size: list,
    e_list: List[Tuple[int, int]],
    e_color: list,
    e_fill_color: list,
    e_line_width: list,
):
    n_v = len(v_coor)
    line_paths, arc_paths, vertices = hull_layout(n_v, e_list, v_coor, v_size)

    for eidx, lines in enumerate(line_paths):
        pathdata = []
        for line in lines:
            if len(line) == 0:
                continue
            start_pos, end_pos = line
            pathdata.append((Path.MOVETO, start_pos.tolist()))
            pathdata.append((Path.LINETO, end_pos.tolist()))

        if len(list(zip(*pathdata))) == 0:
            continue
        codes, verts = zip(*pathdata)
        path = Path(verts, codes)
        ax.add_patch(
            PathPatch(path, linewidth=e_line_width[eidx], facecolor=e_fill_color[eidx], edgecolor=e_color[eidx])
        )

    for eidx, arcs in enumerate(arc_paths):
        for arc in arcs:
            center, theta1, theta2, radius = arc
            x, y = center[0], center[1]

            ax.add_patch(
                matplotlib.patches.Arc(
                    (x, y),
                    2 * radius,
                    2 * radius,
                    theta1=theta1,
                    theta2=theta2,
                    # color=e_color[eidx],
                    linewidth=e_line_width[eidx],
                    edgecolor=e_color[eidx],
                    facecolor=e_fill_color[eidx],
                )
            )


def edge_list_to_incidence_matrix(num_v: int, e_list: List[tuple]) -> np.ndarray:
    v_idx = list(chain(*e_list))
    e_idx = [[idx] * len(e) for idx, e in enumerate(e_list)]
    e_idx = list(chain(*e_idx))
    H = np.zeros((num_v, len(e_list)))
    H[v_idx, e_idx] = 1
    return H


def draw_vertex(
    ax: matplotlib.axes.Axes,
    v_coor: List[Tuple[float, float]],
    v_label: Optional[List[str]],
    font_size: int,
    font_family: str,
    v_size: list,
    v_color: list,
    v_line_width: list,
):
    patches = []
    n = v_coor.shape[0]
    if v_label is None:
        v_label = [""] * n
    for coor, label, size, width in zip(v_coor.tolist(), v_label, v_size, v_line_width):
        circle = Circle(coor, size)
        circle.lineWidth = width
        # circle.label = label
        if label != '':
            x, y = coor[0], coor[1]
            offset = 0, -1.3 * size
            x += offset[0]
            y += offset[1]
            ax.text(x, y, label, fontsize=font_size, fontfamily=font_family, ha='center', va='top')
        patches.append(circle)
    p = PatchCollection(patches, facecolors=v_color, edgecolors="black")
    ax.add_collection(p)


def hull_layout(n_v, e_list, pos, v_size, radius_increment=0.3):

    line_paths = [None] * len(e_list)
    arc_paths = [None] * len(e_list)

    polygons_vertices_index = []
    vertices_radius = np.array(v_size)
    vertices_increased_radius = vertices_radius * radius_increment
    vertices_radius += vertices_increased_radius

    e_degree = [len(e) for e in e_list]
    e_idxs = np.argsort(np.array(e_degree))

    # for edge in e_list:
    for e_idx in e_idxs:

        edge = list(e_list[e_idx])

        line_path_for_e = []
        arc_path_for_e = []

        if len(edge) == 1:
            arc_path_for_e.append([pos[edge[0]], 0, 360, vertices_radius[edge[0]]])

            vertices_radius[edge] += vertices_increased_radius[edge]

            line_paths[e_idx] = line_path_for_e
            arc_paths[e_idx] = arc_path_for_e
            continue

        pos_in_edge = pos[edge]
        if len(edge) == 2:
            vertices_index = np.array((0, 1), dtype=np.int64)
        else:
            hull = ConvexHull(pos_in_edge)
            vertices_index = hull.vertices

        n_vertices = vertices_index.shape[0]

        vertices_index = np.append(vertices_index, vertices_index[0])  # close the loop

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

            line_path_for_e.append((start_point, end_point))
            thetas.append(theta)

        for i in range(n_vertices):
            # arcs
            theta_1 = thetas[i - 1]
            theta_2 = thetas[i]

            arc_center = pos[edge[vertices_index[i]]]
            radius = vertices_radius[edge[vertices_index[i]]]

            theta_1, theta_2 = rad_2_deg(theta_1), rad_2_deg(theta_2)
            arc_path_for_e.append((arc_center, theta_1, theta_2, radius))

        vertices_radius[edge] += vertices_increased_radius[edge]

        polygons_vertices_index.append(vertices_index.copy())

        # line_paths.append(line_path_for_e)
        # arc_paths.append(arc_path_for_e)
        line_paths[e_idx] = line_path_for_e
        arc_paths[e_idx] = arc_path_for_e

    return line_paths, arc_paths, polygons_vertices_index
