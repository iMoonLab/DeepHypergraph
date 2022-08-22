from typing import Optional, Union, List, Tuple, Any
from itertools import chain

import numpy as np
import matplotlib
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.collections import PathCollection, PatchCollection


def default_style(
    num_v: int,
    num_e: int,
    v_color: Optional[Union[str, list]] = None,
    e_color: Optional[Union[str, list]] = None,
    e_fill_color: Optional[Union[str, list]] = None,
    font_family: Optional[str] = None,
):
    if v_color is None:
        v_color = (1.0, 0.2, 0.2)
    if not isinstance(v_color, list):
        v_color = [v_color] * num_v

    if e_color is None:
        e_color = (0.2, 0.2, 0.6)
    if not isinstance(e_color, list):
        e_color = [e_color] * num_e

    if e_fill_color is None:
        e_fill_color = (0.2, 0.2, 0.6, 0.1)
    if not isinstance(e_fill_color, list):
        e_fill_color = [e_fill_color] * num_e

    if font_family is None:
        font_family = "sans-serif"

    return v_color, e_color, e_fill_color, font_family


def default_size(
    num_v: int,
    e_list: List[tuple],
    v_size: Optional[Union[float, list]] = None,
    v_line_width: Optional[Union[float, list]] = None,
    e_line_width: Optional[Union[float, list]] = None,
    font_size: Optional[int] = None,
):
    # =============================================================
    # compute default v_size
    _v_size = 1 / np.sqrt(num_v) * 0.1
    # =============================================================
    v_size = fill_sizes(v_size, _v_size, num_v)

    # =============================================================
    # compute default v_size
    _v_line_width = 1
    # =============================================================
    v_line_width = fill_sizes(v_line_width, _v_line_width, num_v)

    # =============================================================
    # compute default e_line_width
    _e_line_width = 1
    # =============================================================
    e_line_width = fill_sizes(e_line_width, _e_line_width, len(e_list))

    font_size = 12 if font_size is None else font_size

    return v_size, v_line_width, e_line_width, font_size


def default_strength(
    num_v: int,
    e_list: List[tuple],
    push_v_strength: Optional[float] = None,
    push_e_strength: Optional[float] = None,
    pull_e_strength: Optional[float] = None,
    pull_center_strength: Optional[float] = None,
):
    # =============================================================
    # compute default push_v_strength
    _push_v_strength = 0.00010
    # =============================================================
    push_v_strength = fill_strength(push_v_strength, _push_v_strength)

    # =============================================================
    # compute default push_e_strength
    _push_e_strength = 0.0
    # =============================================================
    push_e_strength = fill_strength(push_e_strength, _push_e_strength)

    # =============================================================
    # compute default pull_e_strength
    _pull_e_strength = 0.0
    # =============================================================
    pull_e_strength = fill_strength(pull_e_strength, _pull_e_strength)

    # =============================================================
    # compute default pull_center_strength
    _pull_center_strength = 0.5
    # =============================================================
    pull_center_strength = fill_strength(pull_center_strength, _pull_center_strength)

    return push_v_strength, push_e_strength, pull_e_strength, pull_center_strength


def default_bipartite_size(
    num_u: int,
    num_v: int,
    e_list: List[tuple],
    u_size: Optional[Union[float, list]] = None,
    u_line_width: Optional[Union[float, list]] = None,
    v_size: Optional[Union[float, list]] = None,
    v_line_width: Optional[Union[float, list]] = None,
    e_line_width: Optional[Union[float, list]] = None,
    u_font_size: Optional[int] = None,
    v_font_size: Optional[int] = None,
):
    u_size = 1 if u_size is None else u_size
    u_size = [u_size] * num_u if not isinstance(u_size, list) else u_size

    u_line_width = 1 if u_line_width is None else u_line_width
    u_line_width = [u_line_width] * num_u if not isinstance(u_line_width, list) else u_line_width

    v_size = 1 if v_size is None else v_size
    v_size = [v_size] * num_v if not isinstance(v_size, list) else v_size

    v_line_width = 1 if v_line_width is None else v_line_width
    v_line_width = [v_line_width] * num_v if not isinstance(v_line_width, list) else v_line_width

    e_line_width = 1 if e_line_width is None else e_line_width
    e_line_width = [e_line_width] * len(e_list) if not isinstance(e_line_width, list) else e_line_width

    u_font_size = 12 if u_font_size is None else u_font_size
    v_font_size = 12 if v_font_size is None else v_font_size

    return u_size, u_line_width, v_size, v_line_width, e_line_width, u_font_size, v_font_size


def default_bipartite_strength(
    num_u: int,
    num_v: int,
    e_list: List[tuple],
    push_u_strength: Optional[float] = None,
    push_v_strength: Optional[float] = None,
    push_e_strength: Optional[float] = None,
    pull_e_strength: Optional[float] = None,
    pull_u_center_strength: Optional[float] = None,
    pull_v_center_strength: Optional[float] = None,
):

    push_u_strength = 1 if push_u_strength is None else push_u_strength
    push_v_strength = 1 if push_v_strength is None else push_v_strength
    push_e_strength = 1 if push_e_strength is None else push_e_strength
    pull_e_strength = 1 if pull_e_strength is None else pull_e_strength
    pull_u_center_strength = 1 if pull_u_center_strength is None else pull_u_center_strength
    pull_v_center_strength = 1 if pull_v_center_strength is None else pull_v_center_strength

    return (
        push_u_strength,
        push_v_strength,
        push_e_strength,
        pull_e_strength,
        pull_u_center_strength,
        pull_v_center_strength,
    )


def fill_sizes(custom_scales: Optional[Union[float, list]], default_value: Any, length: int):
    if custom_scales is None:
        return [default_value] * length
    elif isinstance(custom_scales, list):
        assert len(custom_scales) == length, "The specified value list has the wrong length."
        return [default_value * scale for scale in custom_scales]
    elif isinstance(custom_scales, float):
        return [default_value * custom_scales] * length
    elif isinstance(custom_scales, int):
        return [default_value * float(custom_scales)] * length
    else:
        raise ValueError("The specified value is not a valid type.")


def fill_strength(custom_scale: Optional[float], default_value: float):
    if custom_scale is None:
        return default_value
    return custom_scale * default_value


def draw_line_edge(
    ax: matplotlib.axes.Axes,
    v_coor: np.array,
    v_size: list,
    e_list: List[Tuple[int, int]],
    show_arrow: bool,
    e_color: list,
    e_line_width: list,
):
    arrow_head_width = 3 * e_line_width if show_arrow else 0
    arrow_head_lenght = 1.5 * arrow_head_width

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
            head_width=arrow_head_width,
            head_length=arrow_head_lenght,
            color=e_color[eidx],
            linewidth=e_line_width[eidx],
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
    pass


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
        circle.label = label
        patches.append(circle)
    p = PatchCollection(patches, facecolors=v_color, edgecolors="black")
    ax.add_collection(p)
