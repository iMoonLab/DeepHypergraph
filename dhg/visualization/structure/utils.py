from typing import Optional, Union, List, Tuple, Any
from itertools import chain

import numpy as np
import matplotlib
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.collections import PathCollection, PatchCollection


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


def default_style(
    num_v: int,
    num_e: int,
    v_color: Optional[Union[str, list]] = None,
    e_color: Optional[Union[str, list]] = None,
    e_fill_color: Optional[Union[str, list]] = None,
    font_family: Optional[str] = None,
):
    _v_color = "r"
    v_color = fill_color(v_color, _v_color, num_v)

    _e_color = (0.7, 0.7, 0.7)
    e_color = fill_color(e_color, _e_color, num_e)

    _e_fill_color = (0.2, 0.2, 0.6, 0.1)
    e_fill_color = fill_color(e_fill_color, _e_fill_color, num_e)

    if font_family is None:
        font_family = "sans-serif"

    return v_color, e_color, e_fill_color, font_family


def default_bipartite_style(
    num_u: int,
    num_v: int,
    num_e: int,
    u_color: Optional[Union[str, list]] = None,
    v_color: Optional[Union[str, list]] = None,
    e_color: Optional[Union[str, list]] = None,
    e_fill_color: Optional[Union[str, list]] = None,
    font_family: Optional[str] = None,
):
    _u_color = "m"
    u_color = fill_color(u_color, _u_color, num_u)

    _v_color = "r"
    v_color = fill_color(v_color, _v_color, num_v)

    _e_color = (0.7, 0.7, 0.7)
    e_color = fill_color(e_color, _e_color, num_e)

    _e_fill_color = (0.2, 0.2, 0.6, 0.1)
    e_fill_color = fill_color(e_fill_color, _e_fill_color, num_e)

    if font_family is None:
        font_family = "sans-serif"

    return u_color, v_color, e_color, e_fill_color, font_family


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
    _v_size = 1 / np.sqrt(num_v + 12) * 0.08
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
    # =============================================================
    # compute default v_size
    _u_size = 1 / np.sqrt(num_u + 12) * 0.08
    # =============================================================
    u_size = fill_sizes(u_size, _u_size, num_u)

    # =============================================================
    # compute default v_size
    _u_line_width = 1
    # =============================================================
    u_line_width = fill_sizes(u_line_width, _u_line_width, num_u)

    # =============================================================
    # compute default v_size
    _v_size = 1 / np.sqrt(num_v + 12) * 0.08
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

    u_font_size = 12 if u_font_size is None else u_font_size
    v_font_size = 12 if v_font_size is None else v_font_size

    return u_size, u_line_width, v_size, v_line_width, e_line_width, u_font_size, v_font_size


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
    _push_v_strength = 0.006
    # =============================================================
    push_v_strength = fill_strength(push_v_strength, _push_v_strength)

    # =============================================================
    # compute default push_e_strength
    _push_e_strength = 0.0
    # =============================================================
    push_e_strength = fill_strength(push_e_strength, _push_e_strength)

    # =============================================================
    # compute default pull_e_strength
    _pull_e_strength = 0.045
    # =============================================================
    pull_e_strength = fill_strength(pull_e_strength, _pull_e_strength)

    # =============================================================
    # compute default pull_center_strength
    _pull_center_strength = 0.01
    # =============================================================
    pull_center_strength = fill_strength(pull_center_strength, _pull_center_strength)

    return push_v_strength, push_e_strength, pull_e_strength, pull_center_strength


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
    # =============================================================
    # compute default push_u_strength
    _push_u_strength = 0.005
    # =============================================================
    push_u_strength = fill_strength(push_u_strength, _push_u_strength)

    # =============================================================
    # compute default push_v_strength
    _push_v_strength = 0.005
    # =============================================================
    push_v_strength = fill_strength(push_v_strength, _push_v_strength)

    # =============================================================
    # compute default push_e_strength
    _push_e_strength = 0.0
    # =============================================================
    push_e_strength = fill_strength(push_e_strength, _push_e_strength)

    # =============================================================
    # compute default pull_e_strength
    _pull_e_strength = 0.03
    # =============================================================
    pull_e_strength = fill_strength(pull_e_strength, _pull_e_strength)

    # =============================================================
    # compute default pull_center_strength
    _pull_u_center_strength = 0.04
    # =============================================================
    pull_u_center_strength = fill_strength(pull_u_center_strength, _pull_u_center_strength)

    # =============================================================
    # compute default pull_center_strength
    _pull_v_center_strength = 0.04
    # =============================================================
    pull_v_center_strength = fill_strength(pull_v_center_strength, _pull_v_center_strength)

    return (
        push_u_strength,
        push_v_strength,
        push_e_strength,
        pull_e_strength,
        pull_u_center_strength,
        pull_v_center_strength,
    )


def fill_color(custom_color: Optional[Union[str, list]], default_color: Any, length: int):
    if custom_color is None:
        return [default_color] * length
    elif isinstance(custom_color, list):
        if isinstance(custom_color[0], str) or isinstance(custom_color[0], tuple) or isinstance(custom_color[0], list):
            return custom_color
        else:
            return [custom_color] * length
    elif isinstance(custom_color, str):
        return [custom_color] * length
    else:
        raise ValueError("The specified value is not a valid type.")


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
    arrow_head_width = [0.015 * w for w in e_line_width] if show_arrow else [0] * len(e_list)
    arrow_head_lenght = [0.8 * w for w in arrow_head_width]

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
            head_length=arrow_head_lenght[eidx],
            color=e_color[eidx],
            linewidth=e_line_width[eidx],
            length_includes_head=True
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
