from typing import Optional, Union, List, Any

import numpy as np


def default_style(
    num_v: int,
    num_e: int,
    v_color: Optional[Union[str, list]] = None,
    e_color: Optional[Union[str, list]] = None,
    e_fill_color: Optional[Union[str, list]] = None,
    font_family: Optional[str] = None,
):
    _v_color = "r"
    _e_color = "gray"
    _e_fill_color = "whitesmoke"

    v_color = fill_color(v_color, _v_color, num_v)
    e_color = fill_color(e_color, _e_color, num_e)
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
    _v_color = "r"
    _e_color = "gray"
    _e_fill_color = "whitesmoke"

    u_color = fill_color(u_color, _u_color, num_u)
    v_color = fill_color(v_color, _v_color, num_v)
    e_color = fill_color(e_color, _e_color, num_e)
    e_fill_color = fill_color(e_fill_color, _e_fill_color, num_e)

    if font_family is None:
        font_family = "sans-serif"

    return u_color, v_color, e_color, e_fill_color, font_family


def default_hypergraph_style(
    num_v: int,
    num_e: int,
    v_color: Optional[Union[str, list]] = None,
    e_color: Optional[Union[str, list]] = None,
    e_fill_color: Optional[Union[str, list]] = None,
    font_family: Optional[str] = None,
):
    _v_color = "r"
    _e_color = "gray"
    _e_fill_color = "whitesmoke"

    v_color = fill_color(v_color, _v_color, num_v)
    e_color = fill_color(e_color, _e_color, num_e)
    e_fill_color = fill_color(e_fill_color, _e_fill_color, num_e)

    font_family = "sans-serif" if font_family is None else font_family

    return v_color, e_color, e_fill_color, font_family


def default_size(
    num_v: int,
    e_list: List[tuple],
    v_size: Optional[Union[float, list]] = None,
    v_line_width: Optional[Union[float, list]] = None,
    e_line_width: Optional[Union[float, list]] = None,
    font_size: Optional[int] = None,
):
    _v_size = 1 / np.sqrt(num_v + 10) * 0.1
    _v_line_width = 1 * np.exp(-num_v / 50)
    _e_line_width = 1 * np.exp(-len(e_list) / 100)

    v_size = fill_sizes(v_size, _v_size, num_v)
    v_line_width = fill_sizes(v_line_width, _v_line_width, num_v)
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
    _u_size = 1 / np.sqrt(num_u + 12) * 0.08
    _u_line_width = 1 * np.exp(-num_u / 50)
    _v_size = 1 / np.sqrt(num_v + 12) * 0.08
    _v_line_width = 1 * np.exp(-num_v / 50)
    _e_line_width = 1 * np.exp(-len(e_list) / 100)

    u_size = fill_sizes(u_size, _u_size, num_u)
    u_line_width = fill_sizes(u_line_width, _u_line_width, num_u)
    v_size = fill_sizes(v_size, _v_size, num_v)
    v_line_width = fill_sizes(v_line_width, _v_line_width, num_v)
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
    _push_v_strength = 0.006
    _push_e_strength = 0.0
    _pull_e_strength = 0.045
    _pull_center_strength = 0.01

    push_v_strength = fill_strength(push_v_strength, _push_v_strength)
    push_e_strength = fill_strength(push_e_strength, _push_e_strength)
    pull_e_strength = fill_strength(pull_e_strength, _pull_e_strength)
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
    _push_u_strength = 0.005
    _push_v_strength = 0.005
    _push_e_strength = 0.0
    _pull_e_strength = 0.03
    _pull_u_center_strength = 0.04
    _pull_v_center_strength = 0.04

    push_u_strength = fill_strength(push_u_strength, _push_u_strength)
    push_v_strength = fill_strength(push_v_strength, _push_v_strength)
    push_e_strength = fill_strength(push_e_strength, _push_e_strength)
    pull_e_strength = fill_strength(pull_e_strength, _pull_e_strength)
    pull_u_center_strength = fill_strength(pull_u_center_strength, _pull_u_center_strength)
    pull_v_center_strength = fill_strength(pull_v_center_strength, _pull_v_center_strength)

    return (
        push_u_strength,
        push_v_strength,
        push_e_strength,
        pull_e_strength,
        pull_u_center_strength,
        pull_v_center_strength,
    )


def default_hypergraph_strength(
    num_v: int,
    e_list: List[tuple],
    push_v_strength: Optional[float] = None,
    push_e_strength: Optional[float] = None,
    pull_e_strength: Optional[float] = None,
    pull_center_strength: Optional[float] = None,
):
    _push_v_strength = 0.006
    _push_e_strength = 0.008
    _pull_e_strength = 0.007
    _pull_center_strength = 0.001

    push_v_strength = fill_strength(push_v_strength, _push_v_strength)
    push_e_strength = fill_strength(push_e_strength, _push_e_strength)
    pull_e_strength = fill_strength(pull_e_strength, _pull_e_strength)
    pull_center_strength = fill_strength(pull_center_strength, _pull_center_strength)

    return push_v_strength, push_e_strength, pull_e_strength, pull_center_strength


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
