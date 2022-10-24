from copy import deepcopy
from typing import Union, Optional, List

import numpy as np
import matplotlib.pyplot as plt

import dhg
from .layout import force_layout, bipartite_force_layout
from .utils import draw_vertex, draw_line_edge, draw_circle_edge

from .defaults import (
    default_style,
    default_size,
    default_strength,
    default_bipartite_style,
    default_bipartite_size,
    default_bipartite_strength,
    default_hypergraph_style,
    default_hypergraph_strength,
)


def draw_graph(
    g: "dhg.Graph",
    e_style: str = "line",
    v_label: Optional[List[str]] = None,
    v_size: Union[float, list] = 1.0,
    v_color: Union[str, list] = "r",
    v_line_width: Union[str, list] = 1.0,
    e_color: Union[str, list] = "gray",
    e_fill_color: Union[str, list] = "whitesmoke",
    e_line_width: Union[str, list] = 1.0,
    font_size: float = 1.0,
    font_family: str = "sans-serif",
    push_v_strength: float = 1.0,
    push_e_strength: float = 1.0,
    pull_e_strength: float = 1.0,
    pull_center_strength: float = 1.0,
):
    r"""Draw the graph structure. The supported edge styles are: ``'line'`` and ``'circle'``.

    Args:
        ``g`` (``dhg.Graph``): The DHG's graph object.
        ``e_style`` (``str``): The edge style. The supported edge styles are: ``'line'`` and ``'circle'``. Defaults to ``'line'``.
        ``v_label`` (``list``, optional): A list of vertex labels. Defaults to ``None``.
        ``v_size`` (``Union[float, list]``): The vertex size. If ``v_size`` is a ``float``, all vertices will have the same size. If ``v_size`` is a ``list``, the size of each vertex will be set according to the corresponding element in the list. Defaults to ``1.0``.
        ``v_color`` (``Union[str, list]``): The vertex `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_. If ``v_color`` is a ``str``, all vertices will have the same color. If ``v_color`` is a ``list``, the color of each vertex will be set according to the corresponding element in the list. Defaults to ``'r'``.
        ``v_line_width`` (``Union[str, list]``): The vertex line width. If ``v_line_width`` is a ``float``, all vertices will have the same line width. If ``v_line_width`` is a ``list``, the line width of each vertex will be set according to the corresponding element in the list. Defaults to ``1.0``.
        ``e_color`` (``Union[str, list]``): The edge `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_. If ``e_color`` is a ``str``, all edges will have the same color. If ``e_color`` is a ``list``, the color of each edge will be set according to the corresponding element in the list. Defaults to ``'gray'``.
        ``e_fill_color`` (``Union[str, list]``): The edge fill color. If ``e_fill_color`` is a ``str``, all edges will have the same fill color. If ``e_fill_color`` is a ``list``, the fill color of each edge will be set according to the corresponding element in the list. Defaults to ``'whitesmoke'``. This argument is only valid when ``e_style`` is ``'circle'``.
        ``e_line_width`` (``Union[str, list]``): The edge line width. If ``e_line_width`` is a ``float``, all edges will have the same line width. If ``e_line_width`` is a ``list``, the line width of each edge will be set according to the corresponding element in the list. Defaults to ``1.0``.
        ``font_size`` (``int``): The font size. Defaults to ``1.0``.
        ``font_family`` (``str``): The font family. Defaults to ``'sans-serif'``.
        ``push_v_strength`` (``float``): The vertex push strength. Defaults to ``1.0``.
        ``push_e_strength`` (``float``): The edge push strength. Defaults to ``1.0``.
        ``pull_e_strength`` (``float``): The edge pull strength. Defaults to ``1.0``.
        ``pull_center_strength`` (``float``): The center pull strength. Defaults to ``1.0``.
    """
    assert isinstance(g, dhg.Graph), "The input object must be a DHG's graph object."
    assert e_style in ["line", "circle"], "e_style must be 'line' or 'circle'"
    assert g.num_e > 0, "g must be a non-empty structure"
    fig, ax = plt.subplots(figsize=(6, 6))
    num_v, e_list = g.num_v, deepcopy(g.e[0])
    # default configures
    v_color, e_color, e_fill_color = default_style(g.num_v, g.num_e, v_color, e_color, e_fill_color)
    v_size, v_line_width, e_line_width, font_size = default_size(num_v, e_list, v_size, v_line_width, e_line_width)
    (push_v_strength, push_e_strength, pull_e_strength, pull_center_strength,) = default_strength(
        num_v, e_list, push_v_strength, push_e_strength, pull_e_strength, pull_center_strength,
    )
    # layout
    v_coor = force_layout(num_v, e_list, push_v_strength, None, pull_e_strength, pull_center_strength)

    if e_style == "line":
        draw_line_edge(
            ax, v_coor, v_size, e_list, False, e_color, e_line_width,
        )
    elif e_style == "circle":
        draw_circle_edge(
            ax, v_coor, v_size, e_list, e_color, e_fill_color, e_line_width,
        )

    draw_vertex(
        ax, v_coor, v_label, font_size, font_family, v_size, v_color, v_line_width,
    )

    plt.xlim((0, 1.0))
    plt.ylim((0, 1.0))
    plt.axis("off")
    fig.tight_layout()


def draw_digraph(
    g: "dhg.DiGraph",
    e_style: str = "line",
    v_label: Optional[List[str]] = None,
    v_size: Union[float, list] = 1.0,
    v_color: Union[str, list] = "r",
    v_line_width: Union[str, list] = 1.0,
    e_color: Union[str, list] = "gray",
    e_line_width: Union[str, list] = 1.0,
    font_size: float = 1.0,
    font_family: str = "sans-serif",
    push_v_strength: float = 1.0,
    push_e_strength: float = 1.0,
    pull_e_strength: float = 1.0,
    pull_center_strength: float = 1.0,
):
    r"""Draw the directed graph structure.

    Args:
        ``g`` (``dhg.DiGraph``): The DHG's directed graph object.
        ``e_style`` (``str``): The edge style. The supported styles are only ``'line'``. Defaults to ``'line'``.
        ``v_label`` (``list``): The vertex label. Defaults to ``None``.
        ``v_size`` (``Union[str, list]``): The vertex size. If ``v_size`` is a ``float``, all vertices will have the same size. If ``v_size`` is a ``list``, the size of each vertex will be set according to the corresponding element in the list. Defaults to ``1.0``.
        ``v_color`` (``Union[str, list]``): The vertex `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_. If ``v_color`` is a ``str``, all vertices will have the same color. If ``v_color`` is a ``list``, the color of each vertex will be set according to the corresponding element in the list. Defaults to ``'r'``.
        ``v_line_width`` (``Union[str, list]``): The vertex line width. If ``v_line_width`` is a ``float``, all vertices will have the same line width. If ``v_line_width`` is a ``list``, the line width of each vertex will be set according to the corresponding element in the list. Defaults to ``1.0``.
        ``e_color`` (``Union[str, list]``): The edge `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_. If ``e_color`` is a ``str``, all edges will have the same color. If ``e_color`` is a ``list``, the color of each edge will be set according to the corresponding element in the list. Defaults to ``'gray'``.
        ``e_line_width`` (``Union[str, list]``): The edge line width. If ``e_line_width`` is a ``float``, all edges will have the same line width. If ``e_line_width`` is a ``list``, the line width of each edge will be set according to the corresponding element in the list. Defaults to ``1.0``.
        ``font_size`` (``int``): The font size. Defaults to ``1.0``.
        ``font_family`` (``str``): The font family. Defaults to ``'sans-serif'``.
        ``push_v_strength`` (``float``): The vertex push strength. Defaults to ``1.0``.
        ``push_e_strength`` (``float``): The edge push strength. Defaults to ``1.0``.
        ``pull_e_strength`` (``float``): The edge pull strength. Defaults to ``1.0``.
        ``pull_center_strength`` (``float``): The center pull strength. Defaults to ``1.0``.
    """
    assert isinstance(g, dhg.DiGraph), "The input object must be a DHG's digraph object."
    assert e_style in ["line"], "e_style must be 'line'"
    assert g.num_e > 0, "g must be a non-empty structure"
    fig, ax = plt.subplots(figsize=(6, 6))
    num_v, e_list = g.num_v, deepcopy(g.e[0])
    # default configures
    v_color, e_color, _ = default_style(g.num_v, g.num_e, v_color, e_color, None)
    v_size, v_line_width, e_line_width, font_size = default_size(num_v, e_list, v_size, v_line_width, e_line_width)
    (push_v_strength, push_e_strength, pull_e_strength, pull_center_strength,) = default_strength(
        num_v, e_list, push_v_strength, push_e_strength, pull_e_strength, pull_center_strength,
    )
    # layout
    v_coor = force_layout(num_v, e_list, push_v_strength, None, pull_e_strength, pull_center_strength)

    if e_style == "line":
        draw_line_edge(
            ax, v_coor, v_size, e_list, True, e_color, e_line_width,
        )
    else:
        raise ValueError("e_style must be 'line'")

    draw_vertex(
        ax, v_coor, v_label, font_size, font_family, v_size, v_color, v_line_width,
    )

    plt.xlim((0, 1.0))
    plt.ylim((0, 1.0))
    plt.axis("off")
    fig.tight_layout()


def draw_bigraph(
    g: "dhg.BiGraph",
    e_style: str = "line",
    u_label: Optional[List[str]] = None,
    u_size: Union[float, list] = 1.0,
    u_color: Union[str, list] = "m",
    u_line_width: Union[str, list] = 1.0,
    v_label: Optional[List[str]] = None,
    v_size: Union[float, list] = 1.0,
    v_color: Union[str, list] = "r",
    v_line_width: Union[str, list] = 1.0,
    e_color: Union[str, list] = "gray",
    e_line_width: Union[str, list] = 1.0,
    u_font_size: float = 1.0,
    v_font_size: float = 1.0,
    font_family: str = "sans-serif",
    push_u_strength: float = 1.0,
    push_v_strength: float = 1.0,
    push_e_strength: float = 1.0,
    pull_e_strength: float = 1.0,
    pull_u_center_strength: float = 1.0,
    pull_v_center_strength: float = 1.0,
):
    r"""Draw the bipartite graph structure.

    Args:
        ``g`` (``dhg.BiGraph``): The DHG's bipartite graph object.
        ``e_style`` (``str``): The edge style. The supported edge styles are only ``'line'``. Defaults to ``'line'``.
        ``u_label`` (``list``): The label of vertices in set :math:`\mathcal{U}`. Defaults to ``None``.
        ``u_size`` (``Union[str, list]``): The size of vertices in set :math:`\mathcal{U}`. If ``u_size`` is a ``float``, all vertices will have the same size. If ``u_size`` is a ``list``, the size of each vertex will be set according to the corresponding element in the list. Defaults to ``1.0``.
        ``u_color`` (``Union[str, list]``): The `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ of vertices in set :math:`\mathcal{U}`. If ``u_color`` is a ``str``, all vertices will have the same color. If ``u_color`` is a ``list``, the color of each vertex will be set according to the corresponding element in the list. Defaults to ``'m'``.
        ``u_line_width`` (``Union[str, list]``): The line width of vertices in set :math:`\mathcal{U}`. If ``u_line_width`` is a ``float``, all vertices will have the same line width. If ``u_line_width`` is a ``list``, the line width of each vertex will be set according to the corresponding element in the list. Defaults to ``1.0``.
        ``v_label`` (``list``): The label of vertices in set :math:`\mathcal{V}`. Defaults to ``None``.
        ``v_size`` (``Union[str, list]``): The size of vertices in set :math:`\mathcal{V}`. If ``v_size`` is a ``float``, all vertices will have the same size. If ``v_size`` is a ``list``, the size of each vertex will be set according to the corresponding element in the list. Defaults to ``1.0``.
        ``v_color`` (``Union[str, list]``): The `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ of vertices in set :math:`\mathcal{V}`. If ``v_color`` is a ``str``, all vertices will have the same color. If ``v_color`` is a ``list``, the color of each vertex will be set according to the corresponding element in the list. Defaults to ``'r'``.
        ``v_line_width`` (``Union[str, list]``): The line width of vertices in set :math:`\mathcal{V}`. If ``v_line_width`` is a ``float``, all vertices will have the same line width. If ``v_line_width`` is a ``list``, the line width of each vertex will be set according to the corresponding element in the list. Defaults to ``1.0``.
        ``e_color`` (``Union[str, list]``): The `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ of edges. If ``e_color`` is a ``str``, all edges will have the same color. If ``e_color`` is a ``list``, the color of each edge will be set according to the corresponding element in the list. Defaults to ``'gray'``.
        ``e_line_width`` (``Union[str, list]``): The line width of edges. If ``e_line_width`` is a ``float``, all edges will have the same line width. If ``e_line_width`` is a ``list``, the line width of each edge will be set according to the corresponding element in the list. Defaults to ``1.0``.
        ``u_font_size`` (``float``): The font size of vertex labels in set :math:`\mathcal{U}`. Defaults to ``1.0``.
        ``v_font_size`` (``float``): The font size of vertex labels in set :math:`\mathcal{V}`. Defaults to ``1.0``.
        ``font_family`` (``str``): The font family of vertex labels. Defaults to ``'sans-serif'``.
        ``push_u_strength`` (``float``): The strength of pushing vertices in set :math:`\mathcal{U}`. Defaults to ``1.0``.
        ``push_v_strength`` (``float``): The strength of pushing vertices in set :math:`\mathcal{V}`. Defaults to ``1.0``.
        ``push_e_strength`` (``float``): The strength of pushing edges. Defaults to ``1.0``.
        ``pull_e_strength`` (``float``): The strength of pulling edges. Defaults to ``1.0``.
        ``pull_u_center_strength`` (``float``): The strength of pulling vertices in set :math:`\mathcal{U}` to the center. Defaults to ``1.0``.
        ``pull_v_center_strength`` (``float``): The strength of pulling vertices in set :math:`\mathcal{V}` to the center. Defaults to ``1.0``.
    """
    assert isinstance(g, dhg.BiGraph), "The input object must be a DHG's bigraph object."
    assert e_style in ["line"], "e_style must be 'line'"
    assert g.num_e > 0, "g must be a non-empty structure"
    fig, ax = plt.subplots(figsize=(6, 6))
    num_u, num_v, e_list = g.num_u, g.num_v, deepcopy(g.e[0])
    # default configures
    u_color, v_color, e_color, _ = default_bipartite_style(num_u, num_v, g.num_e, u_color, v_color, e_color, None)
    (u_size, u_line_width, v_size, v_line_width, e_line_width, u_font_size, v_font_size,) = default_bipartite_size(
        num_u, num_v, e_list, u_size, u_line_width, v_size, v_line_width, e_line_width, u_font_size, v_font_size,
    )
    (
        push_u_strength,
        push_v_strength,
        push_e_strength,
        pull_e_strength,
        pull_u_center_strength,
        pull_v_center_strength,
    ) = default_bipartite_strength(
        num_u,
        num_v,
        e_list,
        push_u_strength,
        push_v_strength,
        push_e_strength,
        pull_e_strength,
        pull_u_center_strength,
        pull_v_center_strength,
    )
    # layout
    u_coor, v_coor = bipartite_force_layout(
        num_u,
        num_v,
        e_list,
        push_u_strength,
        push_v_strength,
        None,
        pull_e_strength,
        pull_u_center_strength,
        pull_v_center_strength,
    )

    # preprocess
    e_list = [(u, v + num_u) for u, v in e_list]
    if e_style == "line":
        draw_line_edge(
            ax, np.vstack([u_coor, v_coor]), u_size + v_size, e_list, False, e_color, e_line_width,
        )
    else:
        raise ValueError("e_style must be 'line'")

    draw_vertex(
        ax,
        np.vstack([u_coor, v_coor]),
        list(u_label) + list(v_label) if u_label is not None and v_label is not None else None,
        u_font_size + v_font_size,
        font_family,
        u_size + v_size,
        u_color + v_color,
        u_line_width + v_line_width,
    )

    plt.xlim((0, 1.0))
    plt.ylim((0, 1.0))
    plt.axis("off")
    fig.tight_layout()


def draw_hypergraph(
    hg: "dhg.Hypergraph",
    e_style: str = "circle",
    v_label: Optional[List[str]] = None,
    v_size: Union[float, list] = 1.0,
    v_color: Union[str, list] = "r",
    v_line_width: Union[str, list] = 1.0,
    e_color: Union[str, list] = "gray",
    e_fill_color: Union[str, list] = "whitesmoke",
    e_line_width: Union[str, list] = 1.0,
    font_size: float = 1.0,
    font_family: str = "sans-serif",
    push_v_strength: float = 1.0,
    push_e_strength: float = 1.0,
    pull_e_strength: float = 1.0,
    pull_center_strength: float = 1.0,
):
    r"""Draw the hypergraph structure.

    Args:
        ``hg`` (``dhg.Hypergraph``): The DHG's hypergraph object.
        ``e_style`` (``str``): The style of hyperedges. The available styles are only ``'circle'``. Defaults to ``'circle'``.
        ``v_label`` (``list``): The labels of vertices. Defaults to ``None``.
        ``v_size`` (``float`` or ``list``): The size of vertices. Defaults to ``1.0``.
        ``v_color`` (``str`` or ``list``): The `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ of vertices. Defaults to ``'r'``.
        ``v_line_width`` (``float`` or ``list``): The line width of vertices. Defaults to ``1.0``.
        ``e_color`` (``str`` or ``list``): The `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ of hyperedges. Defaults to ``'gray'``.
        ``e_fill_color`` (``str`` or ``list``): The fill `color <https://matplotlib.org/stable/gallery/color/named_colors.html>`_ of hyperedges. Defaults to ``'whitesmoke'``.
        ``e_line_width`` (``float`` or ``list``): The line width of hyperedges. Defaults to ``1.0``.
        ``font_size`` (``float``): The font size of labels. Defaults to ``1.0``.
        ``font_family`` (``str``): The font family of labels. Defaults to ``'sans-serif'``.
        ``push_v_strength`` (``float``): The strength of pushing vertices. Defaults to ``1.0``.
        ``push_e_strength`` (``float``): The strength of pushing hyperedges. Defaults to ``1.0``.
        ``pull_e_strength`` (``float``): The strength of pulling hyperedges. Defaults to ``1.0``.
        ``pull_center_strength`` (``float``): The strength of pulling vertices to the center. Defaults to ``1.0``.
    """
    assert isinstance(hg, dhg.Hypergraph), "The input object must be a DHG's hypergraph object."
    assert e_style in ["circle"], "e_style must be 'circle'"
    assert hg.num_e > 0, "g must be a non-empty structure"
    fig, ax = plt.subplots(figsize=(6, 6))

    num_v, e_list = hg.num_v, deepcopy(hg.e[0])
    # default configures
    v_color, e_color, e_fill_color = default_hypergraph_style(hg.num_v, hg.num_e, v_color, e_color, e_fill_color)
    v_size, v_line_width, e_line_width, font_size = default_size(num_v, e_list, v_size, v_line_width, e_line_width)
    (push_v_strength, push_e_strength, pull_e_strength, pull_center_strength,) = default_hypergraph_strength(
        num_v, e_list, push_v_strength, push_e_strength, pull_e_strength, pull_center_strength,
    )
    # layout
    v_coor = force_layout(num_v, e_list, push_v_strength, push_e_strength, pull_e_strength, pull_center_strength)
    if e_style == "circle":
        draw_circle_edge(
            ax, v_coor, v_size, e_list, e_color, e_fill_color, e_line_width,
        )
    else:
        raise ValueError("e_style must be 'circle'")

    draw_vertex(
        ax, v_coor, v_label, font_size, font_family, v_size, v_color, v_line_width,
    )

    plt.xlim((0, 1.0))
    plt.ylim((0, 1.0))
    plt.axis("off")
    fig.tight_layout()
