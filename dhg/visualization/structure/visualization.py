from typing import Optional, Union, List
from pathlib import Path
import numpy as np


from .layout import _force_directed_layout, _hull_layout, _check_cover
from .painter import Painter, SizeAdaptor4Cairo

from dhg.structure.graphs import Graph, DiGraph
from dhg.structure.hypergraphs import Hypergraph


def vis_graph(
    graph: 'Graph',
    save_filename: Optional[str] = None,
    v_labels: Optional[List[str]] = None,
    e_weighted: bool = False,
    v_colors: Union[str, List[str]] = "pink",
    e_colors: Union[str, List[str]] = "black",
    background_color: Optional[str] = None,
):
    ...


def vis_hypergraph(
    hypergraph: 'Hypergraph',
    save_filename: Optional[str] = None,
    v_labels: Optional[List[str]] = None,
    e_weighted: bool = False,
    v_colors: Union[str, List[str]] = "pink",
    e_colors: Union[str, List[str]] = "black",
    background_color: Optional[str] = None,
):
    ...

def visualize(graph, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    v_list = graph.v_list
    e_list = graph.e_list
    w = graph.w

    n_v = len(v_list)
    n_e = len(e_list)

    adaptor = SizeAdaptor4Cairo(n_v)

    def _generate_H_from_e_list(n_v, e_list):
        H = np.zeros((n_v, len(e_list)), dtype=np.float32)
        for idx, edge in enumerate(e_list):
            for node in edge:
                H[node, idx] = 1
        return H

    H = _generate_H_from_e_list(n_v, e_list)

    position = _force_directed_layout(n_v, H)
    paths, _ = _hull_layout(
        n_v,
        e_list,
        position,
        init_radius=adaptor.edge_radius,
        radius_increment=adaptor.edge_radius_increment,
    )

    pt = Painter((1024, 1024), adaptor=adaptor)
    pt.draw_nodes(v_list, position)
    pt.draw_edges(paths, w)
    pt.save(filename)
