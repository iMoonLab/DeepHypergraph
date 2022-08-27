import pytest
import numpy as np
from dhg.random.graphs.graph import graph_Gnp

from dhg.visualization.structure.draw2 import draw_graph
from dhg.random import graph_Gnm, graph_Gnp, graph_Gnp_fast


import matplotlib.pyplot as plt


def test_visual():
    g = graph_Gnp_fast(100, 0.015)
    draw_graph(g, e_style="line")

