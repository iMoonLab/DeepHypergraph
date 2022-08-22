import pytest
import numpy as np

from dhg.visualization.structure.draw2 import draw_graph
from dhg.random import graph_Gnm


import matplotlib.pyplot as plt

# from dhg.visualization.structure import visualization

# class Hygraph:
#     def __init__(self) -> None:
#         self.v_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']
#         self.e_list = [
#             [0, 1],
#             [0, 2],
#             [0, 2, 3],
#             [0, 2, 4],
#             [1, 2, 5, 6, 7],
#             [1, 2, 3],
#             [3, 4, 8, 9, 10, 12],
#             [4, 5, 7, 6, 10, 11, 13],
#             [10, 11, 7, 8, 0, 1, 2],
#             [13, 4, 2, 5, 7, 6, 8],
#         ]
#         self.w = np.ones(len(self.e_list))
# class Graph:
#     def __init__(self) -> None:
#         self.v_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']
#         self.e_list = [
#             [0, 1],
#             [0, 2],
#             [0, 3],
#             [0, 4],
#             [5, 7],
#             [2, 3],
#             [8, 9],
#             [10, 11],
#             [7, 8],
#             [13, 8],
#         ]
#         self.w = np.ones(len(self.e_list))

def test_visual():
    g = graph_Gnm(100, 200)
    draw_graph(g, e_style='line')
    plt.show()


# def test_visual2():
#     # g = graph_Gnm(100, 20)
#     # draw_graph(g)
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.arrow(0, 0, 0.2, 0.2)
#     ax.arrow(1, 0, -0.2, 0.2, head_width=0)
#     plt.show()
