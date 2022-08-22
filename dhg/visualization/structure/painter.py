import math
from math import pi

import cairo

class Painter:
    def __init__(self, figsize, adaptor) -> None:
        self.width, self.height = figsize
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.width, self.height)
        self.cr = cairo.Context(self.surface)
        self.cr.scale(self.width, self.height)

        self.node_color = (1.0, 0.2, 0.2)
        self.node_alpha = 1.0
        self.text_color = (0.0, 0.0, 0.0)

        self.base_point_radius = adaptor.point_radius
        self.base_text_font_size = adaptor.text_font_size
        self.base_width = adaptor.width

    def draw_nodes(self, v_list, position, colors=None):
        for idx, text in enumerate(v_list):
            x, y = position[idx]
            if colors is not None:
                color = colors[idx]
            else:
                color = self.node_color
            self._draw_point(x, y, radius=self.base_point_radius, color=color, alpha=self.node_alpha)
            self._draw_text(x, y, text, font_size=self.base_text_font_size, color=self.text_color)

    def draw_edges(self, paths, w, colors=None):

        def _w2width(w):
            return self.base_width * w

        edge_index = 0
        for path in paths:
            if path == 'END':
                edge_index += 1
            elif len(path) == 2: # line
                self._draw_line(path[0][0], path[0][1], path[1][0], path[1][1], width=_w2width(w[edge_index]))
            else: # arc
                self._draw_arc(path[0][0], path[0][1], path[1], path[2], path[3], width=_w2width(w[edge_index]))


    def save(self, filename):
        self.surface.write_to_png(filename)


    def _draw_point(self, x, y, radius=0.01, color=None, alpha=1.0):
        color = color if color is not None else  (0.6, 0.6, 0.2)
        self.cr.set_source_rgba(*color, alpha)
        self.cr.arc(x, y, radius, 0, 2 * pi)
        self.cr.fill()


    def _draw_text(self, x, y, text, font_size=0.02, color=None):
        self.cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL,
                                 cairo.FONT_WEIGHT_BOLD)
        color = color if color is not None else (0.5, 0.5, 1)

        self.cr.set_font_size(font_size)
        self.cr.move_to(x, y)
        self.cr.set_source_rgb(*color)
        self.cr.show_text(str(text))
        self.cr.stroke()

    def _draw_line(self, x1, y1, x2, y2, width=0.003, color=None):
        color = color if color is not None else  (0.2, 0.2, 0.6)
        self.cr.set_source_rgb(*color)

        self.cr.set_line_width(width)
        self.cr.move_to(x1, y1)
        self.cr.line_to(x2, y2)
        self.cr.stroke()

    def _draw_arc(self, xc, yc, angle_start, angle_end, radius=0.005, width=0.003, color=None):
        color = color if color is not None else  (0.2, 0.2, 0.6)
        self.cr.set_source_rgb(*color)

        self.cr.set_line_width(width)
        self.cr.arc(xc, yc, radius, angle_start, angle_end)
        self.cr.stroke()


class SizeAdaptor4Cairo:
    def __init__(self, n_nodes) -> None:
        self._factor = 4.0 / math.sqrt(n_nodes)

        self._base_point_radius = 0.02
        self._base_text_font_size = 0.02
        self._base_width = 0.003
        self._base_edge_radius = 0.025
        self._base_edge_radius_increment = 0.008

        self.point_radius = self._base_point_radius * self._factor
        self.text_font_size = self._base_text_font_size
        self.width = self._base_width * self._factor
        self.edge_radius = self._base_edge_radius * self._factor
        self.edge_radius_increment = self._base_edge_radius_increment * self._factor
