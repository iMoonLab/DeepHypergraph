import itertools
import random
import re
from typing import Iterable, List, Optional, Union

import torch

from .hypergraph import Hypergraph


class MultilayerHypergraph(Hypergraph):
    """A hypergraph composed of multiple hypergraph layers."""

    def __init__(
        self,
        num_v: int,
        num_layers: int,
        layers_list: List[Hypergraph],
        e_list=None,
        e_weight=None,
        v_weight=None,
        prob_inner_layer_connect: float = 0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(num_v, e_list, e_weight, v_weight, device=device)
        if num_v <= 1:
            raise ValueError("num_v must be greater than 1")
        if num_layers <= 1:
            raise ValueError("num_layers must be greater than 1")
        if layers_list is None or num_layers != len(layers_list):
            raise ValueError("num_layers must equal the length of layers_list")
        if not 0 <= prob_inner_layer_connect <= 1:
            raise ValueError("prob_inner_layer_connect must be between 0 and 1")

        self.num_layers = num_layers
        self.layers_list = list(layers_list)
        self.prob_inner_layer_connect = prob_inner_layer_connect
        self.node_layer: List[List[int]] = []

    def _compute_offsets(self) -> List[int]:
        """Return each layer's starting vertex index in the combined structure."""
        offsets = []
        total = 0
        for layer in self.layers_list:
            offsets.append(total)
            total += layer.num_v
        return offsets

    def construct_multi_layer_hypergraph(self) -> Hypergraph:
        """Build and return the flattened hypergraph for all layers."""
        offsets = self._compute_offsets()
        layer_nodes = [
            self.select_nodes(range(layer.num_v), self.prob_inner_layer_connect)
            for layer in self.layers_list
        ]
        self.node_layer = [
            [node + offset for node in nodes]
            for nodes, offset in zip(layer_nodes, offsets)
        ]

        inter_layer_hyperedges = list(itertools.product(*self.node_layer))
        layer_hyperedges = self._inter_layer_connect()
        all_hyperedges = inter_layer_hyperedges + [
            edge for edges in layer_hyperedges for edge in edges
        ]
        return Hypergraph(
            self.num_v_multi(), all_hyperedges, device=self.device
        )

    def num_v_multi(self) -> int:
        """Return the total number of vertices across all layers."""
        return sum(layer.num_v for layer in self.layers_list)

    def _inter_layer_connect(self) -> List[list]:
        """Return each layer's hyperedges with its global vertex offset applied."""
        offsets = self._compute_offsets()
        return [
            self.add_numbers_list(layer.e[0], offset)
            for layer, offset in zip(self.layers_list, offsets)
        ]

    def load_interlayer_mapping(
        self,
        path: Optional[Union[str, List[str]]],
        save: bool = False,
        save_path=None,
    ) -> List[tuple]:
        """Load gene mappings and return hyperedges for genes shared across layers.

        ``path`` may be one file containing ``gene [codes]`` records, or a list
        of tab-separated ``gene<TAB>code`` files, one per layer.
        """
        if isinstance(path, str):
            gene_encoding = {}
            with open(path, "r", encoding="utf-8") as file:
                for line in file:
                    match = re.findall(r"([^\[]+)\s+\[([^\]]+)\]", line)
                    if not match:
                        continue
                    gene, codes = match[0]
                    gene_encoding[gene.strip()] = [
                        int(code.strip()) for code in codes.split(",")
                    ]
        elif isinstance(path, list):
            gene_encoding = {}
            for file_path in path:
                with open(file_path, "r", encoding="utf-8") as file:
                    for line in file:
                        gene, code = line.rstrip("\n").split("\t", 1)
                        gene_encoding.setdefault(gene, []).append(int(code))
        else:
            raise TypeError("path must be a file path or a list of file paths")

        if save:
            if save_path is None:
                raise ValueError("save_path is required when save=True")
            with open(save_path, "w", encoding="utf-8") as file:
                for gene, codes in gene_encoding.items():
                    file.write(f"{gene} {codes}\n")

        return [tuple(codes) for codes in gene_encoding.values() if len(codes) > 1]

    def node_layer_list(self) -> List[List[int]]:
        """Return the vertices in each layer using flattened global indices."""
        offsets = self._compute_offsets()
        return [
            [node + offset for node in layer.v]
            for layer, offset in zip(self.layers_list, offsets)
        ]

    @staticmethod
    def select_nodes(num_v: Iterable[int], prob: float) -> list:
        """Select each vertex independently with probability ``prob``."""
        return [node for node in num_v if random.random() < prob]

    @staticmethod
    def add_numbers_list(lst: Iterable[Iterable[int]], num: int) -> list:
        """Add ``num`` to every vertex in each hyperedge."""
        return [tuple(vertex + num for vertex in edge) for edge in lst]

    def __str__(self):
        return (
            "MultilayerHypergraph(num_v={}, num_e={}, num_layers={}, "
            "layers_list={}, prob_inner_layer_connect={})"
        ).format(
            self.num_v,
            self.num_e,
            self.num_layers,
            self.layers_list,
            self.prob_inner_layer_connect,
        )

    __repr__ = __str__