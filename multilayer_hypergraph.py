import random
import re
import itertools

import matplotlib.pyplot as plt
import torch

from typing import Optional, Union, List
from dhg import Hypergraph
from tqdm import tqdm


class MultilayerHypergraph(Hypergraph):
    r"""The ``MultilayerHypergraph`` class is developed for multilayer hypergraph structures.
    Args:
        ``num_v`` (``int``): The number of vertices in the multilayer hypergraph.
        ``num_layers`` (``int``): The number of layers in the multilayer hypergraph.
        ``layers_list`` (``List[Hypergraph]``): A list of hypergraphs representing each layer in the multilayer hypergraph.
        ``e_list`` (``Optional[Union[List[int], List[List[int]]]]``): A list of hyperedges in the multilayer hypergraph. Default is None.
        ``e_weight`` (``Optional[Union[float, List[float]]]``): A list of weights for the hyperedges in the multilayer hypergraph. Default is None.
        ``v_weight`` (``Optional[List[float]]``): A list of weights for the vertices in the multilayer hypergraph. Default is None.
        ``prob_inner_layer_connect`` (``float``): The probability of connecting nodes between layers. Default is 0.
        ``device`` (``torch.device``): The device on which the multilayer hypergraph will be constructed. Default is CPU.
    """

    def __init__(self,
                 num_v: int,
                 num_layers: int,
                 layers_list: Optional[List[Hypergraph]],
                 e_list: Optional[Union[List[int], List[List[int]]]] = None,
                 e_weight: Optional[Union[float, List[float]]] = None,
                 v_weight: Optional[List[float]] = None,
                 prob_inner_layer_connect: float = 0,
                 device: torch.device = torch.device('cpu')) -> None:

        super().__init__(num_v, e_list, e_weight, v_weight, device=device)

        assert num_v > 1, "number of node must be greater than 1"
        assert num_layers > 1, "number of layers must be greater than 1"
        assert num_layers == len(
            layers_list), "number of layers must be equal the length of layers_list"
        assert 0 <= prob_inner_layer_connect <= 1, "prob_inner_layer_connect must be between 0 and 1"

        self.num_layers = num_layers
        self.layers_list = layers_list
        self.prob_inner_layer_connect = prob_inner_layer_connect
        self.device = device
        self.node_sum_before = []  # 总的节点数必须是列表类型
        self.node_layer = []  # 每层的按概率选择的节点列表必须是列表类型

    def construct_multi_layer_hypergraph(self) -> Hypergraph:
        r"""
        Construct the multilayer hypergraph.
        """

        print("Multilayer hypergraph constructing...")

        prob = self.prob_inner_layer_connect
        _node_layer = []
        layer_num_v = [layer.num_v for layer in self.layers_list]

        for i in tqdm(range(self.num_layers)):
            _node_layer.append(
                MultilayerHypergraph.select_nodes(
                    range(
                        self.layers_list[i].num_v),
                    prob))
            self.node_sum_before.append(sum(layer_num_v[:i]))
            self.node_layer.append(
                MultilayerHypergraph.add_numbers_list(
                    _node_layer, self.node_sum_before[i]))

        temp = [
            self.node_layer[i][j] for i in range(
                self.num_layers) for j in range(
                self.num_layers) if i == j]
        between_layer_hyperedges = list(itertools.product(*temp))
        inter_layer_hyperedges_list = self._inter_layer_connect()
        between_layer_hyperedges += [
            edge for edges in inter_layer_hyperedges_list for edge in edges]

        print("Multilayer hypergraph constructed")
        return MultilayerHypergraph(
            self.num_v_multi(),
            self.num_layers,
            self.layers_list,
            e_list=between_layer_hyperedges,
            device=self.device)

    def num_v_multi(self) -> int:
        r"""
        the number of node in multi-hyper-network
        """
        layer_num_v = []
        for i in range(self.num_layers):
            layer_num_v.append(self.layers_list[i].num_v)
        return sum(layer_num_v)

    def _inter_layer_connect(self) -> list:
        r"""多层超图排序后的超边列表(无层间超边)
        """
        layer_e_list = [layer.e[0] for layer in self.layers_list]
        layer_num_v = [layer.num_v for layer in self.layers_list]
        _layers_e_list = []

        for i in range(self.num_layers):
            self.node_sum_before.append(sum(layer_num_v[:i]))
            _layers_e_list.append(
                MultilayerHypergraph.add_numbers_list(
                    layer_e_list[i], self.node_sum_before[i]))
        return _layers_e_list

    def load_interlayer_mapping(self,
                                path: Optional[Union[str, List[str]]],
                                save: bool = False,
                                save_path=None):
        r"""读取多层超图层间节点的映射关系，用超边连起来各个层间的公共点，构成层间超边
            :param path: 映射关系文件路径, 可以是单个文件路径，也可以是文件路径列表，若有全部映射关系的文件的话，用这个
            若无，则用文件路径列表，每个文件对应一层的映射关系
            return: 等待添加的层间超边
            example:
            file_path = "../data/integrated/gene_merged_file.txt"
            merge_gene_mapping = bio_mhg.load_interlayer_mapping(path=file_path)
            bio_mhg.add_hyperedges(merge_gene_mapping)
        """
        if isinstance(path, str):
            gene_encoding = {}
            with open(path, 'r') as file:
                for line in file:
                    gene, codes = re.findall(
                        r'([^\[]+)\s+\[([^\]]+)\]', line)[0]
                    codes = [int(code) for code in codes.split(',')]
                    if len(codes) > 1:
                        gene_encoding[gene] = codes

            hyperedges_list = []
            for key in gene_encoding.keys():
                # 只有一个编码的不用加入
                if len(gene_encoding[key]) == 1:
                    continue
                value = tuple(gene_encoding[key])
                hyperedges_list.append(value)
            if save:
                with open(save_path, 'w') as f:
                    for gene, code in gene_encoding.items():
                        f.write(f"{gene} {code}\n")
        elif isinstance(path, list):
            gene_info = {}
            gene_encoding = {}
            # 遍历文件列表
            for file in path:
                with open(file, 'r') as f:
                    for line in f:
                        gene, code = line.strip().split('\t')
                        if gene not in gene_encoding:
                            gene_info[gene] = int(code)
                            gene_encoding[gene] = [int(code)]
                        else:
                            gene_encoding[gene].append(int(code))
            # 将合并后的基因列表存储为待添加的超边,长度为1的值可以去掉
            hyperedges_list = []
            for key in gene_encoding.keys():
                # 这会过滤掉那些多层超图里只出现过一次的基因
                if len(gene_encoding[key]) == 1:
                    continue
                # 这会不过滤，保留所有基因
                value = tuple(gene_encoding[key])
                hyperedges_list.append(value)
            # print(hyperedges_list)
            if save:
                # 将结果存储到新的txt文件
                with open(save_path, 'w') as f:
                    for gene, code in gene_encoding.items():
                        f.write(f"{gene} {code}\n")
                with open(save_path, 'w') as f:
                    for gene, code in gene_info.items():
                        f.write(f"{gene} {code}\n")
        return hyperedges_list

    def node_layer_list(self) -> List[List[int]]:
        r"""返回多层超图的节点列表(有层次)
        """
        _node_layer_list = []
        layer_num_v = [layer.num_v for layer in self.layers_list]
        layer_v_list = [layer.v for layer in self.layers_list]

        for i in range(self.num_layers):
            self.node_sum_before.append(sum(layer_num_v[:i]))
            _node_layer_list.append(
                [num + self.node_sum_before[i] for num in layer_v_list[i]])
        return _node_layer_list

    @staticmethod
    def select_nodes(num_v: int, prob: float) -> list:
        r"""依据概率从每层的节点组内选取固定节点数，作为层间相连超边的节点依据"""
        res = []
        for i in range(len(num_v)):
            if random.random() < prob:
                res.append(num_v[i])
        return res

    @staticmethod
    def add_numbers_list(lst: List[List[int]], num: int) -> list:
        res = []
        for e in lst:
            new = [i + num for i in e]
            res.append(tuple(new))
        return res

    def __str__(self):
        r"""Print the multilayer_biological_hypernetwork hypergraph information
        """
        return 'MultilayerHypergraph(num_v={}, num_e={}, num_layers={}, layers_list={}, prob_inner_layer_connect={})'.format(
            self.num_v, self.num_e, self.num_layers, self.layers_list, self.prob_inner_layer_connect)

    def __repr__(self):
        r"""Print the multilayer_biological_hypernetwork hypergrpah information
        """
        return 'MultilayerHypergraph(num_v={}, num_e={}, num_layers={}, layers_list={}, prob_inner_layer_connect={})'.format(
            self.num_v, self.num_e, self.num_layers, self.layers_list, self.prob_inner_layer_connect)
  