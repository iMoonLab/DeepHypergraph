from typing import Optional
from functools import partial

from dhg.datapipe import load_from_pickle, norm_ft, to_bool_tensor, to_tensor, to_long_tensor

from .base import BaseData


class DBLP8k(BaseData):
    r"""The DBLP-8k dataset is a citation network dataset for link prediction task. 
    The dataset is a part of the dataset crawled according to DBLP API, and we have selected each item based on some conditions, such as the venue and publication year (from 2018 to 2022). It contains 6498 authors and 2603 papers.
    
    The content of the DBLP-8k dataset includes the following:

    - ``num_vertices``: The number of vertices: :math:`8,657`.
    - ``num_edges``: The number of edges: :math:`2,603`.
    - ``edge_list``: The edge list. ``List`` with length :math:`2,603`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to None.
    """
    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("dblp_8k", data_root)
        self._content = {
            "num_vertices": 8657,
            "num_edges": 2603,
            "edge_list": {
                "upon": [{"filename": "edge_list.pkl", "md5": "46b16106daae8eebfd39c2fc43ecbf0b"}],
                "loader": load_from_pickle,
            },
        }


class DBLP4k(BaseData):
    r"""The DBLP-4k dataset is a citation network dataset for node classification task.
    The dataset is an academic network from four research areas. There are 14,475 authors, 
    14,376 papers, and 20 conferences, among which 4,057 authors, 20 conferences and 
    100 papers are labeled with one of the four research areas (database, data mining, machine learning, and information retrieval). 
    The vertice denotes author, and three types of correlation (co-paper, co-term, co-conference) can be used for building hyperedges.
    More details see the `PathSim: Meta Path-Based Top-K Similarity Search in Heterogeneous Information Networks <http://www.vldb.org/pvldb/vol4/p992-sun.pdf>`_ paper.
    
    The content of the DBLP-4k dataset includes the following:

    - ``num_classes``: The number of classes: :math:`4`.
    - ``num_vertices``: The number of vertices: :math:`4,057`.
    - ``num_paper_edges``: The number of hyperedges constructed by the co-paper correlation: :math:`14,328`.
    - ``num_term_edges``: The number of hyperedges constructed by the co-term correlation: :math:`7,723`.
    - ``num_conf_edges``: The number of hyperedges constructed by the co-conference correlation: :math:`20`.
    - ``dim_features``: The dimension of author features: :math:`334`.
    - ``features``: The author feature matrix. ``torch.Tensor`` with size :math:`(4,057 \times 334)`.
    - ``labels``: The label list. ``torch.LongTensor`` with size :math:`(4,057, )`.
    - ``edge_by_paper``: The hyperedge list constructed by the co-paper correlation. ``List`` with length :math:`(14,328)`.
    - ``edge_by_term``: The hyperedge list constructed by the co-term correlation. ``List`` with length :math:`(7,723)`.
    - ``edge_by_conf``: The hyperedge list constructed by the co-conference correlation. ``List`` with length :math:`(20)`.
    - ``paper_author_dict``: The dictionary of ``{paper_id: [author_id, ...]}``. ``Dict`` with length :math:`(14,328)`.
    - ``term_paper_dict``: The dictionary of ``{term_id: [paper_id, ...]}``. ``Dict`` with length :math:`(7,723)`.
    - ``conf_paper_dict``: The dictionary of ``{conf_id: [paper_id, ...]}``. ``Dict`` with length :math:`(20)`.
    
    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to None.
    """

    def __init__(self, data_root: Optional[str] = None):
        super().__init__("dblp_4k", data_root)
        self._content = {
            'num_classes': 4,
            'num_vertices': 4057,
            'num_paper_edges': 14328,
            'num_term_edges': 7723,
            'num_conf_edges': 20,
            'dim_features': 334,
            "features": {
                "upon": [{"filename": "features.pkl", "md5": "7f8e6c3219026c284342d45c01e16406"}],
                "loader": load_from_pickle,
                "preprocess": [to_tensor, partial(norm_ft, ord=1)],
            },
            'labels': {
                'upon': [{'filename': 'labels.pkl', 'md5': '6ffe5ab8c5670d8b5df595b5c4c63184'}],
                'loader': load_from_pickle,
                'preprocess': [to_long_tensor]
            },
            'edge_by_paper': {
                'upon': [{'filename': 'edge_by_paper.pkl', 'md5': 'e473eddeb4692f732bc1e47ae94d62c2'}],
                'loader': load_from_pickle,
            },
            'edge_by_term': {
                'upon': [{'filename': 'edge_by_term.pkl', 'md5': '1ca7cfbf46a7f5fc743818c65392a0ed'}],
                'loader': load_from_pickle,
            },
            'edge_by_conf': {
                'upon': [{'filename': 'edge_by_conf.pkl', 'md5': '890d683b7d8f943ac6d7e87043e0355e'}],
                'loader': load_from_pickle,
            },
            'paper_author_dict': {
                'upon': [{'filename': 'paper_author_dict.pkl', 'md5': 'eb2922e010a78961b5b66e77f9bdf950'}],
                'loader': load_from_pickle,
            },
            'term_paper_dict': {
                'upon': [{'filename': 'term_paper_dict.pkl', 'md5': '1d71f988b52b0e1da9d12f1d3fe24350'}],
                'loader': load_from_pickle,
            },
            'conf_paper_dict': {
                'upon': [{'filename': 'conf_paper_dict.pkl', 'md5': 'cbf87d64dce4ef40d2ab8406e1ee10e1'}],
                'loader': load_from_pickle,
            },
        }
