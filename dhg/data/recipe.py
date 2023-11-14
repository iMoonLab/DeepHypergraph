from typing import Optional
from functools import partial

from dhg.datapipe import load_from_pickle, norm_ft, to_tensor, to_long_tensor, to_bool_tensor

from .base import BaseData


class Recipe100k(BaseData):
    r"""The Recipe100k dataset is a recipe-ingredient network dataset for vertex classification task. 
    The vertex features are the bag of words from the sentence that making the recipe. 
    Hyperedges are the ingredients of the recipe or the Keywords for food preparation steps. 
    The original dataset is created in `SHARE: a System for Hierarchical Assistive Recipe Editing <https://arxiv.org/pdf/2105.08185.pdf>`_ paper.
    
    The content of the Recipe100k dataset includes the following:

    - ``num_classes``: The number of classes: :math:`8`.
    - ``num_vertices``: The number of vertices: :math:`101,585`.
    - ``num_edges``: The number of edges: :math:`12,387`.
    - ``dim_features``: The dimension of features: :math:`2,254`.
    - ``features``: The vertex feature matrix. ``torch.Tensor`` with size :math:`(101,585 \times 2,254)`.
    - ``edge_list``: The edge list. ``List`` with length :math:`12,387`.
    - ``labels``: The label list. ``torch.LongTensor`` with size :math:`(101,585, )`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to ``None``.
    """

    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("recipe-100k-v2", data_root)
        self._content = {
            "num_classes": 8,
            "num_vertices": 101585,
            "num_edges": 12387,
            "dim_features": 2254,
            "features": {
                "upon": [{"filename": "features.pkl", "md5": "4fdd76cd4108fd07bdd62368067c1eaf"}],
                "loader": load_from_pickle,
                "preprocess": [to_tensor,],
            },
            "edge_list": {
                "upon": [{"filename": "edge_list.pkl", "md5": "3dc1d8fe7a0f91b5c56057500bda9021"}],
                "loader": load_from_pickle,
            },
            "labels": {
                "upon": [{"filename": "labels.pkl", "md5": "bd8a3bcaef27a58c6d1d5def255c5065"}],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
        }


class Recipe200k(BaseData):
    r"""The Recipe200k dataset is a recipe-ingredient network dataset for vertex classification task. 
    The vertex features are the bag of words from the sentence that making the recipe. 
    Hyperedges are the ingredients of the recipe or the Keywords for food preparation steps. 
    The original dataset is created in `SHARE: a System for Hierarchical Assistive Recipe Editing <https://arxiv.org/pdf/2105.08185.pdf>`_ paper.
    
    The content of the Recipe200k dataset includes the following:

    - ``num_classes``: The number of classes: :math:`8`.
    - ``num_vertices``: The number of vertices: :math:`240,094`.
    - ``num_edges``: The number of edges: :math:`18,129`.
    - ``dim_features``: The dimension of features: :math:`3,200`.
    - ``features``: The vertex feature matrix. ``torch.Tensor`` with size :math:`(240,094 \times 3,200)`.
    - ``edge_list``: The edge list. ``List`` with length :math:`18,129`.
    - ``labels``: The label list. ``torch.LongTensor`` with size :math:`(240,094, )`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to ``None``.
    """

    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("recipe-200k-v2", data_root)
        self._content = {
            "num_classes": 8,
            "num_vertices": 240094,
            "num_edges": 18129,
            "dim_features": 3200,
            "features": {
                "upon": [{"filename": "features.pkl", "md5": "a5df55a3e9591d7389f6ea5f09a483f4"}],
                "loader": load_from_pickle,
                "preprocess": [to_tensor,],
            },
            "edge_list": {
                "upon": [{"filename": "edge_list.pkl", "md5": "163ad784e35e56650fc22658d3e88767"}],
                "loader": load_from_pickle,
            },
            "labels": {
                "upon": [{"filename": "labels.pkl", "md5": "05bee03f1c5383f0cde5ea879be090af"}],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
        }

