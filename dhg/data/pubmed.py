from typing import Optional
from functools import partial

from dhg.datapipe import load_from_pickle, norm_ft, to_tensor, to_long_tensor, to_bool_tensor

from .base import BaseData


class Pubmed(BaseData):
    r"""The PubMed dataset is a citation network dataset for vertex classification task. 
    More details can be found in this `website <https://pubmed.ncbi.nlm.nih.gov/download/>`_.

    - ``num_classes``: The number of classes: :math:`3`.
    - ``num_vertices``: The number of vertices: :math:`19,717`.
    - ``num_edges``: The number of edges: :math:`88,676`.
    - ``dim_features``: The dimension of features: :math:`500`.
    - ``features``: The vertex feature matrix. ``torch.Tensor`` with size :math:`(19,717 \times 500)`.
    - ``edge_list``: The edge list. ``List`` with length :math:`(88,676 \times 2)`.
    - ``labels``: The label list. ``torch.LongTensor`` with size :math:`(19,717, )`.
    - ``train_mask``: The train mask. ``torch.BoolTensor`` with size :math:`(19,717, )`.
    - ``val_mask``: The validation mask. ``torch.BoolTensor`` with size :math:`(19,717, )`.
    - ``test_mask``: The test mask. ``torch.BoolTensor`` with size :math:`(19,717, )`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to ``None``.
    """

    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("pubmed", data_root)
        self._content = {
            "num_classes": 3,
            "num_vertices": 19717,
            "num_edges": 88676,
            "dim_features": 500,
            "features": {
                "upon": [{"filename": "features.pkl", "md5": "b8fe6c0ce974d031c9c70266f4ccff44"}],
                "loader": load_from_pickle,
                "preprocess": [to_tensor, partial(norm_ft, ord=1)],
            },
            "edge_list": {
                "upon": [{"filename": "edge_list.pkl", "md5": "9563ff5fc66e56ab53ccb25685e6d540"}],
                "loader": load_from_pickle,
            },
            "labels": {
                "upon": [{"filename": "labels.pkl", "md5": "6132b80c5cea4e73f45920779175e3f8"}],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
            "train_mask": {
                "upon": [{"filename": "train_mask.pkl", "md5": "69d4ef4d7cdb53ff4b3b48ce394363b0"}],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "val_mask": {
                "upon": [{"filename": "val_mask.pkl", "md5": "5a65a2ad27165dd0cea2675592ee414e"}],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "test_mask": {
                "upon": [{"filename": "test_mask.pkl", "md5": "4bbf50754d7fdae2b5c6c12d85ccc3a5"}],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
        }


class PubmedBiGraph(BaseData):
    r"""The PubmedBiGraph dataset is a citation network dataset for vertex classification task. 
    These are synthetic bipartite graph datasets that are generated from citation networks (single graph) 
    where documents and citation links between them are treated as nodes and undirected edges, respectively.
    More details see the `Cascade-BGNN: Toward Efficient Self-supervised Representation Learning on Large-scale Bipartite Graphs <https://arxiv.org/pdf/1906.11994.pdf>`_ paper.
    
    The content of the PubmedBiGraph dataset includes the following:

    - ``num_u_classes``: The number of classes in set :math:`U` : :math:`3`.
    - ``num_u_vertices``: The number of vertices in set :math:`U` : :math:`13,424`.
    - ``num_v_vertices``: The number of vertices in set :math:`V` : :math:`3,435`.
    - ``num_edges``: The number of edges: :math:`18,782`.
    - ``dim_u_features``: The dimension of features in set :math:`U` : :math:`400`.
    - ``dim_v_features``: The dimension of features in set :math:`V` : :math:`500`.
    - ``u_features``: The vertex feature matrix in set :math:`U`. ``torch.Tensor`` with size :math:`(13,424 \times 400)`.
    - ``v_features``: The vertex feature matrix in set :math:`V` . ``torch.Tensor`` with size :math:`(3,435 \times 500)`.
    - ``edge_list``: The edge list. ``List`` with length :math:`(2,314 \times 2)`.
    - ``u_labels``: The label list in set :math:`U` . ``torch.LongTensor`` with size :math:`(13,424, )`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to ``None``.
    """

    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("pubmed_bigraph", data_root)
        self._content = {
            "num_u_classes": 3,
            "num_u_vertices": 13424,
            "num_v_vertices": 3435,
            "num_edges": 18782,
            "dim_u_features": 400,
            "dim_v_features": 500,
            "u_features": {
                "upon": [{"filename": "u_features.pkl", "md5": "0ff95930275f4ce30306defc3cdf488a"}],
                "loader": load_from_pickle,
                "preprocess": [to_tensor, partial(norm_ft, ord=1)],
            },
            "v_features": {
                "upon": [{"filename": "v_features.pkl", "md5": "93760475e0cdd1fa9ce4e97e669d2c7e"}],
                "loader": load_from_pickle,
                "preprocess": [to_tensor, partial(norm_ft, ord=1)],
            },
            "edge_list": {
                "upon": [{"filename": "edge_list.pkl", "md5": "8bf3949ae0df62eb8b032e0e047def32"}],
                "loader": load_from_pickle,
            },
            "u_labels": {
                "upon": [{"filename": "u_labels.pkl", "md5": "ce286f6dd401679461913aad64f0f577"}],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
        }
