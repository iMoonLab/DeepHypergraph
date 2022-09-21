from typing import Optional
from functools import partial

from dhg.datapipe import load_from_pickle, norm_ft, to_tensor, to_long_tensor, to_bool_tensor

from .base import BaseData


class Citeseer(BaseData):
    r"""The Citeseer dataset is a citation network dataset for vertex classification task. 
    More details can be found in this `website <https://relational.fit.cvut.cz/dataset/CiteSeer>`_.

    - ``num_classes``: The number of classes: :math:`6`.
    - ``num_vertices``: The number of vertices: :math:`3,327`.
    - ``num_edges``: The number of edges: :math:`9,464`.
    - ``dim_features``: The dimension of features: :math:`3,703`.
    - ``features``: The vertex feature matrix. ``torch.Tensor`` with size :math:`(3,327 \times 3,703)`.
    - ``edge_list``: The edge list. ``List`` with length :math:`(9,464 \times 2)`.
    - ``labels``: The label list. ``torch.LongTensor`` with size :math:`(3,327, )`.
    - ``train_mask``: The train mask. ``torch.BoolTensor`` with size :math:`(3,327, )`.
    - ``val_mask``: The validation mask. ``torch.BoolTensor`` with size :math:`(3,327, )`.
    - ``test_mask``: The test mask. ``torch.BoolTensor`` with size :math:`(3,327, )`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to ``None``.
    """

    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("citeseer", data_root)
        self._content = {
            "num_classes": 6,
            "num_vertices": 3327,
            "num_edges": 9464,
            "dim_features": 3703,
            "features": {
                "upon": [{"filename": "features.pkl", "md5": "7458c683e584a0c5ce1ab7af763777c6"}],
                "loader": load_from_pickle,
                "preprocess": [to_tensor, partial(norm_ft, ord=1)],
            },
            "edge_list": {
                "upon": [{"filename": "edge_list.pkl", "md5": "1948e9f712bc16ba8ef48a3e79fc2246"}],
                "loader": load_from_pickle,
            },
            "labels": {
                "upon": [{"filename": "labels.pkl", "md5": "f5bcf7815e463af4f88d40195f0d378c"}],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
            "train_mask": {
                "upon": [{"filename": "train_mask.pkl", "md5": "9aae62b41403b976c4cc048685c966e6"}],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "val_mask": {
                "upon": [{"filename": "val_mask.pkl", "md5": "4527d7dc1e2604cdaa9e18916f32714b"}],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "test_mask": {
                "upon": [{"filename": "test_mask.pkl", "md5": "af49e6f6f53c73b7d3a62d6f9b2a3871"}],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
        }


class CiteseerBiGraph(BaseData):
    r"""The CiteseerBiGraph dataset is a citation network dataset for vertex classification task. 
    These are synthetic bipartite graph datasets that are generated from citation networks (single graph) 
    where documents and citation links between them are treated as nodes and undirected edges, respectively.
    More details see the `Cascade-BGNN: Toward Efficient Self-supervised Representation Learning on Large-scale Bipartite Graphs <https://arxiv.org/pdf/1906.11994.pdf>`_ paper.
    
    The content of the CiteseerBiGraph dataset includes the following:

    - ``num_u_classes``: The number of classes in set :math:`U` : :math:`6`.
    - ``num_u_vertices``: The number of vertices in set :math:`U` : :math:`1,237`.
    - ``num_v_vertices``: The number of vertices in set :math:`V` : :math:`742`.
    - ``num_edges``: The number of edges: :math:`1,665`.
    - ``dim_u_features``: The dimension of features in set :math:`U` : :math:`3,703`.
    - ``dim_v_features``: The dimension of features in set :math:`V` : :math:`3,703`.
    - ``u_features``: The vertex feature matrix in set :math:`U`. ``torch.Tensor`` with size :math:`(1,237 \times 3,703)`.
    - ``v_features``: The vertex feature matrix in set :math:`V` . ``torch.Tensor`` with size :math:`(742 \times 3,703)`.
    - ``edge_list``: The edge list. ``List`` with length :math:`(1,665 \times 2)`.
    - ``u_labels``: The label list in set :math:`U` . ``torch.LongTensor`` with size :math:`(1,237, )`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to ``None``.
    """

    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("citeseer_bigraph", data_root)
        self._content = {
            "num_u_classes": 6,
            "num_u_vertices": 1237,
            "num_v_vertices": 742,
            "num_edges": 1665,
            "dim_u_features": 3703,
            "dim_v_features": 3703,
            "u_features": {
                "upon": [{"filename": "u_features.pkl", "md5": "d8c1ccd6026cbb1f05cc3c534b239e00"}],
                "loader": load_from_pickle,
                "preprocess": [to_tensor, partial(norm_ft, ord=1)],
            },
            "v_features": {
                "upon": [{"filename": "v_features.pkl", "md5": "7ca1d16ad557945f9b66ef6ac40c0210"}],
                "loader": load_from_pickle,
                "preprocess": [to_tensor, partial(norm_ft, ord=1)],
            },
            "edge_list": {
                "upon": [{"filename": "edge_list.pkl", "md5": "2a632085fb8f691af6399fbb71dc1f67"}],
                "loader": load_from_pickle,
            },
            "u_labels": {
                "upon": [{"filename": "u_labels.pkl", "md5": "b4d0034c29f6f5b6da17f3037c2af605"}],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
        }
