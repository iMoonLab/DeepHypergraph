from typing import Optional
from functools import partial

from dhg.datapipe import load_from_pickle, norm_ft, to_tensor, to_long_tensor, to_bool_tensor

from .base import BaseData


class Cora(BaseData):
    r"""The Cora dataset is a citation network dataset for vertex classification task. 
    More details can be found in this `website <https://relational.fit.cvut.cz/dataset/CORA>`_.
    
    The content of the Cora dataset includes the following:

    - ``num_classes``: The number of classes: :math:`7`.
    - ``num_vertices``: The number of vertices: :math:`2,708`.
    - ``num_edges``: The number of edges: :math:`10,858`.
    - ``dim_features``: The dimension of features: :math:`1,433`.
    - ``features``: The vertex feature matrix. ``torch.Tensor`` with size :math:`(2,708 \times 1,433)`.
    - ``edge_list``: The edge list. ``List`` with length :math:`(10,858 \times 2)`.
    - ``labels``: The label list. ``torch.LongTensor`` with size :math:`(2,708, )`.
    - ``train_mask``: The train mask. ``torch.BoolTensor`` with size :math:`(2,708, )`.
    - ``val_mask``: The validation mask. ``torch.BoolTensor`` with size :math:`(2,708, )`.
    - ``test_mask``: The test mask. ``torch.BoolTensor`` with size :math:`(2,708, )`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to ``None``.
    """

    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("cora", data_root)
        self._content = {
            "num_classes": 7,
            "num_vertices": 2708,
            "num_edges": 10858,
            "dim_features": 1433,
            "features": {
                "upon": [{"filename": "features.pkl", "md5": "05b45e9c38cc95f4fc44b3668cc9ddc9"}],
                "loader": load_from_pickle,
                "preprocess": [to_tensor, partial(norm_ft, ord=1)],
            },
            "edge_list": {
                "upon": [{"filename": "edge_list.pkl", "md5": "f488389c1edd0d898ce273fbd27822b3"}],
                "loader": load_from_pickle,
            },
            "labels": {
                "upon": [{"filename": "labels.pkl", "md5": "e506014762052c6a36cb583c28bdae1d"}],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
            "train_mask": {
                "upon": [{"filename": "train_mask.pkl", "md5": "a11357a40e1f0b5cce728d1a961b8e13"}],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "val_mask": {
                "upon": [{"filename": "val_mask.pkl", "md5": "355544da566452601bcfa74d30539a71"}],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "test_mask": {
                "upon": [{"filename": "test_mask.pkl", "md5": "bbfc87d661560f55f6946f8cb9d602b9"}],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
        }


class CoraBiGraph(BaseData):
    r"""The CoraBiGraph dataset is a citation network dataset for vertex classification task. 
    These are synthetic bipartite graph datasets that are generated from citation networks (single graph) 
    where documents and citation links between them are treated as nodes and undirected edges, respectively.
    More details see the `Cascade-BGNN: Toward Efficient Self-supervised Representation Learning on Large-scale Bipartite Graphs <https://arxiv.org/pdf/1906.11994.pdf>`_ paper.
    
    The content of the CoraBiGraph dataset includes the following:

    - ``num_u_classes``: The number of classes in set :math:`U` : :math:`7`.
    - ``num_u_vertices``: The number of vertices in set :math:`U` : :math:`1,312`.
    - ``num_v_vertices``: The number of vertices in set :math:`V` : :math:`789`.
    - ``num_edges``: The number of edges: :math:`2,314`.
    - ``dim_u_features``: The dimension of features in set :math:`U` : :math:`1,433`.
    - ``dim_v_features``: The dimension of features in set :math:`V` : :math:`1,433`.
    - ``u_features``: The vertex feature matrix in set :math:`U`. ``torch.Tensor`` with size :math:`(1,312 \times 1,433)`.
    - ``v_features``: The vertex feature matrix in set :math:`V` . ``torch.Tensor`` with size :math:`(789 \times 1,433)`.
    - ``edge_list``: The edge list. ``List`` with length :math:`(2,314 \times 2)`.
    - ``u_labels``: The label list in set :math:`U` . ``torch.LongTensor`` with size :math:`(1,312, )`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to ``None``.
    """

    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("cora_bigraph", data_root)
        self._content = {
            "num_u_classes": 7,
            "num_u_vertices": 1312,
            "num_v_vertices": 789,
            "num_edges": 2314,
            "dim_u_features": 1433,
            "dim_v_features": 1433,
            "u_features": {
                "upon": [{"filename": "u_features.pkl", "md5": "84f0ecee4233ca70d40d36f457470032"}],
                "loader": load_from_pickle,
                "preprocess": [to_tensor, partial(norm_ft, ord=1)],
            },
            "v_features": {
                "upon": [{"filename": "v_features.pkl", "md5": "de65cd478ea05333c26184bc8b2cb468"}],
                "loader": load_from_pickle,
                "preprocess": [to_tensor, partial(norm_ft, ord=1)],
            },
            "edge_list": {
                "upon": [{"filename": "edge_list.pkl", "md5": "e7b82c4a8305e1488beac1b788ad46e3"}],
                "loader": load_from_pickle,
            },
            "u_labels": {
                "upon": [{"filename": "u_labels.pkl", "md5": "65dff86f7920cdab61790d48a39f2e5b"}],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
        }

