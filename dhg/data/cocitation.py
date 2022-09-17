from typing import Optional
from functools import partial

from dhg.datapipe import load_from_pickle, norm_ft, to_tensor, to_long_tensor, to_bool_tensor

from .base import BaseData


class CocitationCora(BaseData):
    r"""The Co-citation Cora dataset is a citation network dataset for vertex classification task. 
    More details see the `HyperGCN <https://papers.nips.cc/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf>`_ paper.
    
    The content of the Co-citation Cora dataset includes the following:

    - ``num_classes``: The number of classes: :math:`7`.
    - ``num_vertices``: The number of vertices: :math:`2,708`.
    - ``num_edges``: The number of edges: :math:`1,579`.
    - ``dim_features``: The dimension of features: :math:`1,433`.
    - ``features``: The vertex feature matrix. ``torch.Tensor`` with size :math:`(2,708 \times 1,433)`.
    - ``edge_list``: The edge list. ``List`` with length :math:`1,579`.
    - ``labels``: The label list. ``torch.LongTensor`` with size :math:`(2,708, )`.
    - ``train_mask``: The train mask. ``torch.BoolTensor`` with size :math:`(2,708, )`.
    - ``val_mask``: The validation mask. ``torch.BoolTensor`` with size :math:`(2,708, )`.
    - ``test_mask``: The test mask. ``torch.BoolTensor`` with size :math:`(2,708, )`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to ``None``.
    """

    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("cocitation_cora", data_root)
        self._content = {
            "num_classes": 7,
            "num_vertices": 2708,
            "num_edges": 1579,
            "dim_features": 1433,
            "features": {
                "upon": [{"filename": "features.pkl", "md5": "14257c0e24b4eb741b469a351e524785"}],
                "loader": load_from_pickle,
                "preprocess": [to_tensor, partial(norm_ft, ord=1)],
            },
            "edge_list": {
                "upon": [{"filename": "edge_list.pkl", "md5": "e43d1321880c8ecb2260d8fb7effd9ea"}],
                "loader": load_from_pickle,
            },
            "labels": {
                "upon": [{"filename": "labels.pkl", "md5": "c8d11c452e0be69f79a47dd839279117"}],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
            "train_mask": {
                "upon": [{"filename": "train_mask.pkl", "md5": "111db6c6f986be2908378df7bdca7a9b"}],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "val_mask": {
                "upon": [{"filename": "val_mask.pkl", "md5": "ffab1055193ffb2fe74822bb575d332a"}],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "test_mask": {
                "upon": [{"filename": "test_mask.pkl", "md5": "ffab1055193ffb2fe74822bb575d332a"}],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
        }


class CocitationCiteseer(BaseData):
    r"""The Co-citation Citeseer dataset is a citation network dataset for vertex classification task. 
    More details see the `HyperGCN <https://papers.nips.cc/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf>`_ paper.

    The content of the Co-citation Citaseer dataset includes the following:

    - ``num_classes``: The number of classes: :math:`6`.
    - ``num_vertices``: The number of vertices: :math:`3,327`.
    - ``num_edges``: The number of edges: :math:`1,079`.
    - ``dim_features``: The dimension of features: :math:`3,703`.
    - ``features``: The vertex feature matrix. ``torch.Tensor`` with size :math:`(3,327 \times 3,703)`.
    - ``edge_list``: The edge list. ``List`` with length :math:`1,079`.
    - ``labels``: The label list. ``torch.LongTensor`` with size :math:`(3,327, )`.
    - ``train_mask``: The train mask. ``torch.BoolTensor`` with size :math:`(3,327, )`.
    - ``val_mask``: The validation mask. ``torch.BoolTensor`` with size :math:`(3,327, )`.
    - ``test_mask``: The test mask. ``torch.BoolTensor`` with size :math:`(3,327, )`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to ``None``.
    """

    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("cocitation_citeseer", data_root)
        self._content = {
            "num_classes": 6,
            "num_vertices": 3312,
            "num_edges": 1079,
            "dim_features": 3703,
            "features": {
                "upon": [{"filename": "features.pkl", "md5": "1ee0dc89e0d5f5ac9187b55a407683e8"}],
                "loader": load_from_pickle,
                "preprocess": [to_tensor, partial(norm_ft, ord=1)],
            },
            "edge_list": {
                "upon": [{"filename": "edge_list.pkl", "md5": "6687b2e96159c534a424253f536b49ae"}],
                "loader": load_from_pickle,
            },
            "labels": {
                "upon": [{"filename": "labels.pkl", "md5": "71069f78e83fa85dd6a4b9b6570447c2"}],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
            "train_mask": {
                "upon": [{"filename": "train_mask.pkl", "md5": "3b831318fc3d3e588bead5ba469fe38f"}],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "val_mask": {
                "upon": [{"filename": "val_mask.pkl", "md5": "c22eb5b7493908042c7e039c8bb5a82e"}],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "test_mask": {
                "upon": [{"filename": "test_mask.pkl", "md5": "c22eb5b7493908042c7e039c8bb5a82e"}],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
        }


class CocitationPubmed(BaseData):
    r"""The Co-citation PubMed dataset is a citation network dataset for vertex classification task. 
    More details see the `HyperGCN <https://papers.nips.cc/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf>`_ paper.

    The content of the Co-citation PubMed dataset includes the following:

    - ``num_classes``: The number of classes: :math:`3`.
    - ``num_vertices``: The number of vertices: :math:`19,717`.
    - ``num_edges``: The number of edges: :math:`7,963`.
    - ``dim_features``: The dimension of features: :math:`500`.
    - ``features``: The vertex feature matrix. ``torch.Tensor`` with size :math:`(19,717 \times 500)`.
    - ``edge_list``: The edge list. ``List`` with length :math:`7,963`.
    - ``labels``: The label list. ``torch.LongTensor`` with size :math:`(19,717, )`.
    - ``train_mask``: The train mask. ``torch.BoolTensor`` with size :math:`(19,717, )`.
    - ``val_mask``: The validation mask. ``torch.BoolTensor`` with size :math:`(19,717, )`.
    - ``test_mask``: The test mask. ``torch.BoolTensor`` with size :math:`(19,717, )`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to ``None``.
    """

    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("cocitation_pubmed", data_root)
        self._content = {
            "num_classes": 3,
            "num_vertices": 19717,
            "num_edges": 7963,
            "dim_features": 500,
            "features": {
                "upon": [{"filename": "features.pkl", "md5": "f89502c432ca451156a8235c5efc034e"}],
                "loader": load_from_pickle,
                "preprocess": [to_tensor, partial(norm_ft, ord=1)],
            },
            "edge_list": {
                "upon": [{"filename": "edge_list.pkl", "md5": "c5fbedf63e5be527f200e8c4e0391b00"}],
                "loader": load_from_pickle,
            },
            "labels": {
                "upon": [{"filename": "labels.pkl", "md5": "c039f778409a15f9b2ceefacad9c2202"}],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
            "train_mask": {
                "upon": [{"filename": "train_mask.pkl", "md5": "81b422937f3adccd89a334d7093b67d7"}],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "val_mask": {
                "upon": [{"filename": "val_mask.pkl", "md5": "10717940ddbfa3e4f6c0b148bb394f79"}],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "test_mask": {
                "upon": [{"filename": "test_mask.pkl", "md5": "10717940ddbfa3e4f6c0b148bb394f79"}],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
        }
