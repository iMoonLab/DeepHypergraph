from typing import Optional
from functools import partial

from dhg.datapipe import load_from_pickle, norm_ft, to_tensor, to_long_tensor, to_bool_tensor

from .base import BaseData


class CoauthorshipCora(BaseData):
    r"""The Co-authorship Cora dataset is a citation network dataset for vertex classification task.
    More details see the `HyperGCN <https://papers.nips.cc/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf>`_ paper.
    
    The content of the Co-authorship Cora dataset includes the following:

    - ``num_classes``: The number of classes: :math:`7`.
    - ``num_vertices``: The number of vertices: :math:`2,708`.
    - ``num_edges``: The number of edges: :math:`1,072`.
    - ``dim_features``: The dimension of features: :math:`1,433`.
    - ``features``: The vertex feature matrix. ``torch.Tensor`` with size :math:`(2,708 \times 1,433)`.
    - ``edge_list``: The edge list. ``List`` with length :math:`1,072`.
    - ``labels``: The label list. ``torch.LongTensor`` with size :math:`(2,708, )`.
    - ``train_mask``: The train mask. ``torch.BoolTensor`` with size :math:`(2,708, )`.
    - ``val_mask``: The validation mask. ``torch.BoolTensor`` with size :math:`(2,708, )`.
    - ``test_mask``: The test mask. ``torch.BoolTensor`` with size :math:`(2,708, )`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to ``None``.
    """
    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("coauthorship_cora", data_root)
        self._content = {
            "num_classes": 7,
            "num_vertices": 2708,
            "num_edges": 1072,
            "dim_features": 1433,
            "features": {
                "upon": [{"filename": "features.pkl", "md5": "14257c0e24b4eb741b469a351e524785"}],
                "loader": load_from_pickle,
                "preprocess": [to_tensor, partial(norm_ft, ord=1)],
            },
            "edge_list": {
                "upon": [{"filename": "edge_list.pkl", "md5": "a17ff337f1b9099f5a9d4d670674e146"}],
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


class CoauthorshipDBLP(BaseData):
    r"""The Co-authorship DBLP dataset is a citation network dataset for vertex classification task.
    More details see the `HyperGCN <https://papers.nips.cc/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf>`_ paper.
    
    The content of the Co-authorship DBLP dataset includes the following:

    - ``num_classes``: The number of classes: :math:`6`.
    - ``num_vertices``: The number of vertices: :math:`41,302`.
    - ``num_edges``: The number of edges: :math:`22,363`.
    - ``dim_features``: The dimension of features: :math:`1,425`.
    - ``features``: The vertex feature matrix. ``torch.Tensor`` with size :math:`(41,302 \times 1,425)`.
    - ``edge_list``: The edge list. ``List`` with length :math:`22,363`.
    - ``labels``: The label list. ``torch.LongTensor`` with size :math:`(41,302, )`.
    - ``train_mask``: The train mask. ``torch.BoolTensor`` with size :math:`(41,302, )`.
    - ``val_mask``: The validation mask. ``torch.BoolTensor`` with size :math:`(41,302, )`.
    - ``test_mask``: The test mask. ``torch.BoolTensor`` with size :math:`(41,302, )`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to None.
    """
    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("coauthorship_dblp", data_root)
        self._content = {
            "num_classes": 6,
            "num_vertices": 41302,
            "num_edges": 22363,
            "dim_features": 1425,
            "features": {
                "upon": [{"filename": "features.pkl", "md5": "b78fd31b2586d1e19a40b3f6cd9cc2e7"}],
                "loader": load_from_pickle,
                "preprocess": [to_tensor, partial(norm_ft, ord=1)],
            },
            "edge_list": {
                "upon": [{"filename": "edge_list.pkl", "md5": "c6bf5f9f3b9683bcc9b7bcc9eb8707d8"}],
                "loader": load_from_pickle,
            },
            "labels": {
                "upon": [{"filename": "labels.pkl", "md5": "2e7a792ea018028d582af8f02f2058ca"}],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
            "train_mask": {
                "upon": [{"filename": "train_mask.pkl", "md5": "a842b795c7cac4c2f98a56cf599bc1de"}],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "val_mask": {
                "upon": [{"filename": "val_mask.pkl", "md5": "2ec4b7df7c5e6b355067a22c391ad578"}],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "test_mask": {
                "upon": [{"filename": "test_mask.pkl", "md5": "2ec4b7df7c5e6b355067a22c391ad578"}],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
        }

