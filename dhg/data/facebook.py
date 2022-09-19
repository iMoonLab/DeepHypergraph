from typing import Optional
from functools import partial

from dhg.datapipe import load_from_pickle, norm_ft, to_tensor, to_long_tensor, to_bool_tensor

from .base import BaseData


class Facebook(BaseData):
    r"""The Facebook dataset is a social network dataset for vertex classification task. 
    A page-page graph of verified Facebook sites. Nodes correspond to official Facebook pages, links to mutual likes between sites. 
    Node features are extracted from the site descriptions. 
    More details see the `Multi-Scale Attributed Node Embedding <https://arxiv.org/pdf/1909.13021.pdf>`_ paper.
    
    .. note:: 
        The L1-normalization for the feature is not recommended for this dataset.

    The content of the Facebook dataset includes the following:

    - ``num_classes``: The number of classes: :math:`4`.
    - ``num_vertices``: The number of vertices: :math:`22,470`.
    - ``num_edges``: The number of edges: :math:`85,501`.
    - ``dim_features``: The dimension of features: :math:`4,714`.
    - ``features``: The vertex feature matrix. ``torch.Tensor`` with size :math:`(22,470\times 4,714)`.
    - ``edge_list``: The edge list. ``List`` with length :math:`(85,501 \times 2)`.
    - ``labels``: The label list. ``torch.LongTensor`` with size :math:`(22,470, )`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to ``None``.
    """

    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("facebook", data_root)
        self._content = {
            "num_classes": 4,
            "num_vertices": 22470,
            "num_edges": 85501,
            "dim_features": 8189,
            "features": {
                "upon": [{"filename": "features.pkl", "md5": "046eec1b67fb5bf504eaad75e98af141"}],
                "loader": load_from_pickle,
                "preprocess": [to_tensor],  # partial(norm_ft, ord=1)
            },
            "edge_list": {
                "upon": [{"filename": "edge_list.pkl", "md5": "98c6551d020c7741554cae5eab8336ef"}],
                "loader": load_from_pickle,
            },
            "labels": {
                "upon": [{"filename": "labels.pkl", "md5": "ae0c116274cedc00522df66bd921affc"}],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
        }
