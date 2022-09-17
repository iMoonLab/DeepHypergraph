from typing import Optional
from functools import partial

from dhg.datapipe import load_from_pickle, norm_ft, to_tensor, to_long_tensor, to_bool_tensor

from .base import BaseData


class Flickr(BaseData):
    r"""The Flickr dataset is a social network dataset for vertex classification task. 
    It is a social network where nodes represent users and edges correspond to friendships among users. 
    The labels represent the interest groups of the users.
    
    .. note:: 
        The L1-normalization for the feature is not recommended for this dataset.

    The content of the Flickr dataset includes the following:

    - ``num_classes``: The number of classes: :math:`9`.
    - ``num_vertices``: The number of vertices: :math:`7,575`.
    - ``num_edges``: The number of edges: :math:`479,476`.
    - ``dim_features``: The dimension of features: :math:`12,047`.
    - ``features``: The vertex feature matrix. ``torch.Tensor`` with size :math:`(7,575 \times 12,047)`.
    - ``edge_list``: The edge list. ``List`` with length :math:`(479,476 \times 2)`.
    - ``labels``: The label list. ``torch.LongTensor`` with size :math:`(7,575, )`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to ``None``.
    """

    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("Flickr", data_root)
        self._content = {
            "num_classes": 9,
            "num_vertices": 7575,
            "num_edges": 239738,
            "dim_features": 12047,
            "features": {
                "upon": [{"filename": "features.pkl", "md5": "8e889c8532a91ddcb29d6a9c377b5528"}],
                "loader": load_from_pickle,
                "preprocess": [to_tensor],  # partial(norm_ft, ord=1)
            },
            "edge_list": {
                "upon": [{"filename": "edge_list.pkl", "md5": "ea7412a30539fbc95f76ee3712a07017"}],
                "loader": load_from_pickle,
            },
            "labels": {
                "upon": [{"filename": "labels.pkl", "md5": "9603c29e31b863a34fc707b606c02880"}],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
        }
