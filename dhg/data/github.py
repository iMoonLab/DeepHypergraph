from typing import Optional
from functools import partial

from dhg.datapipe import load_from_pickle, norm_ft, to_tensor, to_long_tensor, to_bool_tensor

from .base import BaseData


class Github(BaseData):
    r"""The Github dataset is a collaboration network dataset for vertex classification task. 
    Nodes correspond to developers who have starred at least 10 repositories and edges to mutual follower relationships. 
    Node features are location, starred repositories, employer and e-mail address. 
    The labels are binary, where denoting the web developers and machine learning developers.
    More details see the `Multi-Scale Attributed Node Embedding <https://arxiv.org/pdf/1909.13021.pdf>`_ paper.
    
    .. note:: 
        The L1-normalization for the feature is not recommended for this dataset.

    The content of the Github dataset includes the following:

    - ``num_classes``: The number of classes: :math:`4`.
    - ``num_vertices``: The number of vertices: :math:`37,700`.
    - ``num_edges``: The number of edges: :math:`144,501`.
    - ``dim_features``: The dimension of features: :math:`4,005`.
    - ``features``: The vertex feature matrix. ``torch.Tensor`` with size :math:`(37,700 \times 4,005)`.
    - ``edge_list``: The edge list. ``List`` with length :math:`(144,501 \times 2)`.
    - ``labels``: The label list. ``torch.LongTensor`` with size :math:`(37,700, )`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to ``None``.
    """

    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("github", data_root)
        self._content = {
            "num_classes": 2,
            "num_vertices": 37700,
            "num_edges": 144501,
            "dim_features": 4005,
            "features": {
                "upon": [{"filename": "features.pkl", "md5": "f097384b61876a22cf048d28a2193c5a"}],
                "loader": load_from_pickle,
                "preprocess": [to_tensor],  # partial(norm_ft, ord=1)
            },
            "edge_list": {
                "upon": [{"filename": "edge_list.pkl", "md5": "57012ac55fe125d8865a693b09f794b3"}],
                "loader": load_from_pickle,
            },
            "labels": {
                "upon": [{"filename": "labels.pkl", "md5": "9b1282a2a8a23c9f3b480136055c8b6b"}],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
        }
