from typing import Optional
from functools import partial

from dhg.datapipe import load_from_pickle, norm_ft, to_tensor, to_long_tensor, to_bool_tensor

from .base import BaseData


class News20(BaseData):
    r"""The 20 Newsgroups dataset is a newspaper network dataset for vertex classification task. 
    The node features are the TF-IDF representations of news messages.
    More details see the `YOU ARE ALLSET: A MULTISET LEARNING FRAMEWORK FOR HYPERGRAPH NEURAL NETWORKS <https://openreview.net/pdf?id=hpBTIv2uy_E>`_ paper.
    
    The content of the 20 Newsgroups dataset includes the following:

    - ``num_classes``: The number of classes: :math:`4`.
    - ``num_vertices``: The number of vertices: :math:`16,342`.
    - ``num_edges``: The number of edges: :math:`100`.
    - ``dim_features``: The dimension of features: :math:`1,433`.
    - ``features``: The vertex feature matrix. ``torch.Tensor`` with size :math:`(16,342 \times 100)`.
    - ``edge_list``: The edge list. ``List`` with length :math:`100`.
    - ``labels``: The label list. ``torch.LongTensor`` with size :math:`(16,342, )`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to ``None``.
    """

    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("20news", data_root)
        self._content = {
            "num_classes": 4,
            "num_vertices": 16342,
            "num_edges": 100,
            "dim_features": 100,
            "features": {
                "upon": [{"filename": "features.pkl", "md5": "3ccc2220867a13e7477791e9bb732d47"}],
                "loader": load_from_pickle,
                "preprocess": [to_tensor,],
            },
            "edge_list": {
                "upon": [{"filename": "edge_list.pkl", "md5": "b49d5486e08da01f2cbe3419489597ff"}],
                "loader": load_from_pickle,
            },
            "labels": {
                "upon": [{"filename": "labels.pkl", "md5": "66d15dee0ed42ab88fa203c83af02d80"}],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
        }

