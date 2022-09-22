from typing import Optional
from functools import partial

from .base import BaseData
from dhg.datapipe import load_from_pickle, to_tensor, to_long_tensor


class WalmartTrips(BaseData):
    r"""The Walmart Trips dataset is a user-product network dataset for vertex classification task. 
    In Walmart, nodes represent products being purchased at Walmart, 
    and hyperedges equal sets of products purchased together; the node labels are the product categories.
    More details see `this <https://www.cs.cornell.edu/~arb/data/walmart-trips/>`_ and 
    the `YOU ARE ALLSET: A MULTISET LEARNING FRAMEWORK FOR HYPERGRAPH NEURAL NETWORKS <https://openreview.net/pdf?id=hpBTIv2uy_E>`_ paper.
    
    The content of the Walmart Trips dataset includes the following:

    - ``num_classes``: The number of classes: :math:`12`.
    - ``num_vertices``: The number of vertices: :math:`88,860`.
    - ``num_edges``: The number of edges: :math:`69,906`.
    - ``edge_list``: The edge list. ``List`` with length :math:`69,906`.
    - ``labels``: The label list. ``torch.LongTensor`` with size :math:`(88,860, )`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to ``None``.
    """

    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("walmart_trips", data_root)
        self._content = {
            "num_classes": 12,
            "num_vertices": 88860,
            "num_edges": 69906,
            "edge_list": {
                "upon": [{"filename": "edge_list.pkl", "md5": "0827c278282601b9c584f80c3b686a72"}],
                "loader": load_from_pickle,
            },
            "labels": {
                "upon": [{"filename": "labels.pkl", "md5": "981684b84f9e7917e86b5aff08d0c594"}],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
        }

