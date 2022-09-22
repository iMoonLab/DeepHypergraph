from typing import Optional
from functools import partial

from .base import BaseData
from dhg.datapipe import load_from_pickle, to_tensor, to_long_tensor


class HouseCommittees(BaseData):
    r"""The House Committees dataset is a committee network dataset for vertex classification task. 
    In the House dataset, each node is a member of the US House of Representatives and 
    hyperedges are formed by grouping together members of the same committee. Node labels indicate the political party of the representatives.
    More details see `this <https://www.cs.cornell.edu/~arb/data/house-committees/>`_ and 
    the `YOU ARE ALLSET: A MULTISET LEARNING FRAMEWORK FOR HYPERGRAPH NEURAL NETWORKS <https://openreview.net/pdf?id=hpBTIv2uy_E>`_ paper.
    
    The content of the House Committees dataset includes the following:

    - ``num_classes``: The number of classes: :math:`3`.
    - ``num_vertices``: The number of vertices: :math:`1,290`.
    - ``num_edges``: The number of edges: :math:`341`.
    - ``edge_list``: The edge list. ``List`` with length :math:`341`.
    - ``labels``: The label list. ``torch.LongTensor`` with size :math:`(1,290, )`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to ``None``.
    """

    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("house_committees", data_root)
        self._content = {
            "num_classes": 3,
            "num_vertices": 1290,
            "num_edges": 341,
            "edge_list": {
                "upon": [{"filename": "edge_list.pkl", "md5": "8f054ddfc7ba4e1e80010418884c77f7"}],
                "loader": load_from_pickle,
            },
            "labels": {
                "upon": [{"filename": "labels.pkl", "md5": "9084625ef807e61133bdafdf3d8c8c93"}],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
        }

