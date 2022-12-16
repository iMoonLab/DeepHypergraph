from typing import Optional

from dhg.datapipe import load_from_pickle

from .base import BaseData


class DBLP8k(BaseData):
    r"""The DBLP-8k dataset is a citation network dataset for link prediction task. 
    The dataset is a part of the dataset crawled according to DBLP API, and we have selected each item based on some conditions, such as the venue and publication year (from 2018 to 2022). It contains 6498 authors and 2603 papers.
    
    The content of the DBLP-8k dataset includes the following:

    - ``num_vertices``: The number of vertices: :math:`8,657`.
    - ``num_edges``: The number of edges: :math:`2,603`.
    - ``edge_list``: The edge list. ``List`` with length :math:`2,603`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to None.
    """
    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("dblp_8k", data_root)
        self._content = {
            "num_vertices": 8657,
            "num_edges": 2603,
            "edge_list": {
                "upon": [{"filename": "edge_list.pkl", "md5": "46b16106daae8eebfd39c2fc43ecbf0b"}],
                "loader": load_from_pickle,
            },
        }

