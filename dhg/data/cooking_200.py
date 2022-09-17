from typing import Optional

from .base import BaseData
from dhg.datapipe import (
    load_from_pickle,
    to_bool_tensor,
    to_long_tensor,
)


class Cooking200(BaseData):
    r"""The Cooking 200 dataset is collected from `Yummly.com <https://www.yummly.com/>`_ for vertex classification task. 
    It is a hypergraph dataset, in which vertex denotes the dish and hyperedge denotes
    the ingredient. Each dish is also associated with category information, which indicates the dish's cuisine like 
    Chinese, Japanese, French, and Russian.
    
    The content of the Cooking200 dataset includes the following:

    - ``num_classes``: The number of classes: :math:`20`.
    - ``num_vertices``: The number of vertices: :math:`7,403`.
    - ``num_edges``: The number of edges: :math:`2,755`.
    - ``edge_list``: The edge list. ``List`` with length :math:`(2,755)`.
    - ``labels``: The label list. ``torch.LongTensor`` with size :math:`(7,403)`.
    - ``train_mask``: The train mask. ``torch.BoolTensor`` with size :math:`(7,403)`.
    - ``val_mask``: The validation mask. ``torch.BoolTensor`` with size :math:`(7,403)`.
    - ``test_mask``: The test mask. ``torch.BoolTensor`` with size :math:`(7,403)`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to ``None``.
    """

    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("cooking_200", data_root)
        self._content = {
            "num_classes": 20,
            "num_vertices": 7403,
            "num_edges": 2755,
            "edge_list": {
                "upon": [
                    {
                        "filename": "edge_list.pkl",
                        "md5": "2cd32e13dd4e33576c43936542975220",
                    }
                ],
                "loader": load_from_pickle,
            },
            "labels": {
                "upon": [
                    {
                        "filename": "labels.pkl",
                        "md5": "f1f3c0399c9c28547088f44e0bfd5c81",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
            "train_mask": {
                "upon": [
                    {
                        "filename": "train_mask.pkl",
                        "md5": "66ea36bae024aaaed289e1998fe894bd",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "val_mask": {
                "upon": [
                    {
                        "filename": "val_mask.pkl",
                        "md5": "6c0d3d8b752e3955c64788cc65dcd018",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
            "test_mask": {
                "upon": [
                    {
                        "filename": "test_mask.pkl",
                        "md5": "0e1564904551ba493e1f8a09d103461e",
                    }
                ],
                "loader": load_from_pickle,
                "preprocess": [to_bool_tensor],
            },
        }
