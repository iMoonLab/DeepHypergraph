from typing import Optional
from functools import partial

from .base import BaseData
from dhg.datapipe import load_from_txt


class AmazonBook(BaseData):
    r"""The AmazonBook dataset is collected for user-item recommendation task. This dataset is a subset of `Amazon-review <https://jmcauley.ucsd.edu/data/amazon/>`_. Wherein, books are viewed as the items.
    
    The content of the Amazon-Book dataset includes the following:

    - ``num_users``: The number of users: :math:`52,643`.
    - ``num_items``: The number of items: :math:`91,599`.
    - ``num_interactions``: The number of interactions: :math:`2,984,108`.
    - ``train_adj_list``: The train adjacency list.
    - ``test_adj_list``: The test adjacency list.

    .. note::

        The first item of each line in the ``adj_list`` is the user id, and the rest is the item id.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to ``None``.
    """

    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("amazon_book", data_root)
        self._content = {
            "num_users": 52643,
            "num_items": 91599,
            "num_interactions": 2984108,
            "train_adj_list": {
                "upon": [{"filename": "train.txt", "md5": "c916ecac04ca72300a016228258b41ed",}],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },
            "test_adj_list": {
                "upon": [{"filename": "test.txt", "md5": "30f8ccfea18d25007ba9fb9aba4e174d",}],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },
        }
