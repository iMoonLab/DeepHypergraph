from typing import Optional
from functools import partial

from .base import BaseData
from dhg.datapipe import load_from_txt


class MovieLens1M(BaseData):
    r"""The MovieLens1M dataset is collected for user-item recommendation task. Wherein, movies are viewed as items.
    `Released 2/2003 <https://grouplens.org/datasets/movielens/1m/>`_. The dataset contains 1 million ratings from 6022 users on 3043 items.
    
    The content of the MovieLens-1M dataset includes the following:

    - ``num_users``: The number of users: :math:`6,022`.
    - ``num_items``: The number of items: :math:`3,043`.
    - ``num_interactions``: The number of interactions: :math:`995,154`.
    - ``train_adj_list``: The train adjacency list.
    - ``test_adj_list``: The test adjacency list.

    .. note::

        The first item of each line in the ``adj_list`` is the user id, and the rest is the item id.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to ``None``.
    """
    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("movielens_1m", data_root)
        self._content = {
            "num_users": 6022,
            "num_items": 3043,
            "num_interactions": 995154,
            "train_adj_list": {
                "upon": [
                    {
                        "filename": "train.txt",
                        "md5": "db93f671bc5d1b1544ce4c29664f6778",
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },
            "test_adj_list": {
                "upon": [
                    {
                        "filename": "test.txt",
                        "md5": "5e55bcbb6372ad4c6fafe79989e2f956",
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },
        }
