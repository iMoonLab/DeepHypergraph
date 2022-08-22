from typing import Optional
from functools import partial

from .base import BaseData
from dhg.datapipe import load_from_txt


class Yelp2018(BaseData):
    r"""The Yelp2018 dataset is collected for user-item recommendation task. This dataset is adopted from the 2018 edition of the `Yelp challenge <https://www.yelp.com/collection/AHQG3loQRdpVug_8CmsS_Q>`_. Wherein, the local businesses like restaurants and bars are viewed as the items. 
    
    The Content of the Yelp2018 dataset includes the following:

    - ``num_users``: The number of users: :math:`31,668`.
    - ``num_items``: The number of items: :math:`38,048`.
    - ``num_interactions``: The number of interactions: :math:`1,561,406`.
    - ``train_adj_list``: The train adjacency list.
    - ``test_adj_list``: The test adjacency list.

    .. note::

        The first item of each line in the ``adj_list`` is the user id, and the rest is the item id.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to None.
    """
    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("yelp_2018", data_root)
        self._content = {
            "num_users": 31668,
            "num_items": 38048,
            "num_interactions": 1561406,
            "train_adj_list": {
                "upon": [
                    {
                        "filename": "train.txt",
                        "md5": "1b8b5d22a227e01d6de002c53d32b4c4",
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },
            "test_adj_list": {
                "upon": [
                    {
                        "filename": "test.txt",
                        "md5": "0d57d7399862c32152b045ec5d2698e7",
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },
        }