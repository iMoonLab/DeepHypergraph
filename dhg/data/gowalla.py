from typing import Optional
from functools import partial

from .base import BaseData
from dhg.datapipe import load_from_txt


class Gowalla(BaseData):
    r"""The Gowalla dataset is collected for user-item recommendation task. Locations are viewed as items.
    The full dataset can be found in this `website <https://snap.stanford.edu/data/loc-gowalla.html>`_.
    
    The content of the Gowalla dataset includes the following:

    - ``num_users``: The number of users: :math:`29,858`.
    - ``num_items``: The number of items: :math:`40,981`.
    - ``num_interactions``: The number of interactions: :math:`1,027,370`.
    - ``train_adj_list``: The train adjacency list.
    - ``test_adj_list``: The test adjacency list.

    .. note::

        The first item of each line in the ``adj_list`` is the user id, and the rest is the item id.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to ``None``.
    """
    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("gowalla", data_root)
        self._content = {
            "num_users": 29858,
            "num_items": 40981,
            "num_interactions": 1027370,
            "train_adj_list": {
                "upon": [
                    {
                        "filename": "train.txt",
                        "md5": "5eec1eb2edb8dd648377d348b8e136cf",
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },
            "test_adj_list": {
                "upon": [
                    {
                        "filename": "test.txt",
                        "md5": "c04e2c4bcd2389f53ed8281816166149",
                    }
                ],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },
        }
