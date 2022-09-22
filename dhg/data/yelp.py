from typing import Optional
from functools import partial

from .base import BaseData
from dhg.datapipe import load_from_txt, load_from_pickle, to_tensor, to_long_tensor


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
                "upon": [{"filename": "train.txt", "md5": "1b8b5d22a227e01d6de002c53d32b4c4",}],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },
            "test_adj_list": {
                "upon": [{"filename": "test.txt", "md5": "0d57d7399862c32152b045ec5d2698e7",}],
                "loader": partial(load_from_txt, dtype="int", sep=" "),
            },
        }


class YelpRestaurant(BaseData):
    r"""The Yelp-Restaurant dataset is a restaurant-review network dataset for vertex classification task. 
    All businesses in the “restaurant” catalog are selected as our nodes, 
    and formed hyperedges by selecting restaurants visited by the same user. 
    We use the number of stars in the average review of a restaurant as the corresponding node label, 
    starting from 1 and going up to 5 stars, with an interval of 0.5 stars. 
    We then form the node features from the latitude, longitude, one-hot encoding of city and state, 
    and bag-of-word encoding of the top-1000 words in the name of the corresponding restaurants.
    More details see the `YOU ARE ALLSET: A MULTISET LEARNING FRAMEWORK FOR HYPERGRAPH NEURAL NETWORKS <https://openreview.net/pdf?id=hpBTIv2uy_E>`_ paper.
    
    The content of the Yelp-Restaurant dataset includes the following:

    - ``num_classes``: The number of classes: :math:`11`.
    - ``num_vertices``: The number of vertices: :math:`50,758`.
    - ``num_edges``: The number of edges: :math:`679,302`.
    - ``dim_features``: The dimension of features: :math:`1,862`.
    - ``features``: The vertex feature matrix. ``torch.Tensor`` with size :math:`(50,758 \times 1,862)`.
    - ``edge_list``: The edge list. ``List`` with length :math:`679,302`.
    - ``labels``: The label list. ``torch.LongTensor`` with size :math:`(50,758, )`.
    - ``state``: The state list. ``torch.LongTensor`` with size :math:`(50,758, )`.
    - ``city``: The city list. ``torch.LongTensor`` with size :math:`(50,758, )`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to ``None``.
    """

    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("yelp_restaurant", data_root)
        self._content = {
            "num_classes": 11,
            "num_vertices": 50758,
            "num_edges": 679302,
            "dim_features": 1862,
            "features": {
                "upon": [{"filename": "features.pkl", "md5": "cedc4443884477c2e626025411c44cd7"}],
                "loader": load_from_pickle,
                "preprocess": [to_tensor,],
            },
            "edge_list": {
                "upon": [{"filename": "edge_list.pkl", "md5": "4b26eecaa22305dd10edcd6372eb49da"}],
                "loader": load_from_pickle,
            },
            "labels": {
                "upon": [{"filename": "labels.pkl", "md5": "1cdc1ed9fb1f57b2accaa42db214d4ef"}],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
            "state": {
                "upon": [{"filename": "state.pkl", "md5": "eef3b835fad37409f29ad36539296b57"}],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
            "city": {
                "upon": [{"filename": "city.pkl", "md5": "8302b167262b23067698e865cacd0b17"}],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
        }

