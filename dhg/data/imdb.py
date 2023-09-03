from typing import Optional
from functools import partial

from dhg.datapipe import load_from_pickle, norm_ft, to_bool_tensor, to_tensor, to_long_tensor

from .base import BaseData


class IMDB4k(BaseData):
    r"""The IMDB-4k dataset is a movie dataset for node classification task. 
    The dataset is an online database about movies and television programs, including information such as cast, production crew, and plot summaries. 
    This is a subset of IMDB scraped from online, containing 4278 movies, 2081 directors, and 5257 actors after data preprocessing. 
    Movies are labeled as one of three classes (Action, Comedy, and Drama) based on their genre information. 
    Each movie is also described by a bag-of-words representation of its plot keywords. 
    The vertice denotes author, and two types of correlation (co-director, co-actor) can be used for building hyperedges.
    More details see the `MAGNN: Metapath Aggregated Graph Neural Network for Heterogeneous Graph Embedding <https://arxiv.org/pdf/2002.01680.pdf>`_ paper.
    
    The content of the IMDB-4k dataset includes the following:

    - ``num_classes``: The number of classes: :math:`3`.
    - ``num_vertices``: The number of vertices: :math:`4,278`.
    - ``num_director_edges``: The number of hyperedges constructed by the co-director correlation: :math:`2,081`.
    - ``num_actor_edges``: The number of hyperedges constructed by the co-actor correlation: :math:`5,257`.
    - ``dim_features``: The dimension of movie features: :math:`3,066`.
    - ``features``: The movie feature matrix. ``torch.Tensor`` with size :math:`(4,278 \times 3,066)`.
    - ``labels``: The label list. ``torch.LongTensor`` with size :math:`(4,278, )`.
    - ``edge_by_director``: The hyperedge list constructed by the co-director correlation. ``List`` with length :math:`(2,081)`.
    - ``edge_by_actor``: The hyperedge list constructed by the co-actor correlation. ``List`` with length :math:`(5,257)`.
    
    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to None.
    """

    def __init__(self, data_root: Optional[str] = None):
        super().__init__("imdb_4k", data_root)
        self._content = {
            'num_classes': 3,
            'num_vertices': 4278,
            'num_director_edges': 2081,
            'num_actor_edges': 5257,
            'dim_features': 3066,
            "features": {
                "upon": [{"filename": "features.pkl", "md5": "b9cca982d3d5066ddb2013951939c070"}],
                "loader": load_from_pickle,
                "preprocess": [to_tensor, partial(norm_ft, ord=1)],
            },
            'labels': {
                'upon': [{'filename': 'labels.pkl', 'md5': 'a45e5af53d5475ac87f5d8aa779b3a20'}],
                'loader': load_from_pickle,
                'preprocess': [to_long_tensor]
            },
            'edge_by_director': {
                'upon': [{'filename': 'edge_by_director.pkl', 'md5': '671b7c2010e8604f037523738323cd78'}],
                'loader': load_from_pickle,
            },
            'edge_by_actor': {
                'upon': [{'filename': 'edge_by_actor.pkl', 'md5': 'dff7557861445de77b05d6215746c9f1'}],
                'loader': load_from_pickle,
            },
        }
