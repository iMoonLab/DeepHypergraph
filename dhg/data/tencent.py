from typing import Optional
from functools import partial

from dhg.datapipe import load_from_pickle, norm_ft, to_tensor, to_long_tensor, to_bool_tensor

from .base import BaseData


class TencentBiGraph(BaseData):
    r"""The TencentBiGraph dataset is a social network dataset for vertex classification task. 
    This is a large-scale real-world social network represented by a bipartite graph. 
    Nodes in set :math:`U` are social network users, and nodes in set :math:`V` are social communities 
    (e.g., a subset of social network users who share the same interests in electrical products may 
    join the same shopping community). Both users and communities are described by dense off-the-shelf feature vectors. 
    The edge connection between two sets indicates that the user belongs to the community. 
    Note that this dataset provides classification labels for research purposes. 
    In real-world applications, labeling every node is impractical.
    More details see the `Cascade-BGNN: Toward Efficient Self-supervised Representation Learning on Large-scale Bipartite Graphs <https://arxiv.org/pdf/1906.11994.pdf>`_ paper.

    The content of the TencentBiGraph dataset includes the following:

    - ``num_u_classes``: The number of classes in set :math:`U` : :math:`2`.
    - ``num_u_vertices``: The number of vertices in set :math:`U` : :math:`619,030`.
    - ``num_v_vertices``: The number of vertices in set :math:`V` : :math:`90,044`.
    - ``num_edges``: The number of edges: :math:`144,501`.
    - ``dim_u_features``: The dimension of features in set :math:`U` : :math:`8`.
    - ``dim_v_features``: The dimension of features in set :math:`V` : :math:`16`.
    - ``u_features``: The vertex feature matrix in set :math:`U`. ``torch.Tensor`` with size :math:`(619,030 \times 8)`.
    - ``v_features``: The vertex feature matrix in set :math:`V` . ``torch.Tensor`` with size :math:`(90,044 \times 16)`.
    - ``edge_list``: The edge list. ``List`` with length :math:`(991,713 \times 2)`.
    - ``u_labels``: The label list in set :math:`U` . ``torch.LongTensor`` with size :math:`(619,030, )`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to ``None``.
    """

    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__("tencent_bigraph", data_root)
        self._content = {
            "num_u_classes": 2,
            "num_u_vertices": 619030,
            "num_v_vertices": 90044,
            "num_edges": 991713,
            "dim_u_features": 8,
            "dim_v_features": 16,
            "u_features": {
                "upon": [{"filename": "u_features.pkl", "md5": "f6984204d7c9953894678c16670fc7a6"}],
                "loader": load_from_pickle,
                "preprocess": [to_tensor, partial(norm_ft, ord=1)],
            },
            "v_features": {
                "upon": [{"filename": "v_features.pkl", "md5": "5344dadb0b7274c26fc60ddc333e2939"}],
                "loader": load_from_pickle,
                "preprocess": [to_tensor, partial(norm_ft, ord=1)],
            },
            "edge_list": {
                "upon": [{"filename": "edge_list.pkl", "md5": "f5f8a8478133e1c25c01aa09b7eb96a2"}],
                "loader": load_from_pickle,
            },
            "u_labels": {
                "upon": [{"filename": "u_labels.pkl", "md5": "1e4a226acb8b50589ed59623069d3887"}],
                "loader": load_from_pickle,
                "preprocess": [to_long_tensor],
            },
        }
