from typing import Optional
from functools import partial

from dhg.datapipe import load_from_pickle, norm_ft, to_tensor, to_long_tensor, to_bool_tensor

from .base import BaseData


class Cora(BaseData):
    r"""The Cora dataset is a citation network dataset for vertex classification task. More details can be found in this `websit <https://relational.fit.cvut.cz/dataset/CORA>`_.
    
    The content of the Cora dataset includes the following:

    - ``num_classes``: The number of classes: :math:`7`.
    - ``num_vertices``: The number of vertices: :math:`2,708`.
    - ``num_edges``: The number of edges: :math:`10,858`.
    - ``dim_features``: The dimension of features: :math:`1,433`.
    - ``features``: The vertex feature matrix. ``torch.Tensor`` with size :math:`(2,708 \times 1,433)`.
    - ``edge_list``: The edge list. ``List`` with length :math:`(10,858 \times 2)`.
    - ``labels``: The label list. ``torch.LongTensor`` with size :math:`(2,708, )`.
    - ``train_mask``: The train mask. ``torch.BoolTensor`` with size :math:`(2,708, )`.
    - ``val_mask``: The validation mask. ``torch.BoolTensor`` with size :math:`(2,708, )`.
    - ``test_mask``: The test mask. ``torch.BoolTensor`` with size :math:`(2,708, )`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to None.
    """
    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__('cora', data_root)
        self._content = {
            "num_classes": 7,
            "num_vertices": 2708,
            "num_edges": 10858,
            "dim_features": 1433,
            'features': {
                'upon': [{ 'filename': 'features.pkl', 'md5': '05b45e9c38cc95f4fc44b3668cc9ddc9' }],
                'loader': load_from_pickle,
                'preprocess': [to_tensor, partial(norm_ft, ord=1)],
            },
            'edge_list': {
                'upon': [{ 'filename': 'edge_list.pkl', 'md5': 'f488389c1edd0d898ce273fbd27822b3' }],
                'loader': load_from_pickle,
            },
            'labels': {
                'upon': [{ 'filename': 'labels.pkl', 'md5': 'e506014762052c6a36cb583c28bdae1d' }],
                'loader': load_from_pickle,
                'preprocess': [to_long_tensor],
            },
            'train_mask': {
                'upon': [{ 'filename': 'train_mask.pkl', 'md5': 'a11357a40e1f0b5cce728d1a961b8e13' }],
                'loader': load_from_pickle,
                'preprocess': [to_bool_tensor],
            },
            'val_mask': {
                'upon': [{ 'filename': 'val_mask.pkl', 'md5': '355544da566452601bcfa74d30539a71' }],
                'loader': load_from_pickle,
                'preprocess': [to_bool_tensor],
            },
            'test_mask': {
                'upon': [{ 'filename': 'test_mask.pkl', 'md5': 'bbfc87d661560f55f6946f8cb9d602b9' }],
                'loader': load_from_pickle,
                'preprocess': [to_bool_tensor],
            },
        }


class Citeseer(BaseData):
    r"""The Citeseer dataset is a citation network dataset for vertex classification task. More details can be found in this `websit <https://relational.fit.cvut.cz/dataset/CiteSeer>`_.

    - ``num_classes``: The number of classes: :math:`6`.
    - ``num_vertices``: The number of vertices: :math:`3,327`.
    - ``num_edges``: The number of edges: :math:`9,464`.
    - ``dim_features``: The dimension of features: :math:`3,703`.
    - ``features``: The vertex feature matrix. ``torch.Tensor`` with size :math:`(3,327 \times 3,703)`.
    - ``edge_list``: The edge list. ``List`` with length :math:`(9,464 \times 2)`.
    - ``labels``: The label list. ``torch.LongTensor`` with size :math:`(3,327, )`.
    - ``train_mask``: The train mask. ``torch.BoolTensor`` with size :math:`(3,327, )`.
    - ``val_mask``: The validation mask. ``torch.BoolTensor`` with size :math:`(3,327, )`.
    - ``test_mask``: The test mask. ``torch.BoolTensor`` with size :math:`(3,327, )`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to None.
    """
    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__('citeseer', data_root)
        self._content = {
            "num_classes": 6,
            "num_vertices": 3327,
            "num_edges": 9464,
            "dim_features": 3703,
            'features': {
                'upon': [{ 'filename': 'features.pkl', 'md5': '7458c683e584a0c5ce1ab7af763777c6' }],
                'loader': load_from_pickle,
                'preprocess': [to_tensor, partial(norm_ft, ord=1)],
            },
            'edge_list': {
                'upon': [{ 'filename': 'edge_list.pkl', 'md5': '1948e9f712bc16ba8ef48a3e79fc2246' }],
                'loader': load_from_pickle,
            },
            'labels': {
                'upon': [{ 'filename': 'labels.pkl', 'md5': 'f5bcf7815e463af4f88d40195f0d378c' }],
                'loader': load_from_pickle,
                'preprocess': [to_long_tensor],
            },
            'train_mask': {
                'upon': [{ 'filename': 'train_mask.pkl', 'md5': '9aae62b41403b976c4cc048685c966e6' }],
                'loader': load_from_pickle,
                'preprocess': [to_bool_tensor],
            },
            'val_mask': {
                'upon': [{ 'filename': 'val_mask.pkl', 'md5': '4527d7dc1e2604cdaa9e18916f32714b' }],
                'loader': load_from_pickle,
                'preprocess': [to_bool_tensor],
            },
            'test_mask': {
                'upon': [{ 'filename': 'test_mask.pkl', 'md5': 'af49e6f6f53c73b7d3a62d6f9b2a3871' }],
                'loader': load_from_pickle,
                'preprocess': [to_bool_tensor],
            },
        }


class Pubmed(BaseData):
    r"""The PubMed dataset is a citation network dataset for vertex classification task. More details can be found in this `websit <https://pubmed.ncbi.nlm.nih.gov/download/>`_.

    - ``num_classes``: The number of classes: :math:`3`.
    - ``num_vertices``: The number of vertices: :math:`19,717`.
    - ``num_edges``: The number of edges: :math:`88,676`.
    - ``dim_features``: The dimension of features: :math:`500`.
    - ``features``: The vertex feature matrix. ``torch.Tensor`` with size :math:`(19,717 \times 500)`.
    - ``edge_list``: The edge list. ``List`` with length :math:`(88,676 \times 2)`.
    - ``labels``: The label list. ``torch.LongTensor`` with size :math:`(19,717, )`.
    - ``train_mask``: The train mask. ``torch.BoolTensor`` with size :math:`(19,717, )`.
    - ``val_mask``: The validation mask. ``torch.BoolTensor`` with size :math:`(19,717, )`.
    - ``test_mask``: The test mask. ``torch.BoolTensor`` with size :math:`(19,717, )`.

    Args:
        ``data_root`` (``str``, optional): The ``data_root`` has stored the data. If set to ``None``, this function will auto-download from server and save into the default direction ``~/.dhg/datasets/``. Defaults to None.
    """
    def __init__(self, data_root: Optional[str] = None) -> None:
        super().__init__('pubmed', data_root)
        self._content = {
            "num_classes": 3,
            "num_vertices": 19717,
            "num_edges": 88676,
            "dim_features": 500,
            'features': {
                'upon': [{ 'filename': 'features.pkl', 'md5': 'b8fe6c0ce974d031c9c70266f4ccff44' }],
                'loader': load_from_pickle,
                'preprocess': [to_tensor, partial(norm_ft, ord=1)],
            },
            'edge_list': {
                'upon': [{ 'filename': 'edge_list.pkl', 'md5': '9563ff5fc66e56ab53ccb25685e6d540' }],
                'loader': load_from_pickle,
            },
            'labels': {
                'upon': [{ 'filename': 'labels.pkl', 'md5': '6132b80c5cea4e73f45920779175e3f8' }],
                'loader': load_from_pickle,
                'preprocess': [to_long_tensor],
            },
            'train_mask': {
                'upon': [{ 'filename': 'train_mask.pkl', 'md5': '69d4ef4d7cdb53ff4b3b48ce394363b0' }],
                'loader': load_from_pickle,
                'preprocess': [to_bool_tensor],
            },
            'val_mask': {
                'upon': [{ 'filename': 'val_mask.pkl', 'md5': '5a65a2ad27165dd0cea2675592ee414e' }],
                'loader': load_from_pickle,
                'preprocess': [to_bool_tensor],
            },
            'test_mask': {
                'upon': [{ 'filename': 'test_mask.pkl', 'md5': '4bbf50754d7fdae2b5c6c12d85ccc3a5' }],
                'loader': load_from_pickle,
                'preprocess': [to_bool_tensor],
            },
        }
