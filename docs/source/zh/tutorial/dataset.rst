构建输入数据
================

.. hint:: 

    - 作者:  `丰一帆 <https://fengyifan.site/>`_
    - 翻译:  颜杰龙
    - 校对： `丰一帆 <https://fengyifan.site/>`_ 、张欣炜

DHG包含许多可用于训练模型和测试模型的数据集。
在这一节中，将会介绍如何使用DHG的 :doc:`data </api/data>` 模块、构建数据对象的结构和构建自定义数据集的方法及指定预处理步骤。
我们期待您在 `GitHub <https://github.com/iMoonLab/DeepHypergraph>`_ 提交pull request贡献数据集，
请参考 :doc:`贡献教程 </start/contribution>` 的指引。

使用方法
-----------------------

如果您的网络没问题，你可以直接使用 :doc:`/api/data` 内的如下数据集：

.. code-block:: python

    >>> import dhg
    >>> d = dhg.data.Cora()
    >>> d
    cora dataset:
    ->  num_classes
    ->  num_vertices
    ->  num_edges
    ->  dim_features
    ->  features
    ->  edge_list
    ->  labels
    ->  train_mask
    ->  val_mask
    ->  test_mask
    >>> d = dhg.data.Cooking200()
    >>> d
    cooking_200 dataset:
    ->  num_classes
    ->  num_vertices
    ->  num_edges
    ->  edge_list
    ->  labels
    ->  train_mask
    ->  val_mask
    ->  test_mask
    >>> d = dhg.data.MovieLens1M()
    >>> d
    movielens_1m dataset:
    ->  num_users
    ->  num_items
    ->  num_interactions
    ->  train_adj_list
    ->  test_adj_list

或者可以手动从 `DHG's data repository <https://data.deephypergraph.com/>`_ 下载数据集。
然后，您可以把数据集放置在 ``dhg.CAHE_ROOT`` 文件夹或者其它任何目录。
可以使用如下代码获取 ``CACHE_ROOT`` ：

.. code-block:: python

    >>> dhg.CACHE_ROOT
    PosixPath('/home/fengyifan/.dhg')

如果数据集存放在您的指定目录 ``<your-directory>`` ，可以使用如下代码导入数据集：

.. note:: 需要将您下载的数据集的父目录传给参数 ``data_root`` 。

.. code-block:: python

    >>> dhg.data.Cora(data_root=<your-directory>)

一旦您导入了数据集并且获取数据集对象 ``d`` ，可以使用如下代码获取数据集的 **预处理后的数据项**。

.. code-block:: python

    >>> d = dhg.data.Cora()
    >>> # print all available items in the dataset
    >>> d
    cora dataset:
    ->  num_classes
    ->  num_vertices
    ->  num_edges
    ->  dim_features
    ->  features
    ->  edge_list
    ->  labels
    ->  train_mask
    ->  val_mask
    ->  test_mask
    >>> d['num_classes']
    7
    >>> d["edge_list"]
    [(0, 633), (0, 1862), (0, 2582), (1, 2), ..., (2707, 165), (2707, 1473), (2707, 2706)]
    >>> d['features']
    tensor([[0., 0., 0.,  ..., 0., 0., 0.],
            [0., 0., 0.,  ..., 0., 0., 0.],
            [0., 0., 0.,  ..., 0., 0., 0.],
            ...,
            [0., 0., 0.,  ..., 0., 0., 0.],
            [0., 0., 0.,  ..., 0., 0., 0.],
            [0., 0., 0.,  ..., 0., 0., 0.]])
    >>> d['labels']
    >>> d['labels']
    tensor([3, 4, 4,  ..., 3, 3, 3])
    >>> d['train_mask']
    tensor([ True,  True,  True,  ..., False, False, False])

.. code-block:: python

    >>> d = dhg.data.MovieLens1M()
    >>> # print all available items in the dataset
    >>> d
    movielens_1m dataset:
    ->  num_users
    ->  num_items
    ->  num_interactions
    ->  train_adj_list
    ->  test_adj_list
    >>> d['num_users']
    6022
    >>> d['test_adj_list']
    [[0, 2968, 228, 38, 422, 2769], [1, 621, 900, ...], ..., [..., 1579, 3039, 1699, 1195]]

如果需要获取 **未预处理的数据项**，需要调用 :py:meth:`raw() <dhg.data.BaseData.raw>` 方法：

.. code-block:: python

    >>> d = dhg.data.Cora()
    >>> ft = d['features']
    >>> ft.sum(1)
    tensor([1.0000, 1.0000, 1.0000,  ..., 1.0000, 1.0000, 1.0000])
    >>> raw_ft = d.raw('features')
    >>> raw_ft.sum(1)
    matrix([[ 9.],
            [23.],
            [19.],
            ...,
            [18.],
            [14.],
            [13.]], dtype=float32)

Cora数据集内的顶点特征默认使用L1归一化预处理。
下一步，可以参考 :ref:`从边列表构建图 <build_graph>` 教程，来从Cora数据集中构建用于训练的图结构。

模块架构设计
-----------------------
下图展示构建DHG数据集模块的架构设计。

.. image:: ../../_static/img/dataset_arch.jpg
    :align: center
    :alt: dataset_architecture
    :height: 400px

建立自己的数据集
-----------------------

首先，您的数据集类应该继承DHG数据集的基类 :py:class:`BaseData <dhg.data.BaseData>` 。

.. code-block:: python

    >>> from dhg.data import BaseData

数据集中的所有数据项都在 ``_content`` 字典中配置。
同时，支持以下操作：

- 从远程服务器下载 -> 从本地文件加载 -> 预处理并返回
- 从本地文件加载 -> 预处理并返回
- 直接返回

可以在 :ref:`这里 <api_datapipe_loader>` 找到DHG支持的加载函数。

可以在 :ref:`这里 <api_datapipe_preprocess>` 找到DHG支持的预处理函数。

如果数据项 ``item`` 需要从远程服务器下载，您需要在 ``_content`` 字典中指定键 ``upon`` 、 ``loader`` 和 ``preprocess`` 。
键 ``upon`` 为字典列表，每一个字典至少包含 键 ``filename`` 和 ``md5`` 。
``filename`` 为需要下载的文件名， ``md5`` 为文件的md5校验码。
默认情况下，远程文件会存放在 ``REMOTE_DATASETS_ROOT \ data_root \ name \ filename`` 目录。

.. code-block:: python

    self._content = {
        'item': {
            'upon': [
                {'filename': 'part1.pkl', 'md5': '', bk_url: None},
                {'filename': 'part2.pkl', 'md5': '', bk_url: None},
            ],
            'loader': loader_function,
            'preprocess': [datapipe1, datapipe2],
        },
        ...
    }


如果数据项 ``item`` 依赖本地文件，还需要在 ``_content`` 字典中指定键 ``upon`` 、 ``loader`` 和 ``preprocess`` 。
但文件需要放置在 ``data_root \ name \ filename`` 文件夹。
然后， :py:class:`BaseData <dhg.data.BaseData>` 类会自动检查文件的md5校验码。

.. code-block:: python
    
    self._content = {
        'item': {
            'upon': [
                {'filename': 'part1.pkl', 'md5': '', bk_url: None},
                {'filename': 'part2.pkl', 'md5': '', bk_url: None},
            ],
            'loader': loader_function,
            'preprocess': [datapipe1, datapipe2],
        },
        ...
    }

如果数据项 ``item`` 是一个固定的值，您可以直接在 ``_content`` 字典指定 ``value``。

.. code-block:: python
    
    self._content = {
        'item': 666666,
        ...
    }


图数据集示例
++++++++++++++++++++++++++++

.. code-block:: python

    class Cora(BaseData):
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

超图数据集示例
++++++++++++++++++++++++++++++++

.. code-block:: python

    class Cooking200(BaseData):
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


<用户-物品>二分图示例
++++++++++++++++++++++++++++++++++++++++++++

.. code-block:: python

    class MovieLens1M(BaseData):
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

