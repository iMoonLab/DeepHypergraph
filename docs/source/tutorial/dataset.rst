Build Dataset
================

DHG includes a number of datasets that can be used to train and test your models. 
In this section, we will introduce how to use DHG's :doc:`data </api/data>` module, 
the architecture of creating a data object, and how to build your own dataset and specified pre-processing steps.
We welcome to contribute to the dataset by submitting a pull request on `GitHub <https://github.com/iMoonLab/DeepHypergraph>`_, 
please following the :doc:`instruction </start/contribution>` guide.

Usage
-----------------------

If your network is OK, you can directly use any of datasets in :doc:`/api/data` as following:

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

Or you can manually download the dataset from `DHG's data repository <https://data.deephypergraph.com/>`_.
Then, you can put the dataset in the ``dhg.CAHE_ROOT`` directory or any other directory you want.
You can fetch your ``CACHE_ROOT`` by:

.. code-block:: python

    >>> dhg.CACHE_ROOT
    PosixPath('/home/fengyifan/.dhg')

If you put the dataset into the your specified directory ``<your-directory>``, you can use the following code to load the dataset:

.. note:: You should pass the parent directory of your download dataset to the ``data_root`` parameter.

.. code-block:: python

    >>> dhg.data.Cora(data_root=<your-directory>)

As soon as you load the dataset and fetch the data object ``d``, you can use the following code to get **preprocessed** items from the dataset:

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

If you want to get the **un-preprocessed** items you can call the :py:meth:`raw() <dhg.data.BaseData.raw>` method:

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

Defaultly, the vertex featue is pre-processed with L1 normalization in Cora dataset. 
To build a simple graph structucture for training in Cora dataset, you can refer to the :ref:`construct a simple graph from edge list <build_graph>` tutorial.

Architechture
-----------------------
The architecture of constructing DHG's dataset object is shown in the following figure.

.. image:: ../_static/img/dataset_arch.jpg
    :align: center
    :alt: dataset_architecture
    :height: 400px

Build Your Own Dataset
-----------------------
Coming soon...



.. Prepare Dataset
.. -----------------

.. Use the intergrated dataset
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Currently, DHG includes the following datasets:



.. Dataset Pipeline
.. ------------------

.. How to process the data

.. Available Pipeline Functions
.. -----------------------------

.. to_tensor

.. Introduction
.. ------------------------
.. For each dataset in DHG, we have pre-process the feature. and transform them to torch.Tensor.

.. You can access the raw data by data.raw('attribute_name')


.. Examples
.. --------------
