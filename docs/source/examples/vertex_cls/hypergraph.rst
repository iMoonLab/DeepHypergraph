On Simple Hypergraph
==========================================


In the following examples, three typical graph/hypergraph neural network are used to perform vertex classification task on the simple graph structure.

Models
---------------------------

- GCN, `Semi-Supervised Classification with Graph Convolutional Networks <https://arxiv.org/pdf/1609.02907>`_ paper (ICLR 2017).
- HGNN, `Hypergraph Neural Networks <https://arxiv.org/pdf/1809.09401>`_ paper (AAAI 2019).
- HGNN+, `HGNN+: General Hypergraph Neural Networks <https://ieeexplore.ieee.org/document/9795251>`_ paper (IEEE T-PAMI 2022).

Dataset Description
---------------------------

Cooking 200 dataset (:py:class:`dhg.data.Cooking200`) is a cooking recipe dataset for vertex classification task. 
More details can be found in this `websit <https://relational.fit.cvut.cz/dataset/CORA>`_.

Results
----------------

========    ======================  ======================  ======================
Model       Accuracy on Validation  Accuracy on Testing     F1 score on Testing
========    ======================  ======================  ======================
GCN         0.800                   0.823                   0.814
GAT         0.804                   0.824                   0.817
HGNN        0.804                   0.820                   0.811
HGNN+       0.802                   0.827                   0.820
========    ======================  ======================  ======================


GCN on Cooking200
---------------------------


HGNN on Cooking200
---------------------------

HGNN+ on Cooking200
---------------------------


