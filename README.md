<p align="center">
    <img src="https://deephypergraph.com/logo_DHG.png" height="200">
</p>

[![Documentation Status](https://readthedocs.org/projects/deephypergraph/badge/?version=latest)](https://deephypergraph.readthedocs.io/en/latest/?badge=latest)
![license](https://img.shields.io/hexpm/l/plug)
---------------------------------------------------------------

**[Website](https://deephypergraph.com/)** | **[Documentation](https://deephypergraph.readthedocs.io/)** | **[Tutorials](https://deephypergraph.readthedocs.io/en/latest/tutorial/overview.html)** | **[Official Examples]()** | **[Discussions](https://github.com/iMoonLab/DeepHypergraph/discussions)**


**DHG** *(DeepHypergraph)* is a deep learning library built upon [PyTorch](https://pytorch.org) for learning with both Graph Neural Networks and Hypergraph Neural Networks. It is a general framework that supports both low-order and high-order message passing like **from vertex to vertex**, **from vertex in one domain to vertex in another domain**, **from vertex to hyperedge**, **from hyperedge to vertex**, **from vertex set to vertex set**.

It supports a wide variety of structures like low-order structures (simple graph, directed graph, bipartite graph, etc.), high-order structures (simple hypergraph, etc.). Various spectral-based operations (like Laplacian-based smoothing) and spatial-based operations (like message psssing from domain to domain) are integrated inside different structures. It provides multiple common metrics for performance evaluation on different tasks. Many state-of-the-art models are implemented and can be easily used for research. We also provide various visualization tools for both low-order structures and high-order structures. 

In addition, DHG's [dhg.experiments](https://deephypergraph.readthedocs.io/en/latest/api/experiments.html) module (that implements **Auto-ML** upon [Optuna](https://optuna.org)) can help you automatically tune the hyper-parameters of your models in training and easily outperforms the state-of-the-art models.

* [Hightlights](#highlights)
* [Installation](#installation)
* [Quick Start](#quick-start)
* [Examples](#examples)
* [Datasets](#datasets)
* [Metrics](#metrics)
* [Implemented Models](#implemented-models)


---------------------------------------------------------------

## Highlights

- **Support High-Order Message Passing on Structure**: 
DHG supports pair-wise message passing on the graph structure and beyond-pair-wise message passing on the hypergraph structure.

- **Shared Ecosystem with Pytorch Framework**:
DHG is built upon Pytorch, and any Pytorch-based models can be integrated into DHG. If you are familiar with Pytorch, you can easily use DHG.

- **Powerful API for Designing GNNs and HGNNs**:
DHG provides various Laplacian matrices and message passing functions to help build your spectral/spatial-based models, respectively.

- **Visualization of Graphs and Hypergraphs**
DHG provides a powerful visualization tool for graphs and hypergraphs. You can easily visualize the structure of your graphs and hypergraphs.

- **Bridge the Gap between Graphs and Hypergraphs**:
DHG provides functions to build hypergraphs from graphs and build graphs from hypergraphs. Maybe promoting the graphs to hypergraphs can exploit those potential high-order connections and improve the performance of your model.

- **Attach Spectral/Spatial-Based Operations to Structure**:
In DHG, those Laplacian matrices and message passing functions are attached to the graph/hypergraph structure. As soon as you build a structure with DHG, those functions will be ready to be used in the process of building your model.

- **Comprehensive, Flexible, and Convenience**:
DHG provides random graph/hypergraph generators, various state-of-the-art graph/hypergraph convolutional layers and models, various public graph/hypergraph datasets, and various evaluation metrics.

- **Support Tuning Structure and Model with Auto-ML**:
The Optuna library endows DHG with the Auto-ML ability. DHG supports automatically searching the optimal configurations for the construction of graph/hypergraph structure and the optimal hyper-parameters for your model and training.

## Installation

DHG is available for **Python 3.8** and **Pytorch 1.11.0** and above. 

### Install with pip

```bash
pip install dhg
```
### Install from Github Repository

```bash
pip install git+https://github.com/iMoonLab/DeepHypergraph.git
```

## Quick Start

### Learning on Low-Order Structures

On simple graph structures, you can smooth a given vertex features with GCN's Laplacian matrix by:

```python
import torch
import dhg
g = dhg.random.graph_Gnm(5, 8)
X = torch.rand(5, 2)
X_ = g.smoothing_with_GCN(X)
```

On simple graph structures, you can pass messages from vertex to vertex with `mean` aggregation by:

```python
import torch
import dhg
g = dhg.random.graph_Gnm(5, 8)
X = torch.rand(5, 2)
X_ = g.v2v(X, aggr="mean")
```

On directed graph structures, you can pass messages from vertex to vertex with `mean` aggregation by:

```python
import torch
import dhg
g = dhg.random.digraph_Gnm(5, 8)
X = torch.rand(5, 2)
X_ = g.v2v(X, aggr="mean")
```

On bipartite graph structures, you can smoothing vertex features with GCN's Laplacian matrix by:

```python
import torch
import dhg
g = dhg.random.bigraph_Gnm(3, 5, 8)
X_u, X_v = torch.rand(3, 2), torch.rand(5, 2)
X = torch.cat([X_u, X_v], dim=0)
X_ = g.smoothing_with_GCN(X, aggr="mean")
```

On bipartite graph structures, you can pass messages from vertex in `U` set to vertex in `V` set by `mean` aggregation by:

```python
import torch
import dhg
g = dhg.random.bigraph_Gnm(3, 5, 8)
X_u, X_v = torch.rand(3, 2), torch.rand(5, 2)
X_u_ = g.v2u(X_v, aggr="mean")
X_v_ = g.u2v(X_u, aggr="mean")
```

### Learning on High-Order Structures

On simple hypergraph structures, you can smooth a given vertex features with HGNN's Laplacian matrix by:

```python
import torch
import dhg
hg = dhg.random.hypergraph_Gnm(5, 4)
X = torch.rand(5, 2)
X_ = hg.smoothing_with_HGNN(X)
```

On simple hypergraph structures, you can pass messages from vertex to hyperedge with `mean` aggregation by:

```python
import torch
import dhg
hg = dhg.random.hypergraph_Gnm(5, 4)
X = torch.rand(5, 2)
Y_ = hg.v2e(X, aggr="mean")
```
Then, you can pass messages from hyperedge to vertex with `mean` aggregation by:

```python
X_ = hg.e2v(Y_, aggr="mean")
```
Or, you can pass messages from vertex set to vertex set with `mean` aggregation by:

```python
X_ = hg.v2v(X, aggr="mean")
```

## Examples

### Building the Convolution Layer of GCN

```python
class GCNConv(nn.Module):
    def __init__(self,):
        super().__init__()
        ...
        self.reset_parameters()

    def forward(self, X: torch.Tensor, g: dhg.Graph) -> torch.Tensor:
        # apply the trainable parameters ``theta`` to the input ``X``  
        X = self.theta(X)
        # smooth the input ``X`` with the GCN's Laplacian
        X = g.smoothing_with_GCN(X)
        X = F.relu(X)
        return X
```

### Building the Convolution Layer of GAT

```python
class GATConv(nn.Module):
    def __init__(self,):
        super().__init__()
        ...
        self.reset_parameters()

    def forward(self, X: torch.Tensor, g: dhg.Graph) -> torch.Tensor:
        # apply the trainable parameters ``theta`` to the input ``X``
        X = self.theta(X)
        # compute attention weights for each edge
        x_for_src = self.atten_src(X)
        x_for_dst = self.atten_dst(X)
        e_atten_score = x_for_src[g.e_src] + x_for_dst[g.e_dst]
        e_atten_score = F.leaky_relu(e_atten_score).squeeze()
        # apply ``e_atten_score`` to each edge in the graph ``g``, aggragete neighbor messages
        #  with ``softmax_then_sum``, and perform vertex->vertex message passing in graph 
        #  with message passing function ``v2v()``
        X = g.v2v(X, aggr="softmax_then_sum", e_weight=e_atten_score)
        X = F.elu(X)
        return X
```

### Building the Convolution Layer of HGNN

```python
class HGNNConv(nn.Module):
    def __init__(self,):
        super().__init__()
        ...
        self.reset_parameters()

    def forward(self, X: torch.Tensor, hg: dhg.Hypergraph) -> torch.Tensor:
        # apply the trainable parameters ``theta`` to the input ``X``
        X = self.theta(X)
        # smooth the input ``X`` with the HGNN's Laplacian
        X = hg.smoothing_with_HGNN(X)
        X = F.relu(X)
        return X
```


### Building the Convolution Layer of HGNN $^+$

```python
class HGNNPConv(nn.Module):
    def __init__(self,):
        super().__init__()
        ...
        self.reset_parameters()

    def forward(self, X: torch.Tensor, hg: dhg.Hypergraph) -> torch.Tensor:
        # apply the trainable parameters ``theta`` to the input ``X``
        X = self.theta(X)
        # perform vertex->hyperedge->vertex message passing in hypergraph
        #  with message passing function ``v2v``, which is the combination
        #  of message passing function ``v2e()`` and ``e2v()``
        X = hg.v2v(X, aggr="mean")
        X = F.relu(X)
        return X
```


## Datasets

Currently, we have added the following datasets:

- **[Cora](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.Cora.html#dhg.data.Cora)**: A citation network dataset for vertex classification task.

- **[PubMed](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.Pubmed.html#dhg.data.Pubmed)**: A citation network dataset for vertex classification task.

- **[Citeseer](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.Citeseer.html#dhg.data.Citeseer)**: A citation network dataset for vertex classification task.

- **[Cooking200](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.Cooking200.html#dhg.data.Cooking200)**: A cooking recipe dataset for vertex classification task.

- **[MovieLens1M](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.MovieLens1M.html#dhg.data.MovieLens1M)**: A movie dataset for user-item recommendation task.

- **[AmazonBook](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.AmazonBook.html#dhg.data.AmazonBook)**: An Amazon dataset for user-item recommendation task.

- **[Yelp2018](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.Yelp2018.html#dhg.data.Yelp2018)**: A restaurant review dataset for user-item recommendation task.

- **[Gowalla](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.Gowalla.html#dhg.data.Gowalla)**: A location's feedback dataset for user-item recommendation task.

## Metrics

### Classification Metrics

- **[Accuracy](https://deephypergraph.readthedocs.io/en/latest/api/metrics.html#dhg.metrics.classification.accuracy)**: Calculates the accuracy of the predictions.

- **[F1-Score](https://deephypergraph.readthedocs.io/en/latest/api/metrics.html#dhg.metrics.classification.f1_score)**: Calculates the F1-score of the predictions.

- **[Confusion Matrix](https://deephypergraph.readthedocs.io/en/latest/api/metrics.html#dhg.metrics.classification.confusion_matrix)**: Calculates the confusion matrix of the predictions.

### Recommender Metrics

- **[Precision@k](https://deephypergraph.readthedocs.io/en/latest/api/metrics.html#dhg.metrics.recommender.precision)**: Calculates the precision@k of the predictions.

- **[Recall@k](https://deephypergraph.readthedocs.io/en/latest/_modules/dhg/metrics/recommender.html#recall)**: Calculates the recall@k of the predictions.

- **[NDCG@k](https://deephypergraph.readthedocs.io/en/latest/api/metrics.html#dhg.metrics.recommender.ndcg)**: Calculates the normalized discounted cumulative gain@k of the predictions.

### Retrieval Metrics

- **[Precision@k](https://deephypergraph.readthedocs.io/en/latest/api/metrics.html#dhg.metrics.retrieval.precision)**: Calculates the precision@k of the predictions.

- **[Recall@k](https://deephypergraph.readthedocs.io/en/latest/api/metrics.html#dhg.metrics.retrieval.recall)**: Calculates the recall@k of the predictions.

- **[mAP@k](https://deephypergraph.readthedocs.io/en/latest/api/metrics.html#dhg.metrics.retrieval.map)**: Calculates the mAP@k of the predictions.

- **[NDCG@k](https://deephypergraph.readthedocs.io/en/latest/api/metrics.html#dhg.metrics.retrieval.ndcg)**: Calculates the normalized Discounted Cumulative Gain@k of the predictions.

- **[mRR@k](https://deephypergraph.readthedocs.io/en/latest/api/metrics.html#dhg.metrics.retrieval.mrr)**: Calculates the mean Reciprocal Rank@k of the predictions.

- **[PR-Curve](https://deephypergraph.readthedocs.io/en/latest/api/metrics.html#dhg.metrics.retrieval.pr_curve)**: Calculates the precision-recall curve of the predictions.

## Implemented Models

### On Low-Order Structures

- **[GCN](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.models.GCN.html#dhg.models.GCN)** model of [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907) paper (ICLR 2017).

- **[GraphSAGE](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.models.GraphSAGE.html#dhg.models.GraphSAGE)** model of [Inductive Representation Learning on Large Graphs](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf) paper (NeurIPS 2017).

- **[GAT](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.models.GAT.html#dhg.models.GAT)** model of [Graph Attention Networks](https://arxiv.org/pdf/1710.10903) paper (ICLR 2018).

- **[GIN](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.models.GIN.html#dhg.models.GIN)** model of [How Powerful are Graph Neural Networks?](https://arxiv.org/pdf/1810.00826) paper (ICLR 2019).

- **[NGCF](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.models.NGCF.html#dhg.models.NGCF)** model of [Neural Graph Collaborative Filtering](https://arxiv.org/pdf/1905.08108) paper (SIGIR 2019).

- **[LightGCN](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.models.LightGCN.html#dhg.models.LightGCN)** model of [LightGCN: Lightweight Graph Convolutional Networks](https://arxiv.org/pdf/2002.02126) paper (SIGIR 2020).


### On High-Order Structures

- **[HGNN](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.models.HGNN.html#dhg.models.HGNN)** model of [Hypergraph Neural Networks](https://arxiv.org/pdf/1809.09401) paper (AAAI 2019).

- **[HGNN+](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.models.HGNNP.html#dhg.models.HGNNP)** model of [HGNN+: General Hypergraph Neural Networks](https://ieeexplore.ieee.org/document/9795251) paper (IEEE T-PAMI 2022).

- **[HyperGCN](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.models.HyperGCN.html#dhg.models.HyperGCN)** model of [HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs](https://papers.nips.cc/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf) paper (NeurIPS 2019).

- **[HNHN](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.models.HNHN.html#dhg.models.HNHN)** model of [HNHN: Hypergraph Networks with Hyperedge Neurons](https://arxiv.org/pdf/2006.12278.pdf) paper (ICML 2020).

- **[DHCF](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.models.DHCF.html#dhg.models.DHCF)** model of [Dual Channel Hypergraph Collaborative Filtering](https://dl.acm.org/doi/10.1145/3394486.3403253) paper (KDD 2020).



## Citing
If you find **DHG** is useful in your research, please consider citing:

```
@article{gao2022hgnn,
  title={HGNN $\^{}+ $: General Hypergraph Neural Networks},
  author={Gao, Yue and Feng, Yifan and Ji, Shuyi and Ji, Rongrong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}
```
```
@inproceedings{feng2019hypergraph,
  title={Hypergraph neural networks},
  author={Feng, Yifan and You, Haoxuan and Zhang, Zizhao and Ji, Rongrong and Gao, Yue},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={33},
  number={01},
  pages={3558--3565},
  year={2019}
}
```

## The DHG Team

DHG is developed by DHG's core team including [Yifan Feng](http://fengyifan.site/), [Xinwei Zhang](https://github.com/zhangxwww), and [Yue Gao (Adivsor)](http://moon-lab.tech/). It is maintained by the [iMoon-Lab](http://moon-lab.tech/), Tsinghua University. You can contact us at [email](mailto:evanfeng97@gmail.com).


## License

DHG uses Apache License 2.0.