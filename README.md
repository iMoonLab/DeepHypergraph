<p align="center">
    <img src="https://deephypergraph.com/logo_DHG.png" height="200">
</p>

![Release version](https://img.shields.io/github/v/release/iMoonLab/DeepHypergraph)
[![PyPI version](https://img.shields.io/pypi/v/dhg?color=purple)](https://pypi.org/project/dhg/)
[![Website Build Status](https://github.com/yifanfeng97/dhg-page-source/actions/workflows/website.yml/badge.svg)](https://deephypergraph.com/)
[![Documentation Status](https://readthedocs.org/projects/deephypergraph/badge/?version=latest)](https://deephypergraph.readthedocs.io/)
[![Downloads](https://pepy.tech/badge/dhg)](https://pepy.tech/project/dhg)
[![Visits Badge](https://visitor-badge.glitch.me/badge?page_id=iMoonLab.DeepHypergraph)](https://visitor-badge.glitch.me/)
[![license](https://img.shields.io/github/license/imoonlab/DeepHypergraph)](LICENSE)
<!-- [![Code style: Black](https://img.shields.io/badge/code%20style-Black-000000.svg)](https://github.com/psf/black) -->
<!-- [![Supported Python versions](https://img.shields.io/pypi/pyversions/dhg)](https://pypi.org/project/dhg/) -->


**[Website](https://deephypergraph.com/)** | **[Documentation](https://deephypergraph.readthedocs.io/)** | **[Tutorials](https://deephypergraph.readthedocs.io/en/latest/tutorial/overview.html)** | **[中文文档](https://deephypergraph.readthedocs.io/en/latest/zh/overview.html)** | **[Official Examples](https://deephypergraph.readthedocs.io/en/latest/examples/vertex_cls/index.html)** | **[Discussions](https://github.com/iMoonLab/DeepHypergraph/discussions)**


## News
- 2024-01-31 -> **v0.9.4** is now available! Fix some bugs and more datasets are included!
- 2024-01-31 -> **v0.9.4** 正式发布！ 修复了若干bug，包含更多数据集！
- 2022-12-28 -> **v0.9.3** is now available! More datasets and operations of hypergraph are included!
- 2022-12-28 -> **v0.9.3** 正式发布！ 包含更多数据集和超图操作！
- 2022-09-25 -> **v0.9.2** is now available! More datasets, SOTA models, and visualizations are included!
- 2022-09-25 -> **v0.9.2** 正式发布！ 包含更多数据集、最新模型和可视化功能！
- 2022-08-25 -> DHG's first version **v0.9.1** is now available! 
- 2022-08-25 -> DHG的第一个版本 **v0.9.1** 正式发布！


**DHG** *(DeepHypergraph)* is a deep learning library built upon [PyTorch](https://pytorch.org) for learning with both Graph Neural Networks and Hypergraph Neural Networks. It is a general framework that supports both low-order and high-order message passing like **from vertex to vertex**, **from vertex in one domain to vertex in another domain**, **from vertex to hyperedge**, **from hyperedge to vertex**, **from vertex set to vertex set**.

It supports a wide variety of structures like low-order structures (graph, directed graph, bipartite graph, etc.), high-order structures (hypergraph, etc.). Various spectral-based operations (like Laplacian-based smoothing) and spatial-based operations (like message psssing from domain to domain) are integrated inside different structures. It provides multiple common metrics for performance evaluation on different tasks. Many state-of-the-art models are implemented and can be easily used for research. We also provide various visualization tools for both low-order structures and high-order structures. 

In addition, DHG's [dhg.experiments](https://deephypergraph.readthedocs.io/en/latest/api/experiments.html) module (that implements **Auto-ML** upon [Optuna](https://optuna.org)) can help you automatically tune the hyper-parameters of your models in training and easily outperforms the state-of-the-art models.

![Framework of DHG Structures](https://deephypergraph.com/fw_dhg_structure.jpg)

![Framework of DHG Function Library](https://deephypergraph.com/fw_dhg_other.jpg)

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
DHG provides a powerful visualization tool for graph and hypergraph. You can easily visualize the structure of your graph and hypergraph.

- **Bridge the Gap between Graphs and Hypergraphs**:
DHG provides functions to build hypergraph from graph and build graph from hypergraph. Maybe promoting the graph to hypergraph can exploit those potential high-order connections and improve the performance of your model.

- **Attach Spectral/Spatial-Based Operations to Structure**:
In DHG, those Laplacian matrices and message passing functions are attached to the graph/hypergraph structure. As soon as you build a structure with DHG, those functions will be ready to be used in the process of building your model.

- **Comprehensive, Flexible, and Convenience**:
DHG provides random graph/hypergraph generators, various state-of-the-art graph/hypergraph convolutional layers and models, various public graph/hypergraph datasets, and various evaluation metrics.

- **Support Tuning Structure and Model with Auto-ML**:
The Optuna library endows DHG with the Auto-ML ability. DHG supports automatically searching the optimal configurations for the construction of graph/hypergraph structure and the optimal hyper-parameters for your model and training.

## Installation


Current, the stable version of **DHG** is 0.9.4. You can install it with ``pip`` as follows:

```python
pip install dhg
```

You can also try the nightly version (0.9.5) of **DHG** library with ``pip`` as follows:

```python
pip install git+https://github.com/iMoonLab/DeepHypergraph.git
```

Nightly version is the development version of **DHG**. It may include the lastest SOTA methods and datasets, but it can also be unstable and not fully tested. 
If you find any bugs, please report it to us in [GitHub Issues](https://github.com/iMoonLab/DeepHypergraph/issues).

## Quick Start

### Visualization

You can draw the graph, hypergraph, directed graph, and bipartite graph with DHG's visualization tool. More details see the [Tutorial](https://deephypergraph.readthedocs.io/en/latest/tutorial/vis_structure.html)


![Visualization of graph and hypergraph](https://deephypergraph.com/readme_graph_hypergraph.png)

```python
import matplotlib.pyplot as plt
import dhg
# draw a graph
g = dhg.random.graph_Gnm(10, 12)
g.draw()
# draw a hypergraph
hg = dhg.random.hypergraph_Gnm(10, 8)
hg.draw()
# show figures
plt.show()
```

![Visualization of directed graph and bipartite graph](https://deephypergraph.com/readme_digraph_bigraph.png)

```python
import matplotlib.pyplot as plt
import dhg
# draw a directed graph
g = dhg.random.digraph_Gnm(12, 18)
g.draw()
# draw a bipartite graph
g = dhg.random.bigraph_Gnm(30, 40, 20)
g.draw()
# show figures
plt.show()
```

### Learning on Low-Order Structures

On graph structures, you can smooth a given vertex features with GCN's Laplacian matrix by:

```python
import torch
import dhg
g = dhg.random.graph_Gnm(5, 8)
X = torch.rand(5, 2)
X_ = g.smoothing_with_GCN(X)
```

On graph structures, you can pass messages from vertex to vertex with `mean` aggregation by:

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

On hypergraph structures, you can smooth a given vertex features with HGNN's Laplacian matrix by:

```python
import torch
import dhg
hg = dhg.random.hypergraph_Gnm(5, 4)
X = torch.rand(5, 2)
X_ = hg.smoothing_with_HGNN(X)
```

On hypergraph structures, you can pass messages from vertex to hyperedge with `mean` aggregation by:

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

- **[BlogCatalog](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.BlogCatalog.html#dhg.data.BlogCatalog)**: A social network dataset for vertex classification task.

- **[Flickr](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.Flickr.html#dhg.data.Flickr)**: A social network dataset for vertex classification task.

- **[Github](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.Github.html#dhg.data.Github)**: A collaboration network dataset for vertex classification task.

- **[Facebook](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.Facebook.html#dhg.data.Facebook)**: A social network dataset for vertex classification task.

- **[MovieLens1M](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.MovieLens1M.html#dhg.data.MovieLens1M)**: A movie dataset for user-item recommendation task.

- **[AmazonBook](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.AmazonBook.html#dhg.data.AmazonBook)**: An Amazon dataset for user-item recommendation task.

- **[Yelp2018](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.Yelp2018.html#dhg.data.Yelp2018)**: A restaurant review dataset for user-item recommendation task.

- **[Gowalla](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.Gowalla.html#dhg.data.Gowalla)**: A location's feedback dataset for user-item recommendation task.

- **[TecentBiGraph](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.TencentBiGraph.html#dhg.data.TencentBiGraph)**: A social network dataset for vertex classification task.

- **[CoraBiGraph](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.CoraBiGraph.html#dhg.data.CoraBiGraph)**: A citation network dataset for vertex classification task.

- **[PubmedBiGraph](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.PubmedBiGraph.html#dhg.data.PubmedBiGraph)**: A citation network dataset for vertex classification task.

- **[CiteseerBiGraph](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.CiteseerBiGraph.html#dhg.data.CiteseerBiGraph)**: A citation network dataset for vertex classification task.

- **[Cooking200](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.Cooking200.html#dhg.data.Cooking200)**: A cooking recipe dataset for vertex classification task.

- **[CoauthorshipCora](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.CoauthorshipCora.html#dhg.data.CoauthorshipCora)**: A citation network dataset for vertex classification task.

- **[CoauthorshipDBLP](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.CoauthorshipDBLP.html#dhg.data.CoauthorshipDBLP)**: A citation network dataset for vertex classification task.

- **[CocitationCora](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.CocitationCora.html#dhg.data.CocitationCora)**: A citation network dataset for vertex classification task.

- **[CocitationPubmed](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.CocitationCiteseer.html#dhg.data.CocitationCiteseer)**: A citation network dataset for vertex classification task.

- **[CocitationCiteseer](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.CocitationPubmed.html#dhg.data.CocitationPubmed)**: A citation network dataset for vertex classification task.

- **[YelpRestaurant](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.YelpRestaurant.html#dhg.data.YelpRestaurant)**: A restaurant-review network dataset for vertex classification task.

- **[WalmartTrips](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.WalmartTrips.html#dhg.data.WalmartTrips)**: A user-product network dataset for vertex classification task.

- **[HouseCommittees](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.HouseCommittees.html#dhg.data.HouseCommittees)**: A committee network dataset for vertex classification task.

- **[News20](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.News20.html#dhg.data.News20)**: A newspaper network dataset for vertex classification task.

- **[DBLP8k](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.DBLP8k.html#dhg.data.DBLP8k)**: The DBLP-8k dataset is a citation network dataset for link prediction task.

- **[DBLP4k](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.DBLP4k.html#dhg.data.DBLP4k)**: The DBLP-4k dataset is a citation network dataset for vertex classification task.

- **[IMDB4k](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.IMDB4k.html#dhg.data.IMDB4k)**: The IMDB-4k dataset is a movie dataset for vertex classification task.

- **[Recipe100k](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.Recipe100k.html#dhg.data.Recipe100k)**: The Recipe100k dataset is a recipe-ingredient network dataset for vertex classification task.

- **[Recipe200k](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.Recipe200k.html#dhg.data.Recipe200k)**: The Recipe200k dataset is a recipe-ingredient network dataset for vertex classification task.

- **[Yelp3k](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.Yelp3k.html#dhg.data.Yelp3k)**: The Yelp3k dataset is a subset of Yelp-Restaurant dataset for vertex classification task.

- **[Tencent2k](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.data.Tencent2k.html#dhg.data.Tencent2k)**: The Tencent2k dataset is a social network dataset for vertex classification task.

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

- **[BGNN-Adv](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.models.BGNN_Adv.html#dhg.models.BGNN_Adv)** model of [Cascade-BGNN: Toward Efficient Self-supervised Representation Learning on Large-scale Bipartite Graphs](https://arxiv.org/pdf/1906.11994.pdf) paper (TNNLS 2020).

- **[BGNN-MLP](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.models.BGNN_MLP.html#dhg.models.BGNN_MLP)** model of [Cascade-BGNN: Toward Efficient Self-supervised Representation Learning on Large-scale Bipartite Graphs](https://arxiv.org/pdf/1906.11994.pdf) paper (TNNLS 2020).


### On High-Order Structures

- **[HGNN](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.models.HGNN.html#dhg.models.HGNN)** model of [Hypergraph Neural Networks](https://arxiv.org/pdf/1809.09401) paper (AAAI 2019).

- **[HGNN+](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.models.HGNNP.html#dhg.models.HGNNP)** model of [HGNN+: General Hypergraph Neural Networks](https://ieeexplore.ieee.org/document/9795251) paper (IEEE T-PAMI 2022).

- **[HyperGCN](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.models.HyperGCN.html#dhg.models.HyperGCN)** model of [HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs](https://papers.nips.cc/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf) paper (NeurIPS 2019).

- **[DHCF](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.models.DHCF.html#dhg.models.DHCF)** model of [Dual Channel Hypergraph Collaborative Filtering](https://dl.acm.org/doi/10.1145/3394486.3403253) paper (KDD 2020).

- **[HNHN](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.models.HNHN.html#dhg.models.HNHN)** model of [HNHN: Hypergraph Networks with Hyperedge Neurons](https://arxiv.org/pdf/2006.12278.pdf) paper (ICML 2020).

- **[UniGCN](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.models.UniGCN.html#dhg.models.UniGCN)** model of [UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks](https://arxiv.org/pdf/2105.00956.pdf) paper (IJCAI 2021).

- **[UniGAT](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.models.UniGAT.html#dhg.models.UniGAT)** model of [UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks](https://arxiv.org/pdf/2105.00956.pdf) paper (IJCAI 2021).

- **[UniSAGE](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.models.UniSAGE.html#dhg.models.UniSAGE)** model of [UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks](https://arxiv.org/pdf/2105.00956.pdf) paper (IJCAI 2021).

- **[UniGIN](https://deephypergraph.readthedocs.io/en/latest/generated/dhg.models.UniGIN.html#dhg.models.UniGIN)** model of [UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks](https://arxiv.org/pdf/2105.00956.pdf) paper (IJCAI 2021).



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

DHG is developed by DHG's core team including [Yifan Feng](http://fengyifan.site/), [Xinwei Zhang](https://github.com/zhangxwww), [Jielong Yan](https://github.com/JasonYanjl), [Xiangmin Han](https://scholar.google.com/citations?user=Y96h0t0AAAAJ&hl=zh-CN&oi=ao), [Yue Gao](http://moon-lab.tech/), and [Qionghai Dai](https://ysg.ckcest.cn/html/details/8058/index.html). It is maintained by the [iMoon-Lab](http://moon-lab.tech/), Tsinghua University. You can contact us at [email](mailto:evanfeng97@gmail.com).


## License

DHG uses Apache License 2.0.