#Hypergraph Toolbox: HyperG
**HyperG** is a python toolbox for hypergraph-based deep learning, which is built upon [pytorch](https://pytorch.org/). 
 Edge in hypergraph named hyperedge can link more than two nodes, which allows hyperedge to express more than pair-wise 
 relation(like: entity-attribute relation, group relation, hierarchical relation and so on.). Thus, hypergraph owns more 
 powerful model ability than common graph. 
 
 It consists of sparse hypergraph construction, fusion, convolution operations, convenient util functions for medical
 image(MRI, Pathology, etc.), 3D(point cloud, view-based graph, etc.) and other hypergraph applications(to be continue...).
 Hypergrpah inductive learning and hypergraph transductive learning examples is also included in this toolbox. What's more,
 we write several examples that deploy hypergraph in different tasks like: Classification, Segmentation and Regression.    
 
 The supported operations include:
 
 * **Hyperedge base operations**: compute hyperedge/node degree, add/remove hypergraph self loop, count hyperedge/node number,
 
 * **Hyperedge construction operations**: construct hyperedge group from grid-like structure (image) with spatial neighbors, 
 construct hyperedge group from feature spatial neighbors. K Nearest Neighbors algorithm is supported here.
 
 * **Hyperedge group/Hypergraph fusion operations**: fusion hypergraphs with concatenate constructed hypergraph incidence matrix.
 
 * **Hypergraph Convolution**: the common hyconv(hypergrpah convolution) ([Feng et al. AAAI2019](https://github.com/iMoonLab/HGNN)) 
 is implemented here.
 
 * **models**: HGNN([Feng et al. AAAI2019](https://github.com/iMoonLab/HGNN)) with two hyconv layers, ResNet(18, 34, 50, 101, 152)
 ([He et al.](https://arxiv.org/abs/1512.03385)), and ResNet_HGNN a combination of ResNet and HGNN for image input and real-time
 construct hypergraph supported.
 
 * **utils**: some convenient util functions(to be continue... ):
    * **data**: multiple modality data supported (to be continue...)
        * **mri**: mri series read and write functions.
        * **pathology**: sample patches from WSI slide return patch coordinates(left top point) and patch width and height. 
        draw sampled patches on WSI slide function for overview or visualization.
    * **meter**: evaluate meters in hypergraph learning.
        * **inductive**:  *C-Index Meter* for survival prediction.
        * **transductive**: compute class accuracy in classification task for transductive learning, compute IOU Score for 
        segmentation task in transductive learning. 
    * **visualization**: some visualization functions.
        * **transductive**: visualize segmentation result in transductive learning. 
    
 
  
 including edge construction, graph construction and graph/hypergraph convolution.