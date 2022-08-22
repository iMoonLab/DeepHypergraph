构建结构
===================================

Introduction
----------------
The core motivation of DHG is to attach those spectral-based and spatial-based operations to each specified structure.

Currently, the DHG has implemented the following structures:


.. csv-table:: Summary of Supported Structures
    :header: "Structure", "Class Name", "Type", "Spectral-based Operations", "Spatial-based Operations"
    :widths: 4 3 3 5 5

    "Simple Graph", "dhg.Graph", "Low-order", ":math:`\mathcal{L}_{GCN}`", ":math:`v \rightarrow v`"
    "Directed Graph", "dhg.Digraph", "Low-order", *To Be Added*, "| :math:`v_{src} \rightarrow v_{dst}`
    | :math:`v_{dst} \rightarrow v_{src}`"
    "Bipartite Graph", "dhg.Bigraph", "Low-order", *To Be Added*, "| :math:`u \rightarrow v` 
    | :math:`v \rightarrow u`"
    "Simple Hypergraph", "dhg.Hypergraph", "High-order", ":math:`\mathcal{L}_{HGNN}`", "| :math:`v \rightarrow e`
    | :math:`e \rightarrow v`"


Build Low-Order Structure
----------------------------

Build Simple Graphs
+++++++++++++++++++++++++++++++
Smoothing

Build Directed Graphs
+++++++++++++++++++++++++++++
message Passing

Build Bipartite Graphs
++++++++++++++++++++++++++

Build High-Order Structure
-------------------------------------------

Build Simple Hypergraphs
++++++++++++++++++++++++++++


Examples
--------------
