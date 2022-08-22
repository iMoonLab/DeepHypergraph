dhg.nn
========

.. We have implemented several neural network architectures.

Common Layers
----------------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: conv_template.rst

   dhg.nn.MLP


Layers on Low-Order Structures
-------------------------------------------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: conv_template.rst

   dhg.nn.GCNConv
   dhg.nn.GraphSAGEConv
   dhg.nn.GATConv
   dhg.nn.GINConv


Layers on High-Order Structures
----------------------------------------------

.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: conv_template.rst

    dhg.nn.HGNNConv
    dhg.nn.HGNNPConv
    dhg.nn.JHConv
    dhg.nn.HyperGCNConv
    dhg.nn.HNHNConv


Loss Functions
----------------------------------------------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: loss_template.rst

   dhg.nn.BPRLoss

Regularizations
----------------------------------------------

.. autosummary::
   :toctree: ../generated/
   :nosignatures:
   :template: reg_template.rst

   dhg.nn.EmbeddingRegularization
