dhg.utils
=============

Logging
--------------------

.. autofunction:: dhg.utils.default_log_formatter

.. autofunction:: dhg.utils.simple_stdout2file


Download
-----------------------

.. autofunction:: dhg.utils.download_file

.. autofunction:: dhg.utils.check_file 

.. autofunction:: dhg.utils.download_and_check


Structure Helpers
----------------------

.. autofunction:: dhg.utils.remap_edge_list

.. autofunction:: dhg.utils.remap_edge_lists

.. autofunction:: dhg.utils.remap_adj_list

.. autofunction:: dhg.utils.remap_adj_lists

.. autofunction:: dhg.utils.edge_list_to_adj_list

.. autofunction:: dhg.utils.edge_list_to_adj_dict

.. autofunction:: dhg.utils.adj_list_to_edge_list


Dataset Wrapers
-----------------------

.. autoclass:: dhg.utils.UserItemDataset
    :members: __getitem__, __len__, sample_triplet, sample_neg_item, 
    :show-inheritance:
