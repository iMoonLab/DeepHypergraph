dhg.utils
=============


Structure Helpers
----------------------

.. autofunction:: dhg.utils.remap_edge_list

.. autofunction:: dhg.utils.remap_edge_lists

.. autofunction:: dhg.utils.remap_adj_list

.. autofunction:: dhg.utils.remap_adj_lists

.. autofunction:: dhg.utils.edge_list_to_adj_list

.. autofunction:: dhg.utils.edge_list_to_adj_dict

.. autofunction:: dhg.utils.adj_list_to_edge_list


Sparse Operations
-------------------------

.. autofunction:: dhg.utils.sparse_dropout

Dataset Splitting
----------------------

.. autofunction:: dhg.utils.split_by_num
    
.. autofunction:: dhg.utils.split_by_ratio
    
.. autofunction:: dhg.utils.split_by_num_for_UI_bigraph
    
.. autofunction:: dhg.utils.split_by_ratio_for_UI_bigraph


Dataset Wrapers
-----------------------

.. autoclass:: dhg.utils.UserItemDataset
    :members: __getitem__, __len__, sample_triplet, sample_neg_item, 
    :show-inheritance:


Log Helpers
--------------------

.. autofunction:: dhg.utils.default_log_formatter

.. autofunction:: dhg.utils.simple_stdout2file


Download Helpers
-----------------------

.. autofunction:: dhg.utils.download_file

.. autofunction:: dhg.utils.check_file 

.. autofunction:: dhg.utils.download_and_check
