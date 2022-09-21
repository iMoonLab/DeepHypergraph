dhg.datapipe
==============

We have implemented some datapipes to help you with the data processing.

Compose Datapipes 
----------------------

.. autofunction:: dhg.datapipe.compose_pipes

.. _api_datapipe_preprocess:

Transforms
--------------

.. autofunction:: dhg.datapipe.norm_ft

.. autofunction:: dhg.datapipe.min_max_scaler

.. autofunction:: dhg.datapipe.to_tensor 

.. autofunction:: dhg.datapipe.to_bool_tensor 

.. autofunction:: dhg.datapipe.to_long_tensor 


.. _api_datapipe_loader:

Loaders
---------

.. autofunction:: dhg.datapipe.load_from_pickle 

.. autofunction:: dhg.datapipe.load_from_txt

.. autofunction:: dhg.datapipe.load_from_json

