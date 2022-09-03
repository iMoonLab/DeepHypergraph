Contribute to DHG
======================

.. hint:: 

    - Author: `Yifan Feng (丰一帆) <https://fengyifan.site/>`_
    - Proof: Xinwei Zhang

DHG is a free software; you can redistribute it and/or modify it under the terms
of the Apache License 2.0. We welcome contributions.

To contribute the DHG you can select

1. Fix bugs.
2. Implement new features and enhancements.
3. Implement or enhance low-order or high-order structures in DHG.
4. Implement new spectral-based Laplacian matrix on a specified structure.
5. Implement new spatial-based message passing operation or aggregation operation.
6. Implement new layers on low-order or high-order structures.
7. Implement new state-of-the-art models.
8. Implement new loss functions.
9. Implement new metrics on a specified task.
10. Denote or upload new datasets.
11. Improve the quality of the documentation.
12. Auto-ML enhancements.

Once you have selected an option, we recommend first raise an issue or discussion on the Github.
Please read the following sections about coding style and testing first before development.
Our committors (who have write permission on the repository) will review the codes and suggest the necessary changes.
The PR could be merged once the reviewers approve the changes.

Coding Style
----------------
For python codes, we generally follow the `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ style guide.
The doc-string follows the `Google <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google>`_ style.

DHG use the `black <https://black.readthedocs.io/en/stable/>`_ library to format the code.
The configuration of the black formatter is in the ``pyproject.toml`` file.

Testing
-------------
The DHG's testing is located under ``tests/``.
You can add a new test file in the ``tests/``'s sub-directory for your implemented functions.
Run all tests with

.. code-block:: bash

    pytest .


Run individual test with

.. code-block:: bash

    pytest tests/xxx/xxx.py

``tests/xxx/xxx.py`` is an example filename.


Building Documentation
------------------------------
1. Clone the DHG repository.

    .. code-block:: bash

        git clone https://github.com/iMoonLab/DeepHypergraph


2. Install the ``requirements.txt`` below the ``docs/`` directory.

    .. code-block:: bash

        pip install -r docs/requirements.txt

3. Build the documentation with

    .. code-block:: bash

        cd docs
        make clean && make html

4. Open the html file (``docs/build/html/index.html``) with the browser.
