Installation
============

To test changes in the package, you install it in `editable mode`_ locally in your virtualenv by running::

    $ make dev

This will also install our pre-commit hooks and local packages needed for style checks.

.. tip::

    If you need to install a locally edited version of bootleg in a separate location, such as an application, you can directly install your locally modified version::

        $ pip install -e path/to/bootleg/

    in the virtualenv of your application.

Note, you can test the `pip` downloadable version using `TestPyPI <https://test.pypi.org/>`_. To handle dependencies, run

.. code-block::

    pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple bootleg

.. _editable mode: https://packaging.python.org/tutorials/distributing-packages/#working-in-development-mode
