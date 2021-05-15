Code Style
==========

For code consistency, we have a `pre-commit`_ configuration file so that you can easily install pre-commit hooks to run style checks before you commit your files. You can setup our pre-commit hooks by running::

    $ pip install -r requirements-dev.txt
    $ pre-commit install

Or, just run::

    $ make dev

Now, each time you commit, checks will be run using the packages explained below.

We use `black`_ as our Python code formatter with its default settings. Black helps minimize the line diffs and allows you to not worry about formatting during your own development. Just run black on each of your files before committing them.

.. tip::

    Whatever editor you use, we recommend checking out `black editor integrations`_ to help make the code formatting process just a few keystrokes.


For sorting imports, we reply on `isort`_. Our repository already includes a ``.isort.cfg`` that is compatible with black. You can run a code style check on your local machine by running our checks::

    $ make check

.. _pre-commit: https://pre-commit.com/
.. _isort: https://github.com/timothycrosley/isort
.. _black editor integrations: https://github.com/ambv/black#editor-integration
.. _black: https://github.com/ambv/black
