Configuration
==============

Quinine
---------

Configurations are specified using the `Quinine <https://github.com/krandiash/quinine>`_ library.

Quinine allows users to integrate multiple config files and layer configs on top of each other.
It is designed for machine learning projects with large sets of nested hyperparameters.

The easiest way to understand Quinine is to study ``conf/hello-world.yaml`` which is presented below.

This config specifies a variety of settings, and draws configurations from ``conf/datasets/wikitext103.yaml``,
``conf/models/gpt2-small.yaml`` and ``conf/trainers/gpt2-small.yaml``. This allows for clean separation of the
configs for the dataset (e.g. name or number of pre-processing workers), the model (e.g. sequence length),
and the trainer (e.g. learning rate), while high level configs are specified in the main config file.

Most of the defaults in ``conf/hello-world.yaml`` will work, but you will need to change
the Weights & Biases settings and specify the artifacts directories ``cache_dir`` and ``run_dir``.

Example config: hello-world.yaml
--------------------------------

``conf/hello-world.yaml`` is a basic configuration file that can be used for an introductory training run

.. include:: ../../conf/hello-world.yaml
   :literal:
