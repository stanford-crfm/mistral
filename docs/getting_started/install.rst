Installation
============

Basic Installation
--------------------

Get the code ::

    git clone https://github.com/stanford-crfm/mistral.git

Set up the mistral conda env ::

    conda create -n mistral python=3.8.12 pytorch=1.11.0 torchdata cudatoolkit=11.3 -c pytorch
    conda activate mistral
    pip install -r setup/pip-requirements.txt

You may need to alter this environment depending on your CUDA set up.

Setting Up Weights And Biases
-------------------------------

Training runs transmit logs to `Weights & Biases <https://wandb.ai/>`_.

First make sure to set up an account on their web site.

Before doing training runs, set up your wandb credentials on your machine ::

    conda activate mistral
    cd mistral
    wandb init

The ``init`` process will direct you to a url with an API key you must enter.
During this process you will be asked to specify which team to use as well.

The project and group for a training run are set in the main
config file with the ``wandb`` and ``group`` keys respectively.
See ``conf/mistral-micro.yaml`` for an example.

If you do not want to send logs to Weights & Biases, run this command in the main mistral directory ::

    wandb offline

You can completely deactivate Weights & Biases logging with this command ::

    wandb disabled

For general info on ``wandb`` commands run ::

    wandb --help
