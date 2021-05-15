
Distributed Training
====================

We discuss how to use distributed training to train a Bootleg model on the full Wikipedia save. This tutorial assumes you have already completed the `Basic Training Tutorial`_.

As Wikipedia has over 5 million entities and over 50 million sentences, training on the full Wikipedia save is computationally expensive. We recommend using a `p4d.24xlarge <https://aws.amazon.com/ec2/instance-types/p4/>`_ instance on AWS to train on Wikipedia.

We provide a config for training Wikipedia `here <https://github.com/HazyResearch/bootleg/tree/master/configs/tutorial/wiki_uncased_ft.yaml>`_. Note this config is the config used to train the pretrained model provided in the `End-to-End Tutorial <https://github.com/HazyResearch/bootleg/tree/master/tutorials/end2end_ned_tutorial.ipynb>`_.

1. Downloading the Data
-----------------------

We provide scripts to download:


#. Prepped Wikipedia data (training and dev datasets)
#. Wikipedia entity data and embedding data

To download the Wikipedia data, run the command below with the directory to download the data to. Note that the prepped Wikipedia data will require ~200GB of disk space and will take some time to download and decompress the prepped Wikipedia data (16GB compressed, ~150GB uncompressed).

.. code-block::

   bash download_wiki.sh <DOWNLOAD_DIRECTORY>

To download (2) above, run the command

.. code-block::

   bash download_data.sh <DOWNLOAD_DIRECTORY>

At the end, the directory structure should be

.. code-block::

  <DOWNLOAD_DIRECTORY>
    wiki_data/
        prep/
    entity_db/
        entity_mappings/
        type_mappings/
        kg_mappings/
        prep/

2. Setting up Distributed Training
----------------------------------

`Emmental <https://github.com/SenWu/emmental>`_, the training framework of Bootleg, supports distributed training using PyTorch's `Data Parallel <https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html>`_ or `Distributed Data Parallel <https://pytorch.org/docs/stable/notes/ddp.html>`_ framework. We recommend DDP for training.

There is nothing that needs to change to get distributed training to work. We do, however, recommend setting the following params

.. code-block::

    emmental:
        ...
        distributed_backend: nccl
        fp16: true

This allows for fp16 and making sure the ``nccl`` backend is used. Note that when training with DDP, the ``batch_size`` is **per gpu**. With standard data parallel, the ``batch_size`` is across all GPUs.

From the `Basic Training Tutorial`_, recall that the directory paths should be set to where we want to save our models and read the data, including:

* ``cache_dir`` in ``data_config.word_embedding``
* ``data_dir``, ``entity_dir``, and ``emb_dir`` in ``data_config``

We have already set these directories in the provided Wikipedia config, but you will need to update ``data_dir``, ``entity_dir``, and ``emb_dir`` to where you downloaded the data in step 1 and may want to update ``log_dir`` to where you want to save the model checkpoints and logs.

3. Training the Model
---------------------

As we provide the Wikipedia data already prepped, we can jump immediately to training. To train the model with 8 gpus using DDP, we simply run:

.. code-block::

   python3 -m torch.distributed.launch --nproc_per_node=8  bootleg/run.py --config_script configs/tutorial/wiki_uncased_ft.yaml

To train using DP, simply run

.. code-block::

   python3 bootleg/run.py --config_script configs/tutorial/wiki_uncased_ft.yaml

and Emmental will automatically using distributed training (you can turn this off by ``dataparallel: false`` in the ``emmental`` config block.

Once the training begins, we should see all GPUs being utilized.

If we want to change the config (e.g. change the maximum number of aliases or the maximum word token len), we would need to re-prep the data and would run the command below. Note it takes several hours to perform Wikipedia pre-processing on a 56-core machine:

4. Evaluating with Slices
-------------------------

We use evaluation slices to understand the performance of Bootleg on important subsets of the dataset. To use evaluation slices, alias-entity pairs are labelled as belonging to specific slices in the ``slices`` key of the dataset.

In the Wikipedia data in this tutorial, we provide three "slices" of the dev dataset in addition to the "final_loss" (all examples) slice. For each of these three slices, the alias being scored must have more than one candidate. This filters trivial examples all models get correct.


* ``unif_NS_TS``: The gold entity does not occur in the training dataset (toes).
* ``unif_NS_TL``: The gold entity occurs globally 10 or fewer times in the training dataset (tail).
* ``unif_NS_TO``: The gold entity occurs globally between 11-1000 times in the training dataset (torso).
* ``unif_NS_HD``: The gold entity occurs globally greater than 1000 times in the training dataset (head).
* ``unif_NS_all``: All gold entities.

To use the slices for evaluation, they must also be specified in the ``eval_slices`` section of the ``run_config`` (see the `Wikipedia config`_ as an example).

When the dev evaluation occurs during training, we should see the performance on each of the slices that are specified in ``eval_slices``. These slices help us understand how well Bootleg performs on more challenging subsets. The frequency of dev evaluation can be specified by the ``evaluation_freq`` parameter in the ``emmental`` block.


.. _Basic Training Tutorial: ../gettingstarted/training.html>
.. _Wikipedia config: https://github.com/HazyResearch/bootleg/tree/master/configs/tutorial/wiki_uncased_ft.yaml
