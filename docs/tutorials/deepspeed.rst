Training On Multiple Nodes With DeepSpeed
=========================================

Setting Up DeepSpeed
--------------------

`DeepSpeed <https://www.deepspeed.ai>`_ is an optimization library designed to facilitate distributed training.
The ``mistral`` conda environment (see :doc:`Installation <../getting_started/install>`) will install ``deepspeed``
when set up.

A user can use DeepSpeed for training with multiple gpu's on one node or many nodes. This tutorial will assume
you want to train on multiple nodes.

One essential configuration for DeepSpeed is the hostfile, which contains lists of machines accessible
via passwordless SSH and slot counts, which indicate the amount of available gpu's on each machine.

For this tutorial, we will assume the main machine's address is ``machine1``, that ``machine2`` is operating as a
worker machine, and that both machines have 8 gpu's. The corresponding hostfile should look like this: ::

    machine1 slots=8
    machine2 slots=8

DeepSpeed will look for the hostfile at ``/job/hostfile`` on ``machine1`` if a hostfile is not specified with the
``--hostfile`` argument. An example hostfile can be viewed at ``conf/deepspeed/hostfile``.

Configuring Training
---------------------

When running Deep Speed and Hugging Face, it is necessary to specify a collection of training settings in a DeepSpeed
json config file. These settings will be used to create the final ``TrainingArguments`` object for model training
and include such things as what optimizer or scheduler to use.

An example json config file is available at ``conf/deepspeed/z2-conf.json``:

.. include:: ../../conf/deepspeed/z2-conf.json
   :literal:


Launching A Training Run
------------------------

The following command (run on machine1) will launch training across your cluster: ::

    cd mistral
    conda activate mistral
    deepspeed --num_gpus 8 --num_nodes 2 --master_addr machine1 train.py --fileconf/tutorial-gpt2-micro.yaml --nnodes 2 --nproc_per_node 8 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 4 --training_arguments.deepspeed conf/deepspeed/z2-small-conf.json --run_id tutorial-gpt2-micro-multi-node > tutorial-gpt2-micro-multi-node.out 2> tutorial-gpt2-micro-multi-node.err

This assumes that the appropriate hostfile is set up at ``/job/hostfile`` on ``machine1``.

You should see output similar to the following in ``tutorial-gpt2-micro-multi-node.out`` if training is running as expected: ::

    machine2: {'loss': 6.5859, 'learning_rate': 0.0003537728376673855, 'activations/layer0_attention_weight_max': 6.225409030914307, 'activations/layer0_attention_weight_min': -6.8558735847473145, 'activations/layer1_attention_weight_max': 2.5137383937835693, 'activations/layer1_attention_weight_min': -3.4525303840637207, 'activations/layer2_attention_weight_max': 1.65605628490448, 'activations/layer2_attention_weight_min': -2.03672194480896, 'activations/layer3_attention_weight_max': 1.8134779930114746, 'activations/layer3_attention_weight_min': -1.6253358125686646, 'activations/layer4_attention_weight_max': 1.5045760869979858, 'activations/layer4_attention_weight_min': -1.482985496520996, 'activations/layer5_attention_weight_max': 3.2311043739318848, 'activations/layer5_attention_weight_min': -2.9691357612609863, 'activations/layer6_attention_weight_max': 5.682344913482666, 'activations/layer6_attention_weight_min': -4.275859355926514, 'activations/layer7_attention_weight_max': 0.7755581736564636, 'activations/layer7_attention_weight_min': -0.6805652379989624, 'activations/layer8_attention_weight_max': 1.4897541999816895, 'activations/layer8_attention_weight_min': -1.216135025024414, 'activations/layer9_attention_weight_max': 1.1379717588424683, 'activations/layer9_attention_weight_min': -1.412354826927185, 'activations/layer10_attention_weight_max': 2.4922404289245605, 'activations/layer10_attention_weight_min': -2.0055084228515625, 'activations/layer11_attention_weight_max': 1.4722517728805542, 'activations/layer11_attention_weight_min': -1.2682315111160278, 'epoch': 0.7}
    machine1: [2021-07-01 01:24:59,832] [INFO] [logging.py:60:log_dist] [Rank 0] step=150, skipped=17, lr=[0.0003537728376673855], mom=[[0.9, 0.95]]
    machine1: [2021-07-01 01:24:59,852] [INFO] [timer.py:154:stop] 0/1200, SamplesPerSec=463.8644895928809
    machine1: {'loss': 6.591, 'learning_rate': 0.0003537728376673855, 'activations/layer0_attention_weight_max': 5.9575395584106445, 'activations/layer0_attention_weight_min': -7.12982177734375, 'activations/layer1_attention_weight_max': 2.775029182434082, 'activations/layer1_attention_weight_min': -3.474602222442627, 'activations/layer2_attention_weight_max': 1.8722176551818848, 'activations/layer2_attention_weight_min': -1.927580714225769, 'activations/layer3_attention_weight_max': 1.8707917928695679, 'activations/layer3_attention_weight_min': -1.787396788597107, 'activations/layer4_attention_weight_max': 1.47317636013031, 'activations/layer4_attention_weight_min': -1.391649603843689, 'activations/layer5_attention_weight_max': 3.2698564529418945, 'activations/layer5_attention_weight_min': -2.83353328704834, 'activations/layer6_attention_weight_max': 5.822953701019287, 'activations/layer6_attention_weight_min': -4.2001142501831055, 'activations/layer7_attention_weight_max': 0.782840371131897, 'activations/layer7_attention_weight_min': -0.7528175115585327, 'activations/layer8_attention_weight_max': 1.5653538703918457, 'activations/layer8_attention_weight_min': -1.1807199716567993, 'activations/layer9_attention_weight_max': 1.1230956315994263, 'activations/layer9_attention_weight_min': -1.4319841861724854, 'activations/layer10_attention_weight_max': 2.5261030197143555, 'activations/layer10_attention_weight_min': -1.9104121923446655, 'activations/layer11_attention_weight_max': 1.4361441135406494, 'activations/layer11_attention_weight_min': -1.2555559873580933, 'epoch': 0.7}
    ...
    machine1: [2021-07-01 01:25:12,365] [INFO] [engine.py:1680:_save_zero_checkpoint] zero checkpoint saved hello-world/runs/hello-world-multi-node/checkpoint-150/global_step150/zero_pp_rank_5_mp_rank_00optim_states.pt
    ...
    machine1: [2021-07-01 01:25:48,146] [INFO] [timer.py:154:stop] 0/1460, SamplesPerSec=466.31161256295076
    ...
