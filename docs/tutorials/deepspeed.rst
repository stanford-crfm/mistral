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
--------------------

When running Deep Speed and Hugging Face, it is necessary to specify a collection of training settings in a DeepSpeed
json config file. These settings will be used to create the final ``TrainingArguments`` object for model training,
and include such things as what optimizer or scheduler to use.

An example json config file is available at ``conf/deepspeed/z1-conf.json``: ::

    {
      "optimizer": {
        "type": "AdamW",
        "params": {
          "lr": 0.0006,
          "betas": [
            0.9,
            0.95
          ],
          "eps": 1e-8,
          "weight_decay": 0.1
        }
      },
    
      "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
          "total_num_steps": 400000,
          "warmup_max_lr": 0.0006,
          "warmup_num_steps": 4000
        }
      },
    
      "zero_optimization": {
        "stage": 1,
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "cpu_offload": false
      }
    }

Launching A Training Run
------------------------

The following command (run on machine1) will launch training across your cluster: ::

    conda activate mistral
    cd mistral
    deepspeed --num_gpus 8 --num_nodes 2 --master_addr machine1 train.py --config conf/hello-world.yaml --nnodes 2 --nproc_per_node 8 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 4 --training_arguments.deepspeed conf/deepspeed/z1-conf.json --run_id hello-world-multi-node > hello-world-multi-node.out 2> hello-world-multi-node.err

This assumes that the appropriate hostfile is set up at ``/job/hostfile`` on ``machine1``.

You should see the following output if training is running as expected: ::

    machine2: {'loss': 6.5859, 'learning_rate': 0.0003537728376673855, 'activations/layer0_attention_weight_max': 6.225409030914307, 'activations/layer0_attention_weight_min': -6.8558735847473145, 'activations/layer1_attention_weight_max': 2.5137383937835693, 'activations/layer1_attention_weight_min': -3.4525303840637207, 'activations/layer2_attention_weight_max': 1.65605628490448, 'activations/layer2_attention_weight_min': -2.03672194480896, 'activations/layer3_attention_weight_max': 1.8134779930114746, 'activations/layer3_attention_weight_min': -1.6253358125686646, 'activations/layer4_attention_weight_max': 1.5045760869979858, 'activations/layer4_attention_weight_min': -1.482985496520996, 'activations/layer5_attention_weight_max': 3.2311043739318848, 'activations/layer5_attention_weight_min': -2.9691357612609863, 'activations/layer6_attention_weight_max': 5.682344913482666, 'activations/layer6_attention_weight_min': -4.275859355926514, 'activations/layer7_attention_weight_max': 0.7755581736564636, 'activations/layer7_attention_weight_min': -0.6805652379989624, 'activations/layer8_attention_weight_max': 1.4897541999816895, 'activations/layer8_attention_weight_min': -1.216135025024414, 'activations/layer9_attention_weight_max': 1.1379717588424683, 'activations/layer9_attention_weight_min': -1.412354826927185, 'activations/layer10_attention_weight_max': 2.4922404289245605, 'activations/layer10_attention_weight_min': -2.0055084228515625, 'activations/layer11_attention_weight_max': 1.4722517728805542, 'activations/layer11_attention_weight_min': -1.2682315111160278, 'epoch': 0.7}
    machine1: [2021-07-01 01:24:59,832] [INFO] [logging.py:60:log_dist] [Rank 0] step=150, skipped=17, lr=[0.0003537728376673855], mom=[[0.9, 0.95]]
    machine1: [2021-07-01 01:24:59,852] [INFO] [timer.py:154:stop] 0/1200, SamplesPerSec=463.8644895928809
    machine1: {'loss': 6.591, 'learning_rate': 0.0003537728376673855, 'activations/layer0_attention_weight_max': 5.9575395584106445, 'activations/layer0_attention_weight_min': -7.12982177734375, 'activations/layer1_attention_weight_max': 2.775029182434082, 'activations/layer1_attention_weight_min': -3.474602222442627, 'activations/layer2_attention_weight_max': 1.8722176551818848, 'activations/layer2_attention_weight_min': -1.927580714225769, 'activations/layer3_attention_weight_max': 1.8707917928695679, 'activations/layer3_attention_weight_min': -1.787396788597107, 'activations/layer4_attention_weight_max': 1.47317636013031, 'activations/layer4_attention_weight_min': -1.391649603843689, 'activations/layer5_attention_weight_max': 3.2698564529418945, 'activations/layer5_attention_weight_min': -2.83353328704834, 'activations/layer6_attention_weight_max': 5.822953701019287, 'activations/layer6_attention_weight_min': -4.2001142501831055, 'activations/layer7_attention_weight_max': 0.782840371131897, 'activations/layer7_attention_weight_min': -0.7528175115585327, 'activations/layer8_attention_weight_max': 1.5653538703918457, 'activations/layer8_attention_weight_min': -1.1807199716567993, 'activations/layer9_attention_weight_max': 1.1230956315994263, 'activations/layer9_attention_weight_min': -1.4319841861724854, 'activations/layer10_attention_weight_max': 2.5261030197143555, 'activations/layer10_attention_weight_min': -1.9104121923446655, 'activations/layer11_attention_weight_max': 1.4361441135406494, 'activations/layer11_attention_weight_min': -1.2555559873580933, 'epoch': 0.7}
    machine1: [2021-07-01 01:25:12,365] [INFO] [engine.py:1680:_save_zero_checkpoint] zero checkpoint saved hello-world/runs/hello-world-multi-node/checkpoint-150/global_step150/zero_pp_rank_5_mp_rank_00optim_states.pt
    machine2: [2021-07-01 01:25:12,576] [INFO] [engine.py:1680:_save_zero_checkpoint] zero checkpoint saved hello-world/runs/hello-world-multi-node/checkpoint-150/global_step150/zero_pp_rank_9_mp_rank_00optim_states.pt
    machine2: [2021-07-01 01:25:12,699] [INFO] [engine.py:1680:_save_zero_checkpoint] zero checkpoint saved hello-world/runs/hello-world-multi-node/checkpoint-150/global_step150/zero_pp_rank_15_mp_rank_00optim_states.pt
    machine2: [2021-07-01 01:25:12,727] [INFO] [engine.py:1680:_save_zero_checkpoint] zero checkpoint saved hello-world/runs/hello-world-multi-node/checkpoint-150/global_step150/zero_pp_rank_14_mp_rank_00optim_states.pt
    machine2: [2021-07-01 01:25:12,768] [INFO] [engine.py:1680:_save_zero_checkpoint] zero checkpoint saved hello-world/runs/hello-world-multi-node/checkpoint-150/global_step150/zero_pp_rank_13_mp_rank_00optim_states.pt
    machine2: [2021-07-01 01:25:12,772] [INFO] [engine.py:1680:_save_zero_checkpoint] zero checkpoint saved hello-world/runs/hello-world-multi-node/checkpoint-150/global_step150/zero_pp_rank_10_mp_rank_00optim_states.pt
    machine2: [2021-07-01 01:25:12,772] [INFO] [engine.py:1680:_save_zero_checkpoint] zero checkpoint saved hello-world/runs/hello-world-multi-node/checkpoint-150/global_step150/zero_pp_rank_11_mp_rank_00optim_states.pt
    machine2: [2021-07-01 01:25:12,774] [INFO] [engine.py:1680:_save_zero_checkpoint] zero checkpoint saved hello-world/runs/hello-world-multi-node/checkpoint-150/global_step150/zero_pp_rank_12_mp_rank_00optim_states.pt
    machine2: [2021-07-01 01:25:12,775] [INFO] [engine.py:1680:_save_zero_checkpoint] zero checkpoint saved hello-world/runs/hello-world-multi-node/checkpoint-150/global_step150/zero_pp_rank_8_mp_rank_00optim_states.pt
    machine1: [2021-07-01 01:25:12,905] [INFO] [engine.py:1680:_save_zero_checkpoint] zero checkpoint saved hello-world/runs/hello-world-multi-node/checkpoint-150/global_step150/zero_pp_rank_7_mp_rank_00optim_states.pt
    machine1: [2021-07-01 01:25:12,908] [INFO] [engine.py:1680:_save_zero_checkpoint] zero checkpoint saved hello-world/runs/hello-world-multi-node/checkpoint-150/global_step150/zero_pp_rank_3_mp_rank_00optim_states.pt
    machine1: [2021-07-01 01:25:12,911] [INFO] [engine.py:1680:_save_zero_checkpoint] zero checkpoint saved hello-world/runs/hello-world-multi-node/checkpoint-150/global_step150/zero_pp_rank_2_mp_rank_00optim_states.pt
    machine1: [2021-07-01 01:25:12,912] [INFO] [engine.py:1680:_save_zero_checkpoint] zero checkpoint saved hello-world/runs/hello-world-multi-node/checkpoint-150/global_step150/zero_pp_rank_4_mp_rank_00optim_states.pt
    machine1: [2021-07-01 01:25:12,912] [INFO] [engine.py:1680:_save_zero_checkpoint] zero checkpoint saved hello-world/runs/hello-world-multi-node/checkpoint-150/global_step150/zero_pp_rank_6_mp_rank_00optim_states.pt
    machine1: [2021-07-01 01:25:12,914] [INFO] [engine.py:1680:_save_zero_checkpoint] zero checkpoint saved hello-world/runs/hello-world-multi-node/checkpoint-150/global_step150/zero_pp_rank_1_mp_rank_00optim_states.pt
    machine1: [2021-07-01 01:25:12,915] [INFO] [engine.py:1680:_save_zero_checkpoint] zero checkpoint saved hello-world/runs/hello-world-multi-node/checkpoint-150/global_step150/zero_pp_rank_0_mp_rank_00optim_states.pt
    machine1: [2021-07-01 01:25:14,263] [INFO] [timer.py:154:stop] 0/1210, SamplesPerSec=464.0203879208643
    machine1: [2021-07-01 01:25:15,594] [INFO] [timer.py:154:stop] 0/1220, SamplesPerSec=464.1906754491529
    machine1: [2021-07-01 01:25:16,914] [INFO] [timer.py:154:stop] 0/1230, SamplesPerSec=464.3912136700007
    machine1: [2021-07-01 01:25:18,362] [INFO] [timer.py:154:stop] 0/1240, SamplesPerSec=464.24566491771554
    machine1: [2021-07-01 01:25:19,695] [INFO] [timer.py:154:stop] 0/1250, SamplesPerSec=464.42727423856235
    machine1: [2021-07-01 01:25:21,014] [INFO] [timer.py:154:stop] 0/1260, SamplesPerSec=464.6226536935847
    machine1: [2021-07-01 01:25:22,348] [INFO] [timer.py:154:stop] 0/1270, SamplesPerSec=464.77860039369176
    machine1: [2021-07-01 01:25:23,771] [INFO] [logging.py:60:log_dist] [Rank 0] step=160, skipped=17, lr=[0.0003590172361350027], mom=[[0.9, 0.95]]
    machine1: [2021-07-01 01:25:23,797] [INFO] [timer.py:154:stop] 0/1280, SamplesPerSec=464.6283453267044
    machine1: [2021-07-01 01:25:25,134] [INFO] [timer.py:154:stop] 0/1290, SamplesPerSec=464.78115507602973
    machine1: [2021-07-01 01:25:26,439] [INFO] [timer.py:154:stop] 0/1300, SamplesPerSec=465.0059604093444
    machine1: [2021-07-01 01:25:27,774] [INFO] [timer.py:154:stop] 0/1310, SamplesPerSec=465.15007672116036
    machine1: [2021-07-01 01:25:29,223] [INFO] [timer.py:154:stop] 0/1320, SamplesPerSec=464.99670383095474
    machine1: [2021-07-01 01:25:30,573] [INFO] [timer.py:154:stop] 0/1330, SamplesPerSec=465.10801140307893
    machine1: [2021-07-01 01:25:31,898] [INFO] [timer.py:154:stop] 0/1340, SamplesPerSec=465.2706935850331
    machine1: [2021-07-01 01:25:33,234] [INFO] [timer.py:154:stop] 0/1350, SamplesPerSec=465.4060267775872
    machine1: [2021-07-01 01:25:34,665] [INFO] [logging.py:60:log_dist] [Rank 0] step=170, skipped=17, lr=[0.0003639070036718917], mom=[[0.9, 0.95]]
    machine1: [2021-07-01 01:25:34,684] [INFO] [timer.py:154:stop] 0/1360, SamplesPerSec=465.25861085162535
    machine1: [2021-07-01 01:25:36,013] [INFO] [timer.py:154:stop] 0/1370, SamplesPerSec=465.4240351272059
    machine1: [2021-07-01 01:25:37,336] [INFO] [timer.py:154:stop] 0/1380, SamplesPerSec=465.5869948597482
    machine1: [2021-07-01 01:25:38,649] [INFO] [timer.py:154:stop] 0/1390, SamplesPerSec=465.7649513405123
    machine1: [2021-07-01 01:25:40,096] [INFO] [timer.py:154:stop] 0/1400, SamplesPerSec=465.62227999393195
    machine1: [2021-07-01 01:25:41,429] [INFO] [timer.py:154:stop] 0/1410, SamplesPerSec=465.76990823125493
    machine1: [2021-07-01 01:25:42,740] [INFO] [timer.py:154:stop] 0/1420, SamplesPerSec=465.9543142797093
    machine1: [2021-07-01 01:25:44,067] [INFO] [timer.py:154:stop] 0/1430, SamplesPerSec=466.1050477719339
    machine1: [2021-07-01 01:25:45,492] [INFO] [logging.py:60:log_dist] [Rank 0] step=180, skipped=17, lr=[0.000368487078460078], mom=[[0.9, 0.95]]
    machine1: [2021-07-01 01:25:45,509] [INFO] [timer.py:154:stop] 0/1440, SamplesPerSec=465.97806156640763
    machine1: [2021-07-01 01:25:46,820] [INFO] [timer.py:154:stop] 0/1450, SamplesPerSec=466.1725567389113
    machine1: [2021-07-01 01:25:48,146] [INFO] [timer.py:154:stop] 0/1460, SamplesPerSec=466.31161256295076
    machine1: [2021-07-01 01:25:49,462] [INFO] [timer.py:154:stop] 0/1470, SamplesPerSec=466.47593878624264
    machine1: [2021-07-01 01:25:50,898] [INFO] [timer.py:154:stop] 0/1480, SamplesPerSec=466.35923848990143
    machine1: [2021-07-01 01:25:52,229] [INFO] [timer.py:154:stop] 0/1490, SamplesPerSec=466.49680479051597
    machine1: [2021-07-01 01:25:53,544] [INFO] [timer.py:154:stop] 0/1500, SamplesPerSec=466.6604536243473
    machine1: [2021-07-01 01:25:54,858] [INFO] [timer.py:154:stop] 0/1510, SamplesPerSec=466.8258464618343
    machine1: [2021-07-01 01:25:56,253] [INFO] [logging.py:60:log_dist] [Rank 0] step=190, skipped=17, lr=[0.0003727943635336901], mom=[[0.9, 0.95]]
    machine1: [2021-07-01 01:25:56,270] [INFO] [timer.py:154:stop] 0/1520, SamplesPerSec=466.7695464652509
    machine1: [2021-07-01 01:25:57,591] [INFO] [timer.py:154:stop] 0/1530, SamplesPerSec=466.92683700014027
    machine1: [2021-07-01 01:25:58,923] [INFO] [timer.py:154:stop] 0/1540, SamplesPerSec=467.0475699528104
    machine1: [2021-07-01 01:26:00,248] [INFO] [timer.py:154:stop] 0/1550, SamplesPerSec=467.18073380861307
    machine1: [2021-07-01 01:26:01,711] [INFO] [timer.py:154:stop] 0/1560, SamplesPerSec=467.0128849288976
    machine1: [2021-07-01 01:26:03,039] [INFO] [timer.py:154:stop] 0/1570, SamplesPerSec=467.1410597602756
    machine1: [2021-07-01 01:26:04,376] [INFO] [timer.py:154:stop] 0/1580, SamplesPerSec=467.2457988670264
    machine1: [2021-07-01 01:26:05,739] [INFO] [timer.py:154:stop] 0/1590, SamplesPerSec=467.2957462415879

