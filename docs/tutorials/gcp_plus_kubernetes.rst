Training A Model With GCP + Kubernetes
======================================

This tutorial will walk through training a model on Google Cloud with `Kubernetes <https://kubernetes.io/>`_.

Preliminaries
-------------

We will assume you have a `Google Cloud <https://cloud.google.com/>`_ account set up already.

For this tutorial you will need to install the `gcloud <https://cloud.google.com/sdk/docs/downloads-interactive>`_ and `kubectl <https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/>`_ command line utilities.

Creating A Kubernetes Cluster
-----------------------------

We will now create a basic Kubernetes cluster with

* 2 main machines for managing the overall cluster
* A node pool that will create GPU machines when jobs are submitted
* A 1 TB persistent volume the machines can use for data storage

This tutorial describes the Kubernetes set up we used when training models, but of course you can customize this set up as you see fit for your situation.

On the Google Cloud Console, go to the "Kubernetes Engine (Clusters)" page. Click on "CREATE".

Choose the "GKE Standard" option.

On the "Cluster basics" page, set the name of your cluster and choose the zone you want for your cluster.
You will want to choose a zone with A100 machines such as ``us-central-1a``. In our working example, we
are calling the cluster "tutorial-cluster".

In the "NODE POOLS" section, change the name of the default pool to "main". Click on "Nodes" and change
the machine type to "e2-standard".

When finished, click "CREATE" at the bottom and the Kubernetes cluster will be created.

Adding A Node Pool To Your Cluster
----------------------------------

When the cluster has finished, you can click on its name and see cluster info. Click on "NODES". You
will be brought to a page that shows the node pools for the cluster and the nodes.

At the top of the screen click on "ADD NODE POOL". Set the name of the node pool to "node-1". Set the number of nodes
to 0 and check "Enable autoscaling". With autoscaling, Kubernetes will launch nodes when you submit jobs. When there
are no active jobs, you will have no active machines running. When a job is submitted, the node pool will scale up to
meet the needs of the job. Set the minimum number of nodes to 0 and set the maximum to the maximum number of GPU machines
you want running at any given time.

Click on "Nodes" on the left sidebar, and customize the types of machines the node pool will use. This tutorial will
assume you are running on NVIDIA Tesla A100's with 8 GPUs, and the default machine configuration.

When finished, click "CREATE". You should see "pool-1" show up in your list of node pools.

Creating The Persistent Volume
------------------------------

The next step is to create the persistent volume. We will create a 1 TB volume, though you may want more space.

You will need to have installed ``gcloud`` and ``kubectl``. Instructions for installing them can be found in
the "Preliminaries" section above.

First create the disk: ::

    gcloud compute disks create --size=1000GB --zone=us-central1-a --type pd-ssd pd-tutorial

Then set up the nfs server (from the ``gcp`` directory in the mistral repo): ::

    kubectl apply -f nfs/nfs_server.yaml
    kubectl apply -f nfs/nfs_service.yaml
    kubectl get services

You should see output like this: ::

    NAME         TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)                      AGE
    kubernetes   ClusterIP   10.48.0.1      <none>        443/TCP                      135m
    nfs-server   ClusterIP   10.48.14.252   <none>        2049/TCP,20048/TCP,111/TCP   11s

Extract the IP address for the nfs-server (10.48.14.252 in the example output), and update the ``nfs/nfs_pv.yaml``
file. Then run: ::

    kubectl apply -f nfs_pv.yaml

You should see output like: ::

    NAME                          READY   STATUS    RESTARTS   AGE
    nfs-server-697fbd7f8d-pvsdb   1/1     Running   0          14m

The persistent volume should now be ready for usage.

Installing Drivers
------------------

Run this command to set up the GPU drivers. If you do not run this command, nodes will be unable to use GPUs. ::

    kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml




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

    cd mistral
    conda activate mistral
    deepspeed --num_gpus 8 --num_nodes 2 --master_addr machine1 train.py --config conf/tutorial-gpt2-micro.yaml --nnodes 2 --nproc_per_node 8 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 4 --training_arguments.deepspeed conf/deepspeed/z1-conf.json --run_id tutorial-gpt2-micro-multi-node > tutorial-gpt2-micro-multi-node.out 2> tutorial-gpt2-micro-multi-node.err

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
