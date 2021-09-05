#!/bin/bash
# resume-single-node.sh
#   Single Node GPT-2 Small `Resume from Checkpoint` Debugging. Uses the DeepSpeed ZeRO-2 Optimizer,
#   Per-Device Batch Size of 16.

# Constants
CONFIG="--config conf/gpt2-benchmark-config.yaml"
INFRA="--nnodes 1 --nproc_per_node 8"

# Batch Size
D_BSZ_16="--training_arguments.fp16 true --training_arguments.per_device_train_batch_size 16"

# DeepSpeed Training Configuration
DS_Z2="--training_arguments.deepspeed conf/deepspeed/z2-conf.json"

# Set DeepSpeed Launcher Parameters
DISTRIBUTED_ARGS="--num_gpus 8 --num_nodes 1"

# ---

# Single-Node DS-Z2, Linear LR Schedule, Device BSZ = 16 --> Cleanup --> Sleep
deepspeed "$DISTRIBUTED_ARGS" train.py "$CONFIG" "$INFRA" "$D_BSZ_16" "$DS_Z2"
pkill -f "train.py"
sleep 3
