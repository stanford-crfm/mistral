#!/bin/bash
# mistral-gpt2-medium.sh
#   Mistral GPT-2 Medium Dry-Runs with the DeepSpeed ZeRO-2 Optimizer, Per-Device Batch Size of 16/8/4.

# Constants
CONFIG="--config conf/archive/partial-checkpointing/gpt2-mistral-medium-gcheck-config.yaml"
INFRA="--nnodes 2 --nproc_per_node 8"

# Batch Size
D_BSZ_4="--training_arguments.fp16 true --training_arguments.per_device_train_batch_size 4"
D_BSZ_8="--training_arguments.fp16 true --training_arguments.per_device_train_batch_size 8"
D_BSZ_32="--training_arguments.fp16 true --training_arguments.per_device_train_batch_size 32"

# Gradient Checkpointing
FULL_GC="--model.gradient_checkpointing true --model.gc_checkpoint_every 1"
GC_6="--model.gradient_checkpointing true --model.gc_checkpoint_every 6"
GC_8="--model.gradient_checkpointing true --model.gc_checkpoint_every 8"
GC_12="--model.gradient_checkpointing true --model.gc_checkpoint_every 12"

# DeepSpeed Training Configuration
DS_Z2="--training_arguments.deepspeed conf/deepspeed/z2-conf.json"

# Set DeepSpeed Launcher Parameters
MASTER_ADDR=sphinx1.stanford.edu
MASTER_PORT=7000
DISTRIBUTED_ARGS="--num_gpus 8 --num_nodes 2 --master_addr $MASTER_ADDR"

# ---

## Multi-Node DS-Z2, Linear LR Schedule, Device BSZ = 4 --> Cleanup --> Sleep
#deepspeed $DISTRIBUTED_ARGS train.py $CONFIG $INFRA $D_BSZ_4 $DS_Z2 --run_id gpt2-medium-dry-run-dbsz=4-no-gc
#pkill -f "train.py"
#sleep 3
#
## Multi-Node DS-Z2, Linear LR Schedule, Device BSZ = 8 --> Cleanup --> Sleep
#deepspeed $DISTRIBUTED_ARGS train.py $CONFIG $INFRA $D_BSZ_8 $DS_Z2 --run_id gpt2-medium-dry-run-dbsz=8-no-gc
#pkill -f "train.py"
#sleep 3
#
## Multi-Node DS-Z2, Linear LR Schedule, Device BSZ = 32 (+ GC=ALL) --> Cleanup --> Sleep
#deepspeed $DISTRIBUTED_ARGS train.py $CONFIG $INFRA $FULL_GC $D_BSZ_32 $DS_Z2 --run_id gpt2-medium-dry-run-dbsz=32-gc=all
#pkill -f "train.py"
#sleep 3
#
## Multi-Node DS-Z2, Linear LR Schedule, Device BSZ = 8 (+ GC=6) --> Cleanup --> Sleep
deepspeed "$DISTRIBUTED_ARGS" train.py "$CONFIG" "$INFRA" "$GC_6" "$D_BSZ_8" "$DS_Z2" --run_id gpt2-medium-dry-run-dbsz=8-gc-every=6-gamma
pkill -f "train.py"
sleep 3

## Multi-Node DS-Z2, Linear LR Schedule, Device BSZ = 8 (+ GC=8) --> Cleanup --> Sleep
#deepspeed $DISTRIBUTED_ARGS train.py $CONFIG $INFRA $GC_8 $D_BSZ_8 $DS_Z2 --run_id gpt2-medium-dry-run-dbsz=8-gc=8-evenly
#pkill -f "train.py"
#sleep 3

## Multi-Node DS-Z2, Linear LR Schedule, Device BSZ = 8 (+ GC=12) --> Cleanup --> Sleep
#deepspeed $DISTRIBUTED_ARGS train.py $CONFIG $INFRA $GC_12 $D_BSZ_8 $DS_Z2 --run_id gpt2-medium-dry-run-dbsz=8-gc=12-evenly
#pkill -f "train.py"
#sleep 3
