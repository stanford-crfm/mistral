#!/bin/bash
# fairscale-single.sh
#   Benchmarking Script for Single-Node FairScale Trainer, verifying multi-stage sharded training (ZeRO 1, 2, and 3)
#   with and without gradient checkpointing.

# Constants
CONFIG="--config conf/gpt2-benchmark-config.yaml"
INFRA="--nnodes 1 --nproc_per_node 8"
GC="--model.gradient_checkpointing true"

# A Few Choices for Batch Size
D_BSZ_8="--training_arguments.fp16 true --training_arguments.per_device_train_batch_size 8"
D_BSZ_16="--training_arguments.fp16 true --training_arguments.per_device_train_batch_size 16"
D_BSZ_32="--training_arguments.fp16 true --training_arguments.per_device_train_batch_size 32"

# FairScale Parameter
FS_Z1="--training_arguments.sharded_ddp simple"
FS_Z2="--training_arguments.sharded_ddp zero_dp_2+auto_wrap"
FS_Z3="--training_arguments.sharded_ddp zero_dp_3+auto_wrap"

# Setup Distributed Launch Parameters -- We probably don't need Master Address/Port, but including for completeness
MASTER_ADDR=sphinx1.stanford.edu
MASTER_PORT=7000
WORLD_SIZE=8
DISTRIBUTED_ARGS="--nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr $MASTER_ADDR"
LAUNCHER="torch.distributed.launch"

# ---

# Single Node FS-Z1, No GC, Device BSZ = 8 --> Cleanup --> Sleep
#python -m $LAUNCHER $DISTRIBUTED_ARGS train.py $CONFIG $INFRA $D_BSZ_8 $FS_Z1 --run_id 29-fs=z1-n=1-g=8-fp16-dbsz=8
#pkill -f "train.py"
#sleep 3

# Single Node FS-Z1, No GC, Device BSZ = 16 --> Cleanup --> Sleep
#python -m $LAUNCHER $DISTRIBUTED_ARGS train.py $CONFIG $INFRA $D_BSZ_16 $FS_Z1 --run_id 30-fs=z1-n=1-g=8-fp16-dbsz=16
#pkill -f "train.py"
#sleep 3

# Single Node FS-Z1, ++GC, Device BSZ = 32 --> Cleanup --> Sleep
python -m $LAUNCHER "$DISTRIBUTED_ARGS" train.py "$CONFIG" "$INFRA" "$GC" "$D_BSZ_32" "$FS_Z1" --run_id 31-fs=z1-n=1-g=8-gc-fp16-dbsz=32
pkill -f "train.py"
sleep 3

# Single Node FS-Z2, No GC, Device BSZ = 8 --> Cleanup --> Sleep
python -m $LAUNCHER "$DISTRIBUTED_ARGS" train.py "$CONFIG" "$INFRA" "$D_BSZ_8" "$FS_Z2" --run_id 32-fs=z2-n=1-g=8-fp16-dbsz=8
pkill -f "train.py"
sleep 3

# Single Node FS-Z2, No GC, Device BSZ = 16 --> Cleanup --> Sleep
python -m $LAUNCHER "$DISTRIBUTED_ARGS" train.py "$CONFIG" "$INFRA" "$D_BSZ_16" "$FS_Z2" --run_id 33-fs=z2-n=1-g=8-fp16-dbsz=16
pkill -f "train.py"
sleep 3

# Single Node FS-Z2, ++GC, Device BSZ = 32 --> Cleanup --> Sleep
python -m $LAUNCHER "$DISTRIBUTED_ARGS" train.py "$CONFIG" "$INFRA" "$GC" "$D_BSZ_32" "$FS_Z2" --run_id 34-fs=z2-n=1-g=8-gc-fp16-dbsz=32
pkill -f "train.py"
sleep 3

# Single Node FS-Z3, No GC, Device BSZ = 8 --> Cleanup --> Sleep
python -m $LAUNCHER "$DISTRIBUTED_ARGS" train.py "$CONFIG" "$INFRA" "$D_BSZ_8" "$FS_Z3" --run_id 35-fs=z3-n=1-g=8-fp16-dbsz=8
pkill -f "train.py"
sleep 3

# Single Node FS-Z1, No GC, Device BSZ = 16 --> Cleanup --> Sleep
python -m $LAUNCHER "$DISTRIBUTED_ARGS" train.py "$CONFIG" "$INFRA" "$D_BSZ_16" "$FS_Z3" --run_id 36-fs=z3-n=1-g=8-fp16-dbsz=16
pkill -f "train.py"
sleep 3

# Single Node FS-Z3, ++GC, Device BSZ = 32 --> Cleanup --> Sleep
python -m $LAUNCHER "$DISTRIBUTED_ARGS" train.py "$CONFIG" "$INFRA" "$GC" "$D_BSZ_32" "$FS_Z3" --run_id 37-fs=z3-n=1-g=8-gc-fp16-dbsz=32
pkill -f "train.py"
sleep 3
