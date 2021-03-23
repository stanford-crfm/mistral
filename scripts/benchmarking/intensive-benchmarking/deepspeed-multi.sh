# deepspeed-multi.sh
#   Benchmarking Script for Intensive Multi-Node DeepSpeed Trainer, verifying multi-stage sharded training (ZeRO 1, 2)
#   without gradient checkpointing.

# Constants
CONFIG="--config conf/gpt2-intensive-config.yaml"
INFRA="--nnodes 2 --nproc_per_node 8"

# A Few Choices for Batch Size
D_BSZ_8="--training_arguments.fp16 true --training_arguments.per_device_train_batch_size 8"
D_BSZ_16="--training_arguments.fp16 true --training_arguments.per_device_train_batch_size 16"

# DeepSpeed Configurations
DS_Z1="--training_arguments.deepspeed conf/deepspeed/z1-conf.json"
DS_Z2="--training_arguments.deepspeed conf/deepspeed/z2-conf.json"

# Set DeepSpeed Launcher Parameters
MASTER_ADDR=sphinx1.stanford.edu
MASTER_PORT=7000
DISTRIBUTED_ARGS="--num_gpus 8 --num_nodes 2 --master_addr $MASTER_ADDR"

# ---

# Multi-Node Node DS-Z1, No GC, Device BSZ = 8 --> Cleanup --> Sleep
deepspeed $DISTRIBUTED_ARGS train.py $CONFIG $INFRA $D_BSZ_8 $DS_Z1 --run_id echo-ds=z1-n=2-g=8-fp16-dbsz=8
pkill -f "train.py"
sleep 3

# Multi-Node DS-Z1, No GC, Device BSZ = 16 --> Cleanup --> Sleep
deepspeed $DISTRIBUTED_ARGS train.py $CONFIG $INFRA $D_BSZ_16 $DS_Z1 --run_id foxtrot-ds=z1-n=2-g=8-fp16-dbsz=16
pkill -f "train.py"
sleep 3

# Multi-Node DS-Z2, No GC, Device BSZ = 8 --> Cleanup --> Sleep
deepspeed $DISTRIBUTED_ARGS train.py $CONFIG $INFRA $D_BSZ_8 $DS_Z2 --run_id golf-ds=z2-n=2-g=8-fp16-dbsz=8
pkill -f "train.py"
sleep 3

# Multi-Node DS-Z2, No GC, Device BSZ = 16 --> Cleanup --> Sleep
deepspeed $DISTRIBUTED_ARGS train.py $CONFIG $INFRA $D_BSZ_16 $DS_Z2 --run_id hotel-ds=z2-n=1-g=8-fp16-dbsz=16
pkill -f "train.py"
sleep 3
