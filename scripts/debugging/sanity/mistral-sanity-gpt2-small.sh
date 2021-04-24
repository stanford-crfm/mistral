# mistral-sanity-gpt2-small.sh
#   Mistral Sanity Check -- GPT-2 Small 4K Step Run with the DeepSpeed ZeRO-2 Optimizer, Per-Device Batch Size of 16.

# Constants
CONFIG="--config conf/gpt2-debug-conf.yaml"
INFRA="--nnodes 2 --nproc_per_node 8"

# Batch Size
D_BSZ_16="--training_arguments.fp16 true --training_arguments.per_device_train_batch_size 16"

# DeepSpeed Training Configuration
DS_Z2="--training_arguments.deepspeed conf/deepspeed/z2-debug-conf.json"

# Random Seeds -- Athos :: 21, Blizzard :: 49, Cyclone :: 81
ATHOS="--seed 21"
PORTHOS="--seed 49"
ARAMIS="--seed 81"

# Set DeepSpeed Launcher Parameters
MASTER_ADDR=sphinx1.stanford.edu
MASTER_PORT=7000
DISTRIBUTED_ARGS="--num_gpus 8 --num_nodes 2 --master_addr $MASTER_ADDR"

# Resume
RESUME="--resume true"

# ---

# Multi-Node DS-Z2, Linear LR Schedule, Device BSZ = 16 --> Cleanup --> Sleep =>> Seed 21
deepspeed $DISTRIBUTED_ARGS train.py $CONFIG $INFRA $D_BSZ_16 $ATHOS $DS_Z2 --run_id athos-gpt2-small-debug-x21
pkill -f "train.py"
sleep 3

## Multi-Node DS-Z2, Linear LR Schedule, Device BSZ = 16 --> Cleanup --> Sleep =>> Seed 21 -- REPLICATION
#deepspeed $DISTRIBUTED_ARGS train.py $CONFIG $INFRA $D_BSZ_16 $ATHOS $DS_Z2 --run_id athos-replica-gpt2-small-debug-x21
#pkill -f "train.py"
#sleep 3
#
## Multi-Node DS-Z2, Linear LR Schedule, Device BSZ = 16 --> Cleanup --> Sleep =>> Seed 49
#deepspeed $DISTRIBUTED_ARGS train.py $CONFIG $INFRA $D_BSZ_16 $PORTHOS $DS_Z2 --run_id porthos-gpt2-small-debug-x49
#pkill -f "train.py"
#sleep 3
#
## Multi-Node DS-Z2, Linear LR Schedule, Device BSZ = 16 --> Cleanup --> Sleep =>> Seed 81
#deepspeed $DISTRIBUTED_ARGS train.py $CONFIG $INFRA $D_BSZ_16 $ARAMIS $DS_Z2 --run_id aramis-gpt2-small-debug-x81
#pkill -f "train.py"
#sleep 3
