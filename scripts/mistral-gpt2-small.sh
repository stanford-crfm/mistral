# mistral-gpt2-small.sh
#   Mistral GPT-2 Small Full Run with the DeepSpeed ZeRO-2 Optimizer, Per-Device Batch Size of 16.

# Constants
CONFIG="--config conf/gpt2-mistral-config.yaml"
INFRA="--nnodes 2 --nproc_per_node 8"

# Batch Size
D_BSZ_16="--training_arguments.fp16 true --training_arguments.per_device_train_batch_size 16"

# DeepSpeed Training Configuration
DS_Z2="--training_arguments.deepspeed conf/deepspeed/z2-conf.json"

# Random Seeds -- Aurora :: 21, Blizzard :: 49, Cyclone :: 81
AURORA="--seed 21"
BLIZZARD="--seed 49"
CYCLONE="--seed 81"

# Set DeepSpeed Launcher Parameters
MASTER_ADDR=sphinx1.stanford.edu
MASTER_PORT=7000
DISTRIBUTED_ARGS="--num_gpus 8 --num_nodes 2 --master_addr $MASTER_ADDR"

# ---

# Multi-Node DS-Z2, Linear LR Schedule, Device BSZ = 16 --> Cleanup --> Sleep =>> Seed 21
# deepspeed $DISTRIBUTED_ARGS train.py $CONFIG $INFRA $D_BSZ_16 $AURORA $DS_Z2 --run_id aurora-gpt2-small-x21
# pkill -f "train.py"
# sleep 3

# Multi-Node DS-Z2, Linear LR Schedule, Device BSZ = 16 --> Cleanup --> Sleep =>> Seed 49
# deepspeed $DISTRIBUTED_ARGS train.py $CONFIG $INFRA $D_BSZ_16 $BLIZZARD $DS_Z2 --run_id blizzard-gpt2-small-x49
# pkill -f "train.py"
# sleep 3

# Multi-Node DS-Z2, Linear LR Schedule, Device BSZ = 16 --> Cleanup --> SLeed =>> Seed 81
deepspeed $DISTRIBUTED_ARGS train.py $CONFIG $INFRA $D_BSZ_16 $CYCLONE $DS_Z2 --run_id cyclone-gpt2-small-x81
pkill -f "train.py"
sleep 3
