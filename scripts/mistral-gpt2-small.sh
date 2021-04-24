# mistral-gpt2-small.sh
#   Mistral GPT-2 Small Full Run with the DeepSpeed ZeRO-2 Optimizer, Per-Device Batch Size of 16.

# Constants
CONFIG="--config conf/gpt2-mistral-small-config.yaml"
INFRA="--nnodes 2 --nproc_per_node 8"

# Batch Size
D_BSZ_16="--training_arguments.fp16 true --training_arguments.per_device_train_batch_size 16"

# DeepSpeed Training Configuration
DS_Z2="--training_arguments.deepspeed conf/deepspeed/z2-conf.json"

# Random Seeds -- Alias :: 21, Battlestar :: 49, Caprica :: 81
ALIAS="--seed 21"
BATTLESTAR="--seed 49"
CAPRICA="--seed 81"

# TODO mistral-gpt2-small.sh.A :: Add 7 other seeds + Control-Flow Logic!

# Set DeepSpeed Launcher Parameters
MASTER_ADDR=sphinx1.stanford.edu
MASTER_PORT=7000
DISTRIBUTED_ARGS="--num_gpus 8 --num_nodes 2 --master_addr $MASTER_ADDR"

# Resume
RESUME="--resume true"

# ---

# Multi-Node DS-Z2, Linear LR Schedule, Device BSZ = 16 --> Cleanup --> Sleep =>> Seed 21
# deepspeed $DISTRIBUTED_ARGS train.py $CONFIG $INFRA $D_BSZ_16 $ALIAS $DS_Z2 --run_id alias-gpt2-small-x21
# pkill -f "train.py"
# sleep 3

# Multi-Node DS-Z2, Linear LR Schedule, Device BSZ = 16 --> Cleanup --> Sleep =>> Seed 49
# deepspeed $DISTRIBUTED_ARGS train.py $CONFIG $INFRA $D_BSZ_16 $BATTLESTAR $DS_Z2 --run_id battlestar-gpt2-small-x49
# pkill -f "train.py"
# sleep 3

# Multi-Node DS-Z2, Linear LR Schedule, Device BSZ = 16 --> Cleanup --> Seed =>> Seed 81 (+ Resume!)
deepspeed $DISTRIBUTED_ARGS train.py $CONFIG $INFRA $D_BSZ_16 $CAPRICA $RESUME $DS_Z2 --run_id caprica-gpt2-small-x81
pkill -f "train.py"
sleep 3
