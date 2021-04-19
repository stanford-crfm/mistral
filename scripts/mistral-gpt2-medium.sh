# mistral-gpt2-medium.sh
#   Mistral GPT-2 Medium Full Run with the DeepSpeed ZeRO-2 Optimizer, Per-Device Batch Size of 4

# Constants
CONFIG="--config conf/gpt2-mistral-medium-config.yaml"
INFRA="--nnodes 2 --nproc_per_node 8"

# Batch Size
D_BSZ_4="--training_arguments.fp16 true --training_arguments.per_device_train_batch_size 4"

# DeepSpeed Training Configuration
DS_Z2="--training_arguments.deepspeed conf/deepspeed/z2-medium-conf.json"

# Random Seeds -- Arwen :: 21, Beren :: 49, Celebrimbor :: 81
ARWEN="--seed 21"
BEREN="--seed 49"
CELEBRIMBOR="--seed 81"

# Set DeepSpeed Launcher Parameters
MASTER_ADDR=sphinx1.stanford.edu
MASTER_PORT=7000
DISTRIBUTED_ARGS="--num_gpus 8 --num_nodes 2 --master_addr $MASTER_ADDR"

# ---

# Multi-Node DS-Z2, Linear LR Schedule, Device BSZ = 4 --> Cleanup --> Sleep
deepspeed $DISTRIBUTED_ARGS train.py $CONFIG $INFRA $D_BSZ_4 $ARWEN $DS_Z2 --run_id arwen-gpt2-medium-x21
pkill -f "train.py"
sleep 3
