# mistral-gpt2-small.sh
#   Mistral GPT-2 Small Full Run with the DeepSpeed ZeRO-2 Optimizer, Per-Device Batch Size of 16.

# Constants
CONFIG="--config conf/gpt2-mistral-config.yaml"
INFRA="--nnodes 2 --nproc_per_node 8"

# Batch Size
D_BSZ_16="--training_arguments.fp16 true --training_arguments.per_device_train_batch_size 16"

# DeepSpeed Training Configuration
DS_Z2="--training_arguments.deepspeed conf/deepspeed/z2-conf.json"

# Resume Behavior (for Debugging)
RESUME="--resume true --resume_checkpoint /scr-ssd/mercury/mistral/runs/crash-02-fp16-NaN-aurora-gpt2-small-x21/checkpoint-81000"

# Set DeepSpeed Launcher Parameters
MASTER_ADDR=sphinx1.stanford.edu
MASTER_PORT=7000
DISTRIBUTED_ARGS="--num_gpus 8 --num_nodes 2 --master_addr $MASTER_ADDR"

# ---

# Multi-Node DS-Z2, Linear LR Schedule, Device BSZ = 16 --> Cleanup --> Sleep
deepspeed $DISTRIBUTED_ARGS train.py $CONFIG $INFRA $D_BSZ_16 $DS_Z2 $RESUME --run_id aurora-gpt2-small-x21
pkill -f "train.py"
sleep 3
