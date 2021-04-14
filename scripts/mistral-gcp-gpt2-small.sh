# mistral-gcp-gpt2-small.sh
#   Mistral GPT-2 Small Full Run with the DeepSpeed ZeRO-2 Optimizer, Per-Device Batch Size of 16 on Google Cloud with
#   MegaGPU Instances.

# Constants
CONFIG="--config conf/gpt2-mistral-config.yaml"
GCP_CONFIG="--config conf/gpt2-mistral-gcp-config.yaml"
INFRA="--nnodes 1 --nproc_per_node 16"

# Batch Size
D_BSZ_16="--training_arguments.fp16 true --training_arguments.per_device_train_batch_size 16"

# DeepSpeed Training Configuration
DS_Z2="--training_arguments.deepspeed conf/deepspeed/z2-conf.json"

# Random Seeds -- Cyclone :: 81
CYCLONE="--seed 81"

# Set DeepSpeed Launcher Parameters
DISTRIBUTED_ARGS="--num_gpus 16 --num_nodes 1"

# ---

# Single-Node DS-Z2, Linear LR Schedule, Device BSZ = 16 --> Cleanup --> Seed =>> Seed 81
deepspeed $DISTRIBUTED_ARGS train.py $GCP_CONFIG $INFRA $D_BSZ_16 $CYCLONE $DS_Z2 --run_id cyclone-gpt2-small-x81
pkill -f "train.py"
sleep 3
