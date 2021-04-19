# resume-single-node.sh
#   Single Node GPT-2 Small `Resume from Checkpoint` -- O(1) Fast-Forward -- Debugging. Uses the DeepSpeed ZeRO-2
#   Optimizer, Per-Device Batch Size of 16.

# Constants
CONFIG="--config conf/gpt2-debugging-config.yaml"
INFRA="--nnodes 2 --nproc_per_node 8"
RESUME="--resume true"

# Batch Size
D_BSZ_16="--training_arguments.fp16 true --training_arguments.per_device_train_batch_size 16"

# DeepSpeed Training Configuration
DS_Z2="--training_arguments.deepspeed conf/deepspeed/z2-conf.json"

# Set DeepSpeed Launcher Parameters
MASTER_ADDR=sphinx1.stanford.edu
MASTER_PORT=7000
DISTRIBUTED_ARGS="--num_gpus 8 --num_nodes 2 --master_addr $MASTER_ADDR"

# Run Identifier (for Resuming)
BASELINE_RUN_ID="--run_id baseline-ff-gpt2-small-x21"
FF_RUN_ID="--run_id resume-ff-gpt2-small-x21"

# ---

# Single-Node DS-Z2, Linear LR Schedule, Device BSZ = 16 --> Cleanup --> Sleep
deepspeed $DISTRIBUTED_ARGS train.py $CONFIG $INFRA $RESUME $D_BSZ_16 $DS_Z2 $FF_RUN_ID
pkill -f "train.py"
sleep 3
