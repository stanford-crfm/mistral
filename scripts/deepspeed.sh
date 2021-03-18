# Sphinx1 Private IP: 172.24.67.75
# Sphinx2 Private IP: 172.24.67.78

# Command Line Arguments
nnodes=${1:-1}
node_rank=${2:-0}

# Default Configuration of GPUs on the Sphinx Machines
GPUS_PER_NODE=8

# Assumes `sphinx1` is the main node - node rank must be 0 on sphinx1!
MASTER_ADDR=sphinx1.stanford.edu
MASTER_PORT=7000
WORLD_SIZE=$((${nnodes}*${node_rank}))

DISTRIBUTED_ARGS="--num_gpus $GPUS_PER_NODE --num_nodes ${nnodes} --master_addr $MASTER_ADDR"

# export NCCL_DEBUG=INFO; \
deepspeed "$DISTRIBUTED_ARGS" train.py --config conf/gpt2-sphinx-debug-config.yaml --training_arguments.deepspeed conf/deepspeed/ds_conf.json

# Kill running processing
pkill -f "train.py"
