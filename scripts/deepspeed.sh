# Sphinx1 Private IP: 172.24.67.75
# Sphinx2 Private IP: 172.24.67.78

# Command Line Arguments
nnodes=${1:-1}
node_rank=${2:-0}

# Default Configuration of GPUs on the Sphinx Machines
GPUS_PER=8

# Assumes `sphinx1` is the main node - node rank must be 0 on sphinx1!
MASTER_ADDR=sphinx1.stanford.edu
MASTER_PORT=7000
WORLD_SIZE=$((${nnodes}*${node_rank}))

# DeepSpeed Launch Parameters
DISTRIBUTED_ARGS="--num_gpus $GPUS_PER --num_nodes ${nnodes} --master_addr $MASTER_ADDR"

# Default `train.py` config arguments
CONFIG_ARGS="--config conf/gpt2-sphinx-debug-config.yaml --nproc_per_node $GPUS_PER --nnodes ${nnodes}"

# DeepSpeed Configurations
DEEPSPEED_Z1="--training_arguments.deepspeed scripts/deepspeed/z1-conf.json"
DEEPSPEED_Z2="--training_arguments.deepspeed scripts/deepspeed/z2-conf.json"
DEEPSPEED_Z3="--training_arguments.deepspeed scripts/deepspeed/z3-conf.json"

# export NCCL_DEBUG=INFO; \
# =>> ZeRO-1
# deepspeed $DISTRIBUTED_ARGS train.py $CONFIG_ARGS $DEEPSPEED_Z1

# =>> ZeRO-2
# deepspeed $DISTRIBUTED_ARGS train.py $CONFIG_ARGS $DEEPSPEED_Z2

# =>> ZeRO-3
deepspeed $DISTRIBUTED_ARGS train.py $CONFIG_ARGS $DEEPSPEED_Z3

# Kill running processing
pkill -f "train.py"
