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

# `torch.distributed.launch` Parameters
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER --nnodes ${nnodes} --node_rank ${node_rank} --master_addr $MASTER_ADDR"

# Default `train.py` config arguments
CONFIG_ARGS="--config conf/gpt2-sphinx-debug-config.yaml --nproc_per_node $GPUS_PER --nnodes ${nnodes}"

# FairScale Parameters
FAIRSCALE_Z1="--training_arguments.sharded_ddp simple"
FAIRSCALE_Z2="--training_arguments.sharded_ddp zero_dp_2+auto_wrap"
FAIRSCALE_Z3="--training_arguments.sharded_ddp zero_dp_3+auto_wrap"
FAIRSCALE_Z2_OFF="--training_arguments.sharded_ddp zero_dp_2+auto_wrap+offload"
FAIRSCALE_Z3_OFF="--training_arguments.sharded_ddp zero_dp_3+auto_wrap+offload"

# export NCCL_DEBUG=INFO; \
# =>> ZeRO-1 (Simple)
python -m torch.distributed.launch $DISTRIBUTED_ARGS train.py $CONFIG_ARGS $FAIRSCALE_Z1

# =>> ZeRO-2
# python -m torch.distributed.launch $DISTRIBUTED_ARGS train.py $CONFIG_ARGS $FAIRSCALE_Z2

# =>> ZeRO-3
# python -m torch.distributed.launch $DISTRIBUTED_ARGS train.py $CONFIG_ARGS $FAIRSCALE_Z3

# TODO D :: Offloading Doesn't Work Yet?
# =>> ZeRO-2 Offload
# python -m torch.distributed.launch $DISTRIBUTED_ARGS train.py $CONFIG_ARGS $FAIRSCALE_Z2_OFF

# =>> ZeRO-3 Offload
# python -m torch.distributed.launch $DISTRIBUTED_ARGS train.py $CONFIG_ARGS $FAIRSCALE_Z3_OFF

# Kill Running Processes (Because `torch.distributed.launch` doesn't like to clean up after itself...)
pkill -f "train.py"
