#Sphinx1 Private IP: 172.24.67.75
#Sphinx2 Private IP: 172.24.67.78

nnodes=${1:-1}
node_rank=${2:-0}

GPUS_PER_NODE=8
# Assumes Sphinx1 is the master node - node rank must be 0 pn sphinx1
MASTER_ADDR=sphinx1.stanford.edu
MASTER_PORT=6000
WORLD_SIZE=$((${nnodes}*${node_rank}))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes ${nnodes} --node_rank ${node_rank} --master_addr $MASTER_ADDR"

# export NCCL_DEBUG=INFO; \
python3 -m torch.distributed.launch $DISTRIBUTED_ARGS /juice/scr/lorr1/mistral/train.py --config /juice/scr/lorr1/mistral/conf/gpt2-sphinx-debug-config.yaml

# Kill running processing
pkill -f "mistral/train.py"
