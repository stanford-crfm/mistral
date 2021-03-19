# ddp-single.sh
#   Benchmarking Script for Single-Node DDP Trainer, verifying distributed data parallel training with and without
#   gradient checkpointing as well as with different batch sizes. The choice of batch size in this script were derived
#   directly from the results of the Vanilla runs!

# Constants
CONFIG="--config conf/gpt2-benchmark-config.yaml"
INFRA="--nnodes 1 --nproc_per_node 8"
GC="--model.gradient_checkpointing true"
FP16="--training_arguments.fp16 true"

# Only Two Choices for Batch Size -- Max for w/ Gradient Checkpointing (32 on 40 GB A100) and w/o (8 on 40GB A100)
D_BSZ_8="--training_arguments.per_device_train_batch_size 8"
D_BSZ_32="--training_arguments.per_device_train_batch_size 32"

# Setup Distributed Launch Parameters -- We probably don't need Master Address/Port, but including for completeness
MASTER_ADDR=sphinx1.stanford.edu
MASTER_PORT=7000
WORLD_SIZE=8
DISTRIBUTED_ARGS="--nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr $MASTER_ADDR"
LAUNCHER="torch.distributed.launch"

# ---

# Single Node DDP, No GC, FP32, Device BSZ = 8 --> Cleanup (`torch.distributed.launch` doesn't like cleanup) --> Sleep
#python -m $LAUNCHER $DISTRIBUTED_ARGS train.py $CONFIG $INFRA $D_BSZ_8 --run_id 21-ddp-n=1-g=8-fp32-dbsz=8
#pkill -f "train.py"
#sleep 3

# Single Node DDP, ++GC, FP32, Device BSZ = 32 --> Cleanup (`torch.distributed.launch` doesn't like cleanup) --> Sleep
python -m $LAUNCHER $DISTRIBUTED_ARGS train.py $CONFIG $INFRA $GC $D_BSZ_32 --run_id 22-ddp-n=1-g=8-gc-fp32-dbsz=32
pkill -f "train.py"
sleep 3

# Single Node DDP, No GC, FP16, Device BSZ = 8 --> Cleanup (`torch.distributed.launch` doesn't like cleanup) --> Sleep
python -m $LAUNCHER $DISTRIBUTED_ARGS train.py $CONFIG $INFRA $FP16 $D_BSZ_8 --run_id 23-ddp-n=1-g=8-fp16-dbsz=8
pkill -f "train.py"
sleep 3

# Single Node DDP, ++GC, FP32, Device BSZ = 32 --> Cleanup (`torch.distributed.launch` doesn't like cleanup) --> Sleep
python -m $LAUNCHER $DISTRIBUTED_ARGS train.py $CONFIG $INFRA $GC $FP16 $D_BSZ_32 --run_id 24-ddp-n=1-g=8-gc-fp16-dbsz=32
pkill -f "train.py"
sleep 3
