# vanilla.sh
#   Benchmarking Script for Vanilla Trainer (very top of the Benchmarking table). This is to get a rough upper bound
#   on single-GPU runtime, mostly as a sanity check.

# Constants
CONFIG="--config conf/gpt2-benchmark-config.yaml"
INFRA="--nnodes 1 --nproc_per_node 1"
GC="--model.gradient_checkpointing true"
FP16="--training_arguments.fp16 true"

# Various Device Batch Sizes
D_BSZ_1="--training_arguments.per_device_train_batch_size 1"
D_BSZ_2="--training_arguments.per_device_train_batch_size 2"
D_BSZ_4="--training_arguments.per_device_train_batch_size 4"
D_BSZ_8="--training_arguments.per_device_train_batch_size 8"
D_BSZ_16="--training_arguments.per_device_train_batch_size 16"
D_BSZ_32="--training_arguments.per_device_train_batch_size 32"

# ---

# Single-Node, Single GPU, No GC, FP32, Device BSZ = 1
CUDA_VISIBLE_DEVICES=0 python train.py $CONFIG $INFRA $D_BSZ_1 --run_id 01-vanilla-g=1-fp32-dbsz=1

# Single-Node, Single GPU, No GC, FP32, Device BSZ = 2
CUDA_VISIBLE_DEVICES=0 python train.py $CONFIG $INFRA $D_BSZ_2 --run_id 02-vanilla-g=1-fp32-dbsz=2

# Single-Node, Single GPU, No GC, FP32, Device BSZ = 4
CUDA_VISIBLE_DEVICES=0 python train.py $CONFIG $INFRA $D_BSZ_4 --run_id 03-vanilla-g=1-fp32-dbsz=4

# Single-Node, Single GPU, No GC, FP32, Device BSZ = 8
CUDA_VISIBLE_DEVICES=0 python train.py $CONFIG $INFRA $D_BSZ_8 --run_id 04-vanilla-g=1-fp32-dbsz=8

# ---

# Single-Node, Single GPU, ++GC, FP32, Device BSZ = 1
CUDA_VISIBLE_DEVICES=0 python train.py $CONFIG $INFRA $GC $D_BSZ_1 --run_id 05-vanilla-g=1-gc-fp32-dbsz=1

# Single-Node, Single GPU, ++GC, FP32, Device BSZ = 2
CUDA_VISIBLE_DEVICES=0 python train.py $CONFIG $INFRA $GC $D_BSZ_2 --run_id 06-vanilla-g=1-gc-fp32-dbsz=2

# Single-Node, Single GPU, ++GC, FP32, Device BSZ = 4
CUDA_VISIBLE_DEVICES=0 python train.py $CONFIG $INFRA $GC $D_BSZ_4 --run_id 07-vanilla-g=1-gc-fp32-dbsz=4

# Single-Node, Single GPU, ++GC, FP32, Device BSZ = 8
CUDA_VISIBLE_DEVICES=0 python train.py $CONFIG $INFRA $GC $D_BSZ_8 --run_id 08-vanilla-g=1-gc-fp32-dbsz=8

# ---

# Single-Node, Single GPU, No GC, FP16, Device BSZ = 1
CUDA_VISIBLE_DEVICES=0 python train.py $CONFIG $INFRA $FP16 $D_BSZ_1 --run_id 09-vanilla-g=1-fp16-dbsz=1

# Single-Node, Single GPU, No GC, FP16, Device BSZ = 2
CUDA_VISIBLE_DEVICES=0 python train.py $CONFIG $INFRA $FP16 $D_BSZ_2 --run_id 10-vanilla-g=1-fp16-dbsz=2

# Single-Node, Single GPU, No GC, FP16, Device BSZ = 4
CUDA_VISIBLE_DEVICES=0 python train.py $CONFIG $INFRA $FP16 $D_BSZ_4 --run_id 11-vanilla-g=1-fp16-dbsz=4

# Single-Node, Single GPU, No GC, FP16, Device BSZ = 8
CUDA_VISIBLE_DEVICES=0 python train.py $CONFIG $INFRA $FP16 $D_BSZ_8 --run_id 12-vanilla-g=1-fp16-dbsz=8

# ---

# Single-Node, Single GPU, ++GC, FP16, Device BSZ = 1
CUDA_VISIBLE_DEVICES=0 python train.py $CONFIG $INFRA $GC $FP16 $D_BSZ_1 --run_id 13-vanilla-g=1-gc-fp16-dbsz=1

# Single-Node, Single GPU, ++GC, FP16, Device BSZ = 2
CUDA_VISIBLE_DEVICES=0 python train.py $CONFIG $INFRA $GC $FP16 $D_BSZ_2 --run_id 14-vanilla-g=1-gc-fp16-dbsz=2

# Single-Node, Single GPU, ++GC, FP16, Device BSZ = 4
CUDA_VISIBLE_DEVICES=0 python train.py $CONFIG $INFRA $GC $FP16 $D_BSZ_4 --run_id 15-vanilla-g=1-gc-fp16-dbsz=4

# Single-Node, Single GPU, ++GC, FP16, Device BSZ = 8
CUDA_VISIBLE_DEVICES=0 python train.py $CONFIG $INFRA $GC $FP16 $D_BSZ_8 --run_id 16-vanilla-g=1-gc-fp16-dbsz=8

# --- (Extra Experiments because Gradient Checkpointing Exceeded Expectations)

# Single-Node, Single GPU, ++GC, FP32, Device BSZ = 16
CUDA_VISIBLE_DEVICES=0 python train.py $CONFIG $INFRA $GC $D_BSZ_16 --run_id 17-vanilla-g=1-gc-dbsz=16

# Single-Node, Single GPU, ++GC, FP16, Device BSZ = 16
CUDA_VISIBLE_DEVICES=0 python train.py $CONFIG $INFRA $GC $FP16 $D_BSZ_16 --run_id 18-vanilla-g=1-gc-dbsz=16

# Single-Node, Single GPU, ++GC, FP32, Device BSZ = 32
CUDA_VISIBLE_DEVICES=0 python train.py $CONFIG $INFRA $GC $D_BSZ_32 --run_id 19-vanilla-g=1-gc-dbsz=32

# Single-Node, Single GPU, ++GC, FP16, Device BSZ = 32
CUDA_VISIBLE_DEVICES=0 python train.py $CONFIG $INFRA $GC $FP16 $D_BSZ_32 --run_id 20-vanilla-g=1-gc-dbsz=32
