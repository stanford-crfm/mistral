# ddp-multi.sh
#   Benchmarking Script for Multi-Node DDP Trainer, verifying distributed data parallel training with and without
#   gradient checkpointing as well as with different batch sizes. As with `ddp-single` choice of batch size is
#   directly informed by max/best performing Vanilla runs.
#
# Note: Sidd handwrote these scripts, but would be nice to spend some time figuring out how to automate generating
# these in the future...
# ---

# Multi-Node DDP, No GC, FP32, Device BSZ = 8

## =>> Sphinx1
python -m torch.distributed.launch --nproc_per_node 8 --nnodes 2 --node_rank 0 --master_addr=sphinx1.stanford.edu train.py --file conf/gpt2-benchmark-config.yaml --nnodes 2 --nproc_per_node 8 --training_arguments.per_device_train_batch_size 8 --run_id 25-ddp-n=2-g=8-fp32-dbsz=8; pkill -f "train.py"; sleep 3

## =>> Sphinx 2
python -m torch.distributed.launch --nproc_per_node 8 --nnodes 2 --node_rank 1 --master_addr=sphinx1.stanford.edu train.py --file conf/gpt2-benchmark-config.yaml --nnodes 2 --nproc_per_node 8 --training_arguments.per_device_train_batch_size 8 --run_id 25-ddp-n=2-g=8-fp32-dbsz=8; pkill -f "train.py"; sleep 3

# ---

# Multi-Node DDP, ++GC, FP32, Device BSZ = 32

## =>> Sphinx1
python -m torch.distributed.launch --nproc_per_node 8 --nnodes 2 --node_rank 0 --master_addr=sphinx1.stanford.edu train.py --file conf/gpt2-benchmark-config.yaml --nnodes 2 --nproc_per_node 8 --model.gradient_checkpointing true --training_arguments.per_device_train_batch_size 32 --run_id 26-ddp-n=2-g=8-gc-fp32-dbsz=32; pkill -f "train.py"; sleep 3

## =>> Sphinx2
python -m torch.distributed.launch --nproc_per_node 8 --nnodes 2 --node_rank 1 --master_addr=sphinx1.stanford.edu train.py --file conf/gpt2-benchmark-config.yaml --nnodes 2 --nproc_per_node 8 --model.gradient_checkpointing true --training_arguments.per_device_train_batch_size 32 --run_id 26-ddp-n=2-g=8-gc-fp32-dbsz=32; pkill -f "train.py"; sleep 3

# ---

# Multi-Node DDP, No GC, FP16, Device BSZ = 8

## =>> Sphinx1
python -m torch.distributed.launch --nproc_per_node 8 --nnodes 2 --node_rank 0 --master_addr=sphinx1.stanford.edu train.py --file conf/gpt2-benchmark-config.yaml --nnodes 2 --nproc_per_node 8 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 8 --run_id 27-ddp-n=2-g=8-fp16-dbsz=8; pkill -f "train.py"; sleep 3

## =>> Sphinx2
python -m torch.distributed.launch --nproc_per_node 8 --nnodes 2 --node_rank 1 --master_addr=sphinx1.stanford.edu train.py --file conf/gpt2-benchmark-config.yaml --nnodes 2 --nproc_per_node 8 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 8 --run_id 27-ddp-n=2-g=8-fp16-dbsz=8; pkill -f "train.py"; sleep 3

# ---

# Multi-Node DDP, ++GC, FP16, Device BSZ = 32

## =>> Sphinx1
python -m torch.distributed.launch --nproc_per_node 8 --nnodes 2 --node_rank 0 --master_addr=sphinx1.stanford.edu train.py --file conf/gpt2-benchmark-config.yaml --nnodes 2 --nproc_per_node 8 --model.gradient_checkpointing true --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 32 --run_id 28-ddp-n=2-g=8-gc-fp16-dbsz=32; pkill -f "train.py"; sleep 3

## =>> Sphinx2
python -m torch.distributed.launch --nproc_per_node 8 --nnodes 2 --node_rank 1 --master_addr=sphinx1.stanford.edu train.py --file conf/gpt2-benchmark-config.yaml --nnodes 2 --nproc_per_node 8 --model.gradient_checkpointing true --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 32 --run_id 28-ddp-n=2-g=8-gc-fp16-dbsz=32; pkill -f "train.py"; sleep 3
