# ddp-multi.sh
#   Benchmarking Script for Intense Multi-Node DDP, running FP 16 with and without gradient checkpointing.
#
# Note: Sidd handwrote these scripts, but would be nice to spend some time figuring out how to automate generating
# these in the future...
# ---

## =>> Sphinx1
python -m torch.distributed.launch --nproc_per_node 8 --nnodes 2 --node_rank 0 --master_addr=sphinx1.stanford.edu train.py --fileconf/gpt2-intensive-config.yaml --nnodes 2 --nproc_per_node 8 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 8 --run_id alfa-ddp-n=2-g=8-fp16-dbsz=8; pkill -f "train.py"; sleep 3

## =>> Sphinx2
python -m torch.distributed.launch --nproc_per_node 8 --nnodes 2 --node_rank 1 --master_addr=sphinx1.stanford.edu train.py --fileconf/gpt2-intensive-config.yaml --nnodes 2 --nproc_per_node 8 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 8 --run_id alfa-ddp-n=2-g=8-fp16-dbsz=8; pkill -f "train.py"; sleep 3

# ---

# Multi-Node DDP, ++GC, FP16, Device BSZ = 32

## =>> Sphinx1
python -m torch.distributed.launch --nproc_per_node 8 --nnodes 2 --node_rank 0 --master_addr=sphinx1.stanford.edu train.py --fileconf/gpt2-intensive-config.yaml --nnodes 2 --nproc_per_node 8 --model.gradient_checkpointing true --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 32 --run_id bravo-ddp-n=2-g=8-gc-fp16-dbsz=32; pkill -f "train.py"; sleep 3

## =>> Sphinx2
python -m torch.distributed.launch --nproc_per_node 8 --nnodes 2 --node_rank 1 --master_addr=sphinx1.stanford.edu train.py --fileconf/gpt2-intensive-config.yaml --nnodes 2 --nproc_per_node 8 --model.gradient_checkpointing true --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 32 --run_id bravo-ddp-n=2-g=8-gc-fp16-dbsz=32; pkill -f "train.py"; sleep 3
