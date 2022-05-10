# fairscale-multi.sh
#   Benchmarking Script for Multi-Node FairScale Trainer, verifying multi-stage sharded training (ZeRO 1, 2, and 3)
#   with and without gradient checkpointing. Batch Sizes here are taken from the Single-Node FS Runs (since nothing
#   changes across node boundaries w.r.t. ZeRO.
#
# Note: Sidd handwrote these scripts, but would be nice to spend some time figuring out how to automate generating
# these in the future...
# ---

# Multi-Node FS-Z2, No GC, FP16, Device BSZ = 8

## =>> Sphinx1
python -m torch.distributed.launch --nproc_per_node 8 --nnodes 2 --node_rank 0 --master_addr=sphinx1.stanford.edu train.py --file conf/gpt2-intensive-config.yaml --nnodes 2 --nproc_per_node 8 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 8 --training_arguments.sharded_ddp zero_dp_2+auto_wrap --run_id charlie-fs=z2-n=2-g=8-fp16-dbsz=8; pkill -f "train.py"; sleep 3

## =>> Sphinx2
python -m torch.distributed.launch --nproc_per_node 8 --nnodes 2 --node_rank 1 --master_addr=sphinx1.stanford.edu train.py --file conf/gpt2-intensive-config.yaml --nnodes 2 --nproc_per_node 8 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 8 --training_arguments.sharded_ddp zero_dp_2+auto_wrap --run_id charlie-fs=z2-n=2-g=8-fp16-dbsz=8; pkill -f "train.py"; sleep 3

# ---

# Multi-Node FS-Z3, No GC, FP16, Device BSZ = 8

## =>> Sphinx1
python -m torch.distributed.launch --nproc_per_node 8 --nnodes 2 --node_rank 0 --master_addr=sphinx1.stanford.edu train.py --file conf/gpt2-intensive-config.yaml --nnodes 2 --nproc_per_node 8 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 8 --training_arguments.sharded_ddp zero_dp_3+auto_wrap --run_id delta-fs=z3-n=2-g=8-fp16-dbsz=8; pkill -f "train.py"; sleep 3

## =>> Sphinx2
python -m torch.distributed.launch --nproc_per_node 8 --nnodes 2 --node_rank 1 --master_addr=sphinx1.stanford.edu train.py --file conf/gpt2-intensive-config.yaml --nnodes 2 --nproc_per_node 8 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 8 --training_arguments.sharded_ddp zero_dp_3+auto_wrap --run_id delta-fs=z3-n=2-g=8-fp16-dbsz=8; pkill -f "train.py"; sleep 3
