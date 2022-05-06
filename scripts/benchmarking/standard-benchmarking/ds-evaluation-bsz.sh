# ZeRO-1 -- Multi-Node - Evaluation BSZ = 2
deepspeed --num_gpus 8 --num_nodes 2 --master_addr sphinx1.stanford.edu train.py --fileconf/gpt2-benchmark-config.yaml --nnodes 2 --nproc_per_node 8 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 16 --training_arguments.per_device_eval_batch_size 2 --training_arguments.deepspeed scripts/deepspeed/z1-conf.json --run_id 64-eval-ds=z1-n=2-g=8-fp16-dbsz=16-ebsz=2
pkill -f "train.py"
sleep 3

# ZeRO-1 -- Multi-Node - Evaluation BSZ = 4
deepspeed --num_gpus 8 --num_nodes 2 --master_addr sphinx1.stanford.edu train.py --fileconf/gpt2-benchmark-config.yaml --nnodes 2 --nproc_per_node 8 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 16 --training_arguments.per_device_eval_batch_size 4 --training_arguments.deepspeed scripts/deepspeed/z1-conf.json --run_id 65-eval-ds=z1-n=2-g=8-fp16-dbsz=16-ebsz=4
pkill -f "train.py"
sleep 3

# ZeRO-1 -- Multi-Node - Evaluation BSZ = 8
deepspeed --num_gpus 8 --num_nodes 2 --master_addr sphinx1.stanford.edu train.py --fileconf/gpt2-benchmark-config.yaml --nnodes 2 --nproc_per_node 8 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 16 --training_arguments.per_device_eval_batch_size 8 --training_arguments.deepspeed scripts/deepspeed/z1-conf.json --run_id 66-eval-ds=z1-n=2-g=8-fp16-dbsz=16-ebsz=8
pkill -f "train.py"
sleep 3

# ZeRO-1 -- Multi-Node - Evaluation BSZ = 16
deepspeed --num_gpus 8 --num_nodes 2 --master_addr sphinx1.stanford.edu train.py --fileconf/gpt2-benchmark-config.yaml --nnodes 2 --nproc_per_node 8 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 16 --training_arguments.per_device_eval_batch_size 16 --training_arguments.deepspeed scripts/deepspeed/z1-conf.json --run_id 67-eval-ds=z1-n=2-g=8-fp16-dbsz=16-ebsz=16
pkill -f "train.py"
sleep 3

# ZeRO-1 -- Multi-Node - Evaluation BSZ = 32
deepspeed --num_gpus 8 --num_nodes 2 --master_addr sphinx1.stanford.edu train.py --fileconf/gpt2-benchmark-config.yaml --nnodes 2 --nproc_per_node 8 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 16 --training_arguments.per_device_eval_batch_size 32 --training_arguments.deepspeed scripts/deepspeed/z1-conf.json --run_id 68-eval-ds=z1-n=2-g=8-fp16-dbsz=16-ebsz=32
pkill -f "train.py"
sleep 3

# ZeRO-1 -- Multi-Node - Evaluation BSZ = 64
deepspeed --num_gpus 8 --num_nodes 2 --master_addr sphinx1.stanford.edu train.py --fileconf/gpt2-benchmark-config.yaml --nnodes 2 --nproc_per_node 8 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 16 --training_arguments.per_device_eval_batch_size 64 --training_arguments.deepspeed scripts/deepspeed/z1-conf.json --run_id 69-eval-ds=z1-n=2-g=8-fp16-dbsz=16-ebsz=64
pkill -f "train.py"
sleep 3

# ZeRO-1 -- Multi-Node - Evaluation BSZ = 128
deepspeed --num_gpus 8 --num_nodes 2 --master_addr sphinx1.stanford.edu train.py --fileconf/gpt2-benchmark-config.yaml --nnodes 2 --nproc_per_node 8 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 16 --training_arguments.per_device_eval_batch_size 128 --training_arguments.deepspeed scripts/deepspeed/z1-conf.json --run_id 70-eval-ds=z1-n=2-g=8-fp16-dbsz=16-ebsz=128
pkill -f "train.py"
sleep 3
