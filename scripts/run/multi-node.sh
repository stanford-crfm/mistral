# ZeRO-1 -- Multi-Node!
deepspeed --num_gpus 8 --num_nodes 2 --master_addr sphinx1.stanford.edu train.py --file conf/gpt2-benchmark-config.yaml --nnodes 2 --nproc_per_node 8 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 16 --training_arguments.deepspeed scripts/deepspeed/z1-conf.json --run_id 63-sk-on-eval-ds=z1-n=2-g=8-fp16-dbsz=16
pkill -f "train.py"
sleep 3
