Training With Multiple GPU's
=======================================

Once you've got training working with a single node/single gpu, you can easily move on to training
with multiple GPU's if your machine has them.

This can be done two ways. The first, which we show here, uses `torch.distributed.launch <https://pytorch.org/docs/stable/distributed.html#launch-utility>`_ , a utility for launching multiple processes per node for distributed training. The second uses DeepSpeed, which we go over in our :doc:`multi node training <deepspeed>`.

To use torch, run this command with ``--nproc_per_node`` set to the number of GPU's you want to use (in this example we'll go with 2) ::

    conda activate mistral
    cd mistral
    python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 train.py --config conf/hello-world.yaml --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 8 --run_id hello-world-single-node-multi-gpu

You should see similar output as when running :doc:`single node/single gpu training <../getting_started/train>`, except it should
run twice as fast!

As noted with single node/single gpu training, you may need to adjust the batch size to avoid OOM memories.
