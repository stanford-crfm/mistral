Resuming From Checkpoint
=======================================

To resume from a checkpoint, simply add the ``resume`` and ``resume_checkpoint`` options to any of your training commands. ::

    conda activate mistral
    cd mistral
    python train.py --config conf/mistral-micro.yaml --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 2 --run_id resume-demo --resume true --resume_checkpoint /path/to/checkpoint

When resuming from checkpoint the process should pick up from where it left off, using the same learning rate, same point in the learning rate schedule, same point in the data, etc ...
