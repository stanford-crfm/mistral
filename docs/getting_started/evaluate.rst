Training
========

Evaluating A Model
------------------

Once you've finished training your model, you can run evaluation on any checkpoint to see PPL scores
on OpenWebText, WikiText-103, and Lambada.

To run evaluation, use this command: ::

    cd mistral
    conda activate mistral
    CUDA_VISIBLE_DEVICES=0 python train.py --file conf/tutorial-gpt2-micro.yaml --nnodes 1 --nproc_per_node 1 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 2 --model.initial_weights /path/to/runs/my-run/checkpoint-400000 --run_training False

This will skip the training process and run a final evaluation, initializing from the weights of the checkpoint.

To evaluate a particular model, you need to supply the same config that was used to train the model (e.g. ``conf/tutorial-gpt2-micro.yaml``) in this example.

Example Output
--------------

If all is successful, you should see output similar to this: ::

    |=>> 08/13 [14:00:22] - mistral - INFO :: Running final evaluation...
    ...
    {'eval_openwebtext_loss': 2.99070405960083, 'eval_openwebtext_ppl': 19.899688127064493, 'eval_openwebtext_runtime': 14.8929, 'eval_openwebtext_samples_per_second': 15.376, 'epoch': None, 'eval_wikitext_loss': 2.90213680267334, 'eval_wikitext_runtime': 26.5247, 'eval_wikitext_samples_per_second': 17.192, 'eval_wikitext_ppl': 18.21302145232096, 'eval_lambada_loss': 2.5298995971679688, 'eval_lambada_runtime': 283.1437, 'eval_lambada_samples_per_second': 17.196, 'eval_lambada_ppl': 12.552245792372315, 'eval_mem_cpu_alloc_delta': 532480, 'eval_mem_gpu_alloc_delta': 0, 'eval_mem_cpu_peaked_delta': 98304, 'eval_mem_gpu_peaked_delta': 1242778112}
