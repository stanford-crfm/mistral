Generate Text With A Trained Model
==================================

Once you've completed :doc:`training <../getting_started/train>`, you can use your model to generate text.

In this tutorial we'll walk through getting 🤗 Transformers  et up and generating text with a trained GPT-2 Small model.

Set Up Hugging Face
-------------------

Hugging Face's ``transformers`` repo provides a helpful script for generating text with a GPT-2 model.

To access these scripts, clone the repo ::

    git clone https://github.com/huggingface/transformers.git

Run run_generation.py With Your Model
-------------------------------------

As your model training runs, it should save checkpoints with all of the model resources in the directory
you specified with ``articfacts.run_dir`` in the ``conf/tutorial-gpt2-micro.yaml`` config file.

For this example, lets assume you have saved the checkpoints in ``/home/tutorial-gpt2-micro/runs/run-1``. If you trained
for 400000 steps, you should have a corresponding checkpoint at ``/home/tutorial-gpt2-micro/runs/run-1/checkpoint-400000``.
This directory contains all the resources for your model, with files such as ``pytorch_model.bin`` containing
the actual model and ``vocab.json`` which maps word pieces to their indices among others.

To run text generation, issue the following command: ::

    conda activate mistral
    cd transformers/examples/text-generation
    python run_generation.py --model_type=gpt2 --model_name_or_path=/home/tutorial-gpt2-micro/runs/run-1/checkpoint-400000

This will create the following output requesting a text prompt. ::

    06/28/2021 03:16:16 - WARNING - __main__ - device: cuda, n_gpu: 1, 16-bits training: False
    06/28/2021 03:16:26 - INFO - __main__ - Namespace(device=device(type='cuda'), fp16=False, k=0, length=20, model_name_or_path='hello-world/runs/run-1/checkpoint-400000', model_type='gpt2', n_gpu=1, no_cuda=False, num_return_sequences=1, p=0.9, padding_text='', prefix='', prompt='', repetition_penalty=1.0, seed=42, stop_token=None, temperature=1.0, xlm_language='')
    Model prompt >>>

Enter an example prompt, and the script will generate a text completion for you using your model! ::

    Model prompt >>> Hello world. This is a prompt.
    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
    === GENERATED SEQUENCE 1 ===
    Hello world. This is a prompt. This is no ‘say what, say it’ stuff, it’s all on
