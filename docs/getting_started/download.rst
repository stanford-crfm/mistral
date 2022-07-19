Download Models
===============

Mistral Checkpoints
-------------------

The Mistral team has trained 5 GPT-2 Medium models and 5 GPT-2 Small models on the OpenWebText corpus and is making them available to the public.

Each model is available on the `Hugging Face Hub <https://huggingface.co/stanford-crfm/>`_ and can be accessed via Git LFS.

Checkpoints are branches of each repo for each model. For instance, here is how to get the 300k step checkpoint for battlestar: ::

    # Make sure you have git-lfs installed
    # (https://git-lfs.github.com)
    git lfs install

    # get checkpoint 300000 for battlestar
    git clone https://huggingface.co/stanford-crfm/battlestar-gpt2-small-x49 --branch checkpoint-300000 --single-branch
    cd battlestar-gpt2-small-x49
    git lfs pull


Links to the checkpoints are in the table below.
