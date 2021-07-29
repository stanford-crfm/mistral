# Mistral

> *Mistral*: A strong and cool northwesterly wind that builds as it moves, bringing good health and clear skies.

A framework for fast and efficient large-scale language model training, built with Hugging Face :hugs:. Includes tools
and helpful scripts for incorporating new pre-training datasets, various schemes for single node and distributed
training, and importantly, scripts for evaluation and measuring bias.

Full documentation can be found on our [Read The Docs](https://nlp.stanford.edu/local/mistral/docs/_build/html/index.html) site.

A Project Mercury Endeavor.

---

## Quickstart

### Installation

```bash
git clone https://github.com/stanford-mercury/mistral.git
cd mistral
conda env create -f environments/environment-gpu.yaml  # Choose CUDA Kernel based on Hardware!
conda activate mistral
```

Note: The provided environment assumes CUDA 11.0, you may need to adjust this environment accordingly based on your set up.

### Run Training

First make sure to update `conf/hello-world.yaml` with directories for storing the Hugging Face cache and model runs.

```
# Artifacts & Caching
artifacts:
    cache_dir: /path/to/artifacts
    run_dir: /path/to/runs
```

**Run training (single node/single gpu)**

```bash
cd mistral
conda activate mistral
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/hello-world.yaml --nnodes 1 --nproc_per_node 1 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 8
```

**Run training (multi-node/multi-gpu with DeepSpeed)**

Assuming you want to run on `machine1` and `machine2`, add the following content to `/job/hostfile`

```
machine1 slots=8
machine2 slots=8
```

Note: This assumes each machine has 8 GPU's. Adjust accordingly.

```bash
cd mistral
conda activate mistral
deepspeed --num_gpus 8 --num_nodes 2 --master_addr machine1 train.py --config conf/hello-world.yaml --nnodes 2 --nproc_per_node 8 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 4 --training_arguments.deepspeed conf/deepspeed/z1-conf.json --run_id hello-world-multi-node > hello-world-multi-node.out 2> hello-world-multi-node.err
```

Note: You may need to adjust your batch size depending on the capacity of your GPU.

---

## Resources

The Mistral team has trained 5 GPT-2 Medium models and 5 GPT-2 Small models on the OpenWebText corpus.

Checkpoints can be loaded as Hugging Face models. For each model, checkpoints at 100k, 200k, and 400k steps
are provided.

| Run | Type | Checkpoint | Size | Link |
| --- | --- | --- | --- | --- | --- |
| Arwen | GPT-2 Medium | 400000 | 4.9G | [download](https://storage.googleapis.com/mistral-models/gpt2-medium/arwen-gpt2-medium-x21/arwen-checkpoint-400000.zip) |

| Run | Link | Size |
| --- | --- | --- |
| Arwen | [download](https://storage.googleapis.com/mistral-models/gpt2-medium/arwen-gpt2-medium-x21/arwen-checkpoint-400000.zip) | 4.9G |
| Arwen | [download](https://storage.googleapis.com/mistral-models/gpt2-medium/arwen-gpt2-medium-x21/arwen-checkpoint-400000.zip) | 4.9G |
| Arwen | [download](https://storage.googleapis.com/mistral-models/gpt2-medium/arwen-gpt2-medium-x21/arwen-checkpoint-400000.zip) | 4.9G |
| Arwen | [download](https://storage.googleapis.com/mistral-models/gpt2-medium/arwen-gpt2-medium-x21/arwen-checkpoint-400000.zip) | 4.9G |


---

## Contributing

Please see our [Read The Docs](https://nlp.stanford.edu/local/mistral/docs/_build/html/contributing.html) page for info on contributing.
