# Mistral

> *Mistral*: A strong and cool northwesterly wind that builds as it moves, bringing good health and clear skies.

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-green?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)


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

Note: Use `environments/environment-cpu.yaml` if you want to run on the CPU instead.

### Run Training

First make sure to update `conf/tutorial-gpt2-micro.yaml` with directories for storing the Hugging Face cache and model runs.

```
# Artifacts & Caching
artifacts:
    cache_dir: /path/to/artifacts
    run_dir: /path/to/runs
```

Next, make sure that `/path/to/mistral` is on your `PYTHONPATH`.

**Run training (single node/single gpu)**

```bash
cd mistral
conda activate mistral
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/tutorial-gpt2-micro.yaml --nnodes 1 --nproc_per_node 1 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 2 --run_id tutorial-gpt2-micro
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
deepspeed --num_gpus 8 --num_nodes 2 --master_addr machine1 train.py --config conf/tutorial-gpt2-micro.yaml --nnodes 2 --nproc_per_node 8 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 4 --training_arguments.deepspeed conf/deepspeed/z1-conf.json --run_id tutorial-gpt2-micro-multi-node > tutorial-gpt2-micro-multi-node.out 2> tutorial-gpt2-micro-multi-node.err
```

Note: You may need to adjust your batch size depending on the capacity of your GPU.

### Using The Model

Model checkpoints will be stored in the directory specified by the `artifacts.run_dir`. An example checkpoint might be in `/path/to/runs/tutorial-gpt2-micro/checkpoint-1000`.

Mistral stores model checkpoints in the Hugging Face format, so models can be loaded and used in the same manner as if one had trained the model with Hugging Face.

For instance, to generate text with 🤗 Transformers (you will need to clone the [transformers](https://github.com/huggingface/transformers) repo):

```bash
conda activate mistral
cd transformers/examples/text-generation
python run_generation.py --model_type=gpt2 --model_name_or_path=/path/to/runs/tutorial-gpt2-micro/checkpoint-1000
```

Or to load the model in Python code (make sure `/path/to/mistral` is in your `PYTHONPATH`):

```python
from src.models.mistral_gpt2 import MistralGPT2LMHeadModel

model = MistralGPT2LMHeadModel.from_pretrained("/path/to/runs/tutorial-gpt2-micro/checkpoint-1000")
```

---

## Resources

The Mistral team has trained 5 GPT-2 Medium models and 5 GPT-2 Small models on the OpenWebText corpus.

Checkpoints can be loaded as Hugging Face models. For each model, checkpoints at 100k, 200k, and 400k steps are provided.

Internally we have stored over 600 checkpoints for each model. If you are interested in acquiring additional checkpoints, please contact Laurel (lorr1) or Sidd (skaramcheti) at their @stanford.edu addresses.

GPT-2 Medium

| Run | Type | Checkpoint | Size | Link |
| --- | --- | --- | --- | --- |
| Arwen | GPT-2 Medium | 400000 | 4.9G | [download](https://storage.googleapis.com/mistral-models/gpt2-medium/arwen-gpt2-medium-x21/arwen-x21-checkpoint-400000.zip) |
| Arwen | GPT-2 Medium | 200000 | 4.9G | [download](https://storage.googleapis.com/mistral-models/gpt2-medium/arwen-gpt2-medium-x21/arwen-x21-checkpoint-200000.zip) |
| Arwen | GPT-2 Medium | 100000 | 4.9G | [download](https://storage.googleapis.com/mistral-models/gpt2-medium/arwen-gpt2-medium-x21/arwen-x21-checkpoint-100000.zip) |
| Beren | GPT-2 Medium | 400000 | 4.9G | [download](https://storage.googleapis.com/mistral-models/gpt2-medium/beren-gpt2-medium-x49/beren-x49-checkpoint-400000.zip) |
| Beren | GPT-2 Medium | 200000 | 4.9G | [download](https://storage.googleapis.com/mistral-models/gpt2-medium/beren-gpt2-medium-x49/beren-x49-checkpoint-200000.zip) |
| Beren | GPT-2 Medium | 100000 | 4.9G | [download](https://storage.googleapis.com/mistral-models/gpt2-medium/beren-gpt2-medium-x49/beren-x49-checkpoint-100000.zip) |
| Cerebrimbor | GPT-2 Medium | 400000 | 4.9G | [download](https://storage.googleapis.com/mistral-models/gpt2-medium/cerebrimbor-gpt2-medium-x81/cerebrimbor-x81-checkpoint-400000.zip) |
| Cerebrimbor | GPT-2 Medium | 200000 | 4.9G | [download](https://storage.googleapis.com/mistral-models/gpt2-medium/cerebrimbor-gpt2-medium-x81/cerebrimbor-x81-checkpoint-200000.zip) |
| Cerebrimbor | GPT-2 Medium | 100000 | 4.9G | [download](https://storage.googleapis.com/mistral-models/gpt2-medium/cerebrimbor-gpt2-medium-x81/cerebrimbor-x81-checkpoint-100000.zip) |
| Durin | GPT-2 Medium | 400000 | 4.9G | [download](https://storage.googleapis.com/mistral-models/gpt2-medium/durin-gpt2-medium-x343/durin-x343-checkpoint-400000.zip) |
| Durin | GPT-2 Medium | 200000 | 4.9G | [download](https://storage.googleapis.com/mistral-models/gpt2-medium/durin-gpt2-medium-x343/durin-x343-checkpoint-200000.zip) |
| Durin | GPT-2 Medium | 100000 | 4.9G | [download](https://storage.googleapis.com/mistral-models/gpt2-medium/durin-gpt2-medium-x343/durin-x343-checkpoint-100000.zip) |
| Eowyn | GPT-2 Medium | 400000 | 4.9G | [download](https://storage.googleapis.com/mistral-models/gpt2-medium/eowyn-gpt2-medium-x777/eowyn-x777-checkpoint-400000.zip) |
| Eowyn | GPT-2 Medium | 200000 | 4.9G | [download](https://storage.googleapis.com/mistral-models/gpt2-medium/eowyn-gpt2-medium-x777/eowyn-x777-checkpoint-200000.zip) |
| Eowyn | GPT-2 Medium | 100000 | 4.9G | [download](https://storage.googleapis.com/mistral-models/gpt2-medium/eowyn-gpt2-medium-x777/eowyn-x777-checkpoint-100000.zip) |

GPT-2 Small

| Run | Type | Checkpoint | Size | Link |
| --- | --- | --- | --- | --- |
| Alias | GPT-2 Small | 400000 | 1.8G | [download](https://storage.googleapis.com/mistral-models/gpt2-small/alias-gpt2-small-x21/alias-x21-checkpoint-400000.zip) |
| Alias | GPT-2 Small | 200000 | 1.8G | [download](https://storage.googleapis.com/mistral-models/gpt2-small/alias-gpt2-small-x21/alias-x21-checkpoint-200000.zip) |
| Alias | GPT-2 Small | 100000 | 1.8G | [download](https://storage.googleapis.com/mistral-models/gpt2-small/alias-gpt2-small-x21/alias-x21-checkpoint-100000.zip) |
| Battlestar | GPT-2 Small | 400000 | 1.8G | [download](https://storage.googleapis.com/mistral-models/gpt2-small/battlestar-gpt2-small-x49/battlestar-x49-checkpoint-400000.zip) |
| Battlestar | GPT-2 Small | 200000 | 1.8G | [download](https://storage.googleapis.com/mistral-models/gpt2-small/battlestar-gpt2-small-x49/battlestar-x49-checkpoint-200000.zip) |
| Battlestar | GPT-2 Small | 100000 | 1.8G | [download](https://storage.googleapis.com/mistral-models/gpt2-small/battlestar-gpt2-small-x49/battlestar-x49-checkpoint-100000.zip) |
| Caprica | GPT-2 Small | 400000 | 1.8G | [download](https://storage.googleapis.com/mistral-models/gpt2-small/caprica-gpt2-small-x81/caprica-x81-checkpoint-400000.zip) |
| Caprica | GPT-2 Small | 200000 | 1.8G | [download](https://storage.googleapis.com/mistral-models/gpt2-small/caprica-gpt2-small-x81/caprica-x81-checkpoint-200000.zip) |
| Caprica | GPT-2 Small | 100000 | 1.8G | [download](https://storage.googleapis.com/mistral-models/gpt2-small/caprica-gpt2-small-x81/caprica-x81-checkpoint-100000.zip) |
| Darkmatter | GPT-2 Small | 400000 | 1.8G | [download](https://storage.googleapis.com/mistral-models/gpt2-small/darkmatter-gpt2-small-x343/darkmatter-x343-checkpoint-400000.zip) |
| Darkmatter | GPT-2 Small | 200000 | 1.8G | [download](https://storage.googleapis.com/mistral-models/gpt2-small/darkmatter-gpt2-small-x343/darkmatter-x343-checkpoint-200000.zip) |
| Darkmatter | GPT-2 Small | 100000 | 1.8G | [download](https://storage.googleapis.com/mistral-models/gpt2-small/darkmatter-gpt2-small-x343/darkmatter-x343-checkpoint-100000.zip) |
| Expanse | GPT-2 Small | 400000 | 1.8G | [download](https://storage.googleapis.com/mistral-models/gpt2-small/expanse-gpt2-small-x777/expanse-x777-checkpoint-400000.zip) |
| Expanse | GPT-2 Small | 200000 | 1.8G | [download](https://storage.googleapis.com/mistral-models/gpt2-small/expanse-gpt2-small-x777/expanse-x777-checkpoint-200000.zip) |
| Expanse | GPT-2 Small | 100000 | 1.8G | [download](https://storage.googleapis.com/mistral-models/gpt2-small/expanse-gpt2-small-x777/expanse-x777-checkpoint-100000.zip) |

---

## Issues

To ask questions, report issues, or request features, please use the [GitHub Issue Tracker](https://github.com/stanford-mercury/mistral/issues). Before creating a new issue, please make sure to search for existing issues that may solve your problem.

---

## Contributing

Please see our [Read The Docs](https://nlp.stanford.edu/local/mistral/docs/_build/html/contributing.html) page for info on contributing.
