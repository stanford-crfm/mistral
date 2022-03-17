<div align="center"><img src="https://github.com/stanford-crfm/mistral/raw/main/docs/mistral_components.png" height="300px"/></div>

# Mistral

> *Mistral*: A strong and cool northwesterly wind that builds as it moves, bringing good health and clear skies.

[![license](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-green?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

A framework for transparent and accessible large-scale language model training, built with [Hugging Face 🤗](https://huggingface.co/) . Includes tools
and helpful scripts for incorporating new pre-training datasets, various schemes for single node and distributed training - including on
cloud providers like GCP, and importantly, scripts for evaluation.

Visit our [Read the Docs](https://nlp.stanford.edu/mistral) for the full documentation.

A Propulsion Endeavor 🚀

---

## Quickstart

### Installation

Mistral has been tested with Python 3.8.12, PyTorch 1.11.0 (compiled with CUDA 11.3), CUDA 11.3, NCCL 2.10, Transformers 4.17.0, and DeepSpeed 0.6.0.

The environment can be easily built with the following commands:

```bash
conda create -n mistral python=3.8.12
conda activate mistral
conda install pytorch cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

A `.yaml` export of a tested environment is provided at `environments/environment-gpu.yaml`.

Environments and non-Python dependencies can be managed with conda, and Python dependencies can be managed with pip (note: conda was used for the PyTorch install to get the version compiled with CUDA 11.3).


### Training GPT-2 Micro

#### Prerequisites

First, make sure to update `conf/tutorial-gpt2-micro.yaml` with the directories you want to store the Hugging Face
cache and model runs.

```
# Artifacts & Caching
artifacts:
    cache_dir: /path/to/artifacts
    run_dir: /path/to/runs
```

Next, make sure that `/path/to/mistral` is on your `PYTHONPATH`.

#### Single-node single-GPU training

For single-node single-gpu training, run:

```bash
conda activate mistral
cd mistral
CUDA_VISIBLE_DEVICES=0 python train.py --config conf/tutorial-gpt2-micro.yaml --nnodes 1 --nproc_per_node 1 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 2 --run_id tutorial-gpt2-micro
```

#### Multi-node multi-GPU training with DeepSpeed

Modify `/job/hostfile` in the following way:

```
<Hostname of first machine> slots=<Number of GPUs>
<Hostname of second machine> slots=<Number of GPUs>
...
<Hostname of the nth machine> slots=<Number of GPUs>
```

Below is an example hostfile where we train on `machine1` and `machine2` with 8 GPUs each:

```
machine1 slots=8
machine2 slots=8
```

To start distributed training, run:

```bash
conda activate mistral
cd mistral
deepspeed --num_gpus 8 --num_nodes 2 --master_addr machine1 train.py --config conf/tutorial-gpt2-micro.yaml --nnodes 2 --nproc_per_node 8 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 4 --training_arguments.deepspeed conf/deepspeed/z2-small-conf.json --run_id tutorial-gpt2-micro-multi-node
```

Note: You may need to adjust your batch size depending on the capacity of your GPUs.

If you are interested in training a model on Google Cloud, check out our
[Google Cloud + Kubernetes Tutorial](https://nlp.stanford.edu/mistral/tutorials/gcp_plus_kubernetes.html).

### Using the model

Model checkpoints will be stored in the directory specified by the `artifacts.run_dir`. An example checkpoint might be
in `/path/to/runs/tutorial-gpt2-micro/checkpoint-1000`.

Mistral stores model checkpoints in the Hugging Face format, so models can be loaded and used in the same manner as if
one had trained the model with Hugging Face.

For instance, to generate text with 🤗  Transformers:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model = GPT2LMHeadModel.from_pretrained("stanford-crfm/eowyn-x777-checkpoint-400000")

input_ids = tokenizer.encode(
    "Hello world, this is a language model prompt.", return_tensors="pt"
)

sample_output = model.generate(input_ids, do_sample=True, max_length=50, top_k=50)

print("Output:\n" + 100 * "-")
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))
```

Check out this [Google CoLab Notebook](https://colab.research.google.com/github/stanford-crfm/mistral/blob/main/generate_text.ipynb) to run
this demo!

---

## Resources

The Propulsion team has trained 5 GPT-2 Medium models and 5 GPT-2 Small models on the [OpenWebText corpus](https://huggingface.co/datasets/openwebtext),
as found in [🤗  datasets](https://huggingface.co/datasets).

Each model has 600 checkpoints, subject to the following checkpoint schedule:

- Every 10 Steps, for the first 0 - 100 Steps.
- Every 50 Steps, from 100 - 2000 Steps.
- Every 100 Steps, from 2000 - 20,000 Steps.
- Every 1000 Steps, from 20,000 - 400,000 Steps.

Checkpoints can be downloaded from [🤗 hub](https://huggingface.co/stanford-crfm).

| Run | Type | Seed | Download |
| --- | --- | --- | --- |
| Alias | GPT-2 Small | 21 | [download](https://huggingface.co/stanford-crfm/alias-gpt2-small-x21/tree/main) |
| Battlestar | GPT-2 Small | 49 | [download](https://huggingface.co/stanford-crfm/battlestar-gpt2-small-x49/tree/main) |
| Caprica | GPT-2 Small | 81 | [download](https://huggingface.co/stanford-crfm/caprica-gpt2-small-x81/tree/main) |
| Darkmatter | GPT-2 Small | 343 | [download](https://huggingface.co/stanford-crfm/darkmatter-gpt2-small-x343/tree/main) |
| Expanse | GPT-2 Small | 777 | [download](https://huggingface.co/stanford-crfm/expanse-gpt2-small-x777/tree/main) |
| Arwen | GPT-2 Medium | 21 | [download](https://huggingface.co/stanford-crfm/arwen-gpt2-medium-x21/tree/main) |
| Beren | GPT-2 Medium | 49 | [download](https://huggingface.co/stanford-crfm/beren-gpt2-medium-x49/tree/main) |
| Celebrimbor | GPT-2 Medium | 81 | [download](https://huggingface.co/stanford-crfm/celebrimbor-gpt2-medium-x81/tree/main) |
| Durin | GPT-2 Medium | 343 | [download](https://huggingface.co/stanford-crfm/durin-gpt2-medium-x343/tree/main) |
| Eowyn | GPT-2 Medium | 777 | [download](https://huggingface.co/stanford-crfm/eowyn-gpt2-medium-x777/tree/main) |


Each model has a distinct git repo, and each checkpoint is stored as a branch.

As an example, here's how to get the battlestar model's checkpoint for step 300000:

```
# Make sure you have git-lfs installed
# (https://git-lfs.github.com)
git lfs install

# get checkpoint 300000 for battlestar
git clone https://huggingface.co/stanford-crfm/battlestar-gpt2-small-x49 --branch checkpoint-300000 --single-branch
cd battlestar-gpt2-small-x49
git lfs pull
```

For convenience, every model and step checkpoint is listed in `mistral_models.json`.

---

## Issues

To ask questions, report issues, or request features, please use the [GitHub Issue Tracker](https://github.com/stanford-crfm/mistral/issues).
Before creating a new issue, please make sure to search for existing issues that may solve your problem.

---

## Differences between Mistral and Hugging Face

Please visit the [following page](https://nlp.stanford.edu/mistral/hugging_face_differences.html) that outlines the
differences between the two codebases.

---

## Contributing

Please see the [following page](https://nlp.stanford.edu/mistral/contributing.html) for information on contributing.
