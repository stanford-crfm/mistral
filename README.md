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
## Community

`Mistral` is built to facilitate transparent and accessible training. To do our best to reach this goal, **we will hold community meetings
twice a month** we'll give updates as to where we're at and what we're working on, _and more importantly, hear from you as to how we can help
and possibly work together._

We would love for folks from academia, other community efforts, as well as those in industry to join - all are welcome. The first meeting will
be on [Monday, August 30th at 4 PM PT](https://www.google.com/url?q=https://stanford.zoom.us/j/93555216759?pwd%3DSkd2eURyS0NWTGthWjVGNEJiMXJ3QT09).

We'll post the future dates (and times - which we hope to move around through the day to maximally engage folks in varied timezones)
after the first meeting!

---

## Quickstart

### Installation

The dependencies for Mistral can be installed using Conda. Note that the provided environment assumes that CUDA 11.0
is installed. You may need to adjust the environment YAML file depending on your set up.

```bash
git clone https://github.com/stanford-crfm/mistral.git
cd mistral
conda env create -f environments/environment-gpu.yaml  # Choose CUDA kernel based on the hardware!
```

If you are training on the CPU only, run `conda env create -f environments/environment-cpu.yaml` instead.


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
deepspeed --num_gpus 8 --num_nodes 2 --master_addr machine1 train.py --config conf/tutorial-gpt2-micro.yaml --nnodes 2 --nproc_per_node 8 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 4 --training_arguments.deepspeed conf/deepspeed/z1-conf.json --run_id tutorial-gpt2-micro-multi-node > tutorial-gpt2-micro-multi-node.out 2> tutorial-gpt2-micro-multi-node.err
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

---

## Resources

The Propulsion team has trained 5 GPT-2 Medium models and 5 GPT-2 Small models on the [OpenWebText corpus](https://huggingface.co/datasets/openwebtext),
as found in [🤗  datasets](https://huggingface.co/datasets).

Checkpoints can be loaded as Hugging Face models. For each model, we provide checkpoints at 100k, 200k, 300k and 400k steps.

We have also stored over 600 checkpoints for each model, subject to the following checkpoint schedule:
- Every 10 Steps, for the first 0 - 100 Steps.
- Every 50 Steps, from 100 - 2000 Steps.
- Every 100 Steps, from 2000 - 20,000 Steps.
- Every 1000 Steps, from 20,000 - 400,000 Steps.

This comes out to _610 checkpoints per run, taking up ~22TB for all 10 models_ (making it pretty expensive to host!) If you are interested in acquiring
these additional checkpoints, please [file an issue](https://github.com/stanford-crfm/mistral/issues) or contact Laurel (lorr1) and Sidd (skaramcheti)
at their @cs.stanford.edu email addresses, and we'll be happy to figure out a cost-effective solution to sharing them.

Full model checkpoints can be downloaded from [🤗 hub](https://huggingface.co/stanford-crfm).

| Run | Type | Seed |
| --- | --- | --- |
| Alias | GPT-2 Small | 21 |
| Battlestar | GPT-2 Small | 49 |
| Caprica | GPT-2 Small | 81 |
| Darkmatter | GPT-2 Small | 343 |
| Expanse | GPT-2 Small | 777 |
| Arwen | GPT-2 Medium | 21 |
| Beren | GPT-2 Medium | 49 |
| Celebrimbor | GPT-2 Medium | 81 |
| Durin | GPT-2 Medium | 343 |
| Eowyn | GPT-2 Medium | 777 |

For instance, to download the 400k checkpoint for Arwen, run this command:

```
# Make sure you have git-lfs installed
# (https://git-lfs.github.com/)
git lfs install

git clone https://huggingface.co/stanford-crfm/arwen-x21-checkpoint-400000
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
