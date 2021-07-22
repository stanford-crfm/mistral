# Mistral

> *Mistral*: A strong and cool northwesterly wind that builds as it moves, bringing good health and clear skies.

A framework for fast and efficient large-scale language model training, built with Hugging Face :hugs:. Includes tools
and helpful scripts for incorporating new pre-training datasets, various schemes for single node and distributed
training, and importantly, scripts for evaluation and measuring bias.

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

## Start-Up (from Scratch)

Use these commands if you're starting a repository from scratch (this shouldn't be necessary to use this repo, but is
included for completeness). If you're just trying to run/use this code, look at the Quickstart section above.

## Contributing

Please see our [Read The Docs](https://nlp.stanford.edu/local/mistral/docs/_build/html/contributing.html) page for info on contributing.
