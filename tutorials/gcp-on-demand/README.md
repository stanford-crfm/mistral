# Run Mistral On GCP (on demand)

## Create An A100 With 8 GPU

Go to the VM instances page and click "Create Instance"

Give it an informative name (e.g. "mistral-gcp-demo")

Choose `europe-west4 (Netherlands)` as the zone

Select GPU machine, and choose NVIDIA Tesla A100, with 8 GPUs

Customize the Boot disk OS to "Deep Learning on Linux"/"Debian 10 based Deep Learning VM with CUDA 11.3 M93"

Update the size to 1 TB (or whatever you feel you need)

Hit "Create" !

Wait a few minutes, and then click the "SSH" button on the VM page. Hit "y" when asked to install drivers.

At this point the machine should be set up and operational. Run `nvidia-smi` to confirm.


## Clone Mistral

Clone the repo

```
git clone https://github.com/stanford-crfm/mistral.git
```

## Create Mistral conda environment

Follow the instructions on the main README for setting up the conda env.

Generally this will be:

```
cd setup
bash setup.sh
```

## Set Up WandB

```
cd mistral
wandb login # type in your API key at prompt
wandb init
mkdir /home/username/data # create directory for storing runs and artifacts
```

## Modify Config File

Alter the config file in `conf/gpt2-small.yaml` to customize the datasets you use.

Particularly update the `artifact` entry:

```
artifacts:
    cache_dir: /home/username/data/artifacts
    run_dir: /home/username/data/runs
```

## Launch The Training Run

This command will launch the training process with deepspeed

```
deepspeed --num_gpus 8 --num_nodes 1 --master_addr localhost --config conf/gpt2-small.yaml --nnodes 1 --nproc_per_node 8 --training_arguments.fp16 true --training_arguments.per_device_train_batch_size 4 --training_arguments.deepspeed conf/deepspeed/z2-small-conf.json --run_id mistral-june22-demo
```
