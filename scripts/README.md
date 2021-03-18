# Distributed Training Setup

Distributed Training with FairScale and DeepSpeed behind Hugging Face Transformers can be a bit tricky, especially on
our shared cluster environment. Here are the steps we took to get things working:

## Single-Node DDP Setup

This works out-of-the-box, and didn't require any special installation. There is currently a weird issue where
running with `torch.distributed.launch` doesn't actually transfer `local_rank` to the base Quinfig. We have an open
issue, hopefully will be resolved soon.

Everything else seems to work as desired (including logging).

## Single-Node FairScale Setup

Cluster environment by default has several CUDA versions installed. The default CUDA (default `nvcc` used to build
FairScale, DeepSpeed) is 10.1, but Mistral is built with CUDA 11.0. We followed the Hugging Face instructions to update
our `$PATH` and `$LD_LIBRARY_PATH` prior to running the installation to reconcile this.

This **should** only need to happen once (Sidd took care of it), but if we need to update/transfer machines, follow
these instructions:

```
# On the Sphinxes
export PATH=/usr/local/cuda-11.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH

# Confirm NVCC is on CUDA 11.0
which nvcc

# Make sure `mistral` Conda Environment is Activated
conda activate mistral

# Install `fairscale` -- note that Fairscale is changing rapidly, so may need to update repeatedly.
pip install fairscale

# Install `deepspeed` -- note that DeepSpeed is also changing rapidly (but is more stable and better documented than
#   Fairscale). Usually, try to prefer DeepSpeed.
pip install deepspeed

# Verify DeepSpeed Install --> should not crash, will print stuff about JIT-compiled OPs that you can ignore.
ds_report

# Copy hostfile to /job/hostfile on Sphinxes (Unclear if we need this, but let's suppress the warning...)
cp scripts/deepspeed/hostfile /job/hostfile
```
