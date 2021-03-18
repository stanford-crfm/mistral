# Mistral

> *Mistral*: A strong and cool northwesterly wind that builds as it moves, bringing good health and clear skies.

A framework for fast and efficient large-scale language model training, built with Hugging Face :hugs:. Includes tools
and helpful scripts for incorporating new pre-training datasets, various schemes for single node and distributed
training, and importantly, scripts for evaluation and measuring bias.

A Project Mercury Endeavor.

## Contributing

If contributing to this repository, please make sure to do the following:

+ Read the instructions in [`CONTRIBUTING.md`](./CONTRIBUTING.md) - Notably, before committing to the repository, *make
sure to set up your dev environment and pre-commit install (`pre-commit install`)!*

+ Install and activate the Conda Environment using the `QUICKSTART` instructions below.

+ On installing new dependencies (via `pip` or `conda`), please make sure to update the `environment-<ID>.yaml` files
  via the following command (note that you need to separately create the `environment-cpu.yaml` file by exporting from
  your local development environment!):

  `make serialize-env arch=<cpu | gpu>`

---

## Quickstart

Clones `mistral` to the working directory, then walks through dependency setup, mostly leveraging the
`environment.yaml` files. Note, however, that because most of this work depends on bleeding edge updates to the main
`transformers` repo, you may have to refresh the `transformers` install via `pip install git+https://github.com
/huggingface/transformers`. On any shared resources (NLP Cluster, DGX Boxes) @Sidd will monitor this.

### Shared NLP Environment (Stanford Folks)

Note for @Stanford folks - the NLP Cluster (with the DGX Boxes pending) have all of the following Conda environments
already set up - the only necessary steps are cloning the repo, activating the appropriate env, and running the
`pre-commit install` command.

#### Interactive Session (from a Jagupard Machine) -- Direct Development on Cluster

```bash
cd /nlp/scr/$USER  # Replace $USER with you!
git clone https://github.com/stanford-mercury/mistral.git
cd mistral
conda activate mistral
pre-commit install  # Important!
```

### Local Development - Linux w/ GPU & CUDA 11.0

Note: Assumes that `conda` (Miniconda or Anaconda are both fine) is installed and on your path.

Ensure that you're using the appropriate `environment-<gpu | cpu>.yaml` file --> if PyTorch doesn't build properly for
your setup, checking the CUDA Toolkit is usually a good place to start. We have `environment-<gpu>.yaml` files for CUDA
11.0 (and any additional CUDA Toolkit support can be added -- file an issue if necessary).

```bash
git clone https://github.com/stanford-mercury/mistral.git
cd mistral
conda env create -f environments/environment-gpu.yaml  # Choose CUDA Kernel based on Hardware!
conda activate mistral
pre-commit install  # Important!
```

### Local Development - CPU (Mac OS & Linux)

Note: Assumes that `conda` (Miniconda or Anaconda are both fine) is installed and on your path. Use the `-cpu`
environment file.

```bash
git clone https://github.com/stanford-mercury/mistral.git
cd mistral
conda env create -f environments/environment-cpu.yaml
conda activate mistral
pre-commit install  # Important!
```

---

## Start-Up (from Scratch)

Use these commands if you're starting a repository from scratch (this shouldn't be necessary to use this repo, but is
included for completeness). If you're just trying to run/use this code, look at the Quickstart section above.

### GPU & Cluster Environments (CUDA 11.0)

```bash
conda create --name mistral python=3.8
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch   # CUDA=11.0 on most of Cluster!
conda install ipython jupyter

pip install black datasets flake8 h5py isort matplotlib pre-commit

# Install Bleeding-Edge Quinine Library!
pip install git+https://github.com/krandiash/quinine.git

# Install Bleeding-Edge Transformers Library!
pip install git+https://github.com/huggingface/transformers
```

### CPU Environments (Usually for Local Development -- Geared for Mac OS & Linux)

Similar to the above, but installs the CPU-only versions of Torch and similar dependencies.

```bash
conda create --name mistral python=3.8
conda install pytorch torchvision torchaudio -c pytorch
conda install ipython jupyter

pip install black datasets flake8 h5py isort matplotlib pre-commit

# Install Bleeding-Edge Quinine Library!
pip install git+https://github.com/krandiash/quinine.git

# Install Bleeding-Edge Transformers Library!
pip install git+https://github.com/huggingface/transformers
```

### Containerized Setup

Support for running `mistral` inside of a Docker or Singularity container is TBD. If this support is urgently required,
please file an issue!
