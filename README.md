# Mistral

> *Mistral*: A strong and cool northwesterly wind that builds as it moves, bringing good health and clear skies.

A framework for fast and efficient large-scale language model training, built with Hugging Face :hugs:. Includes tools
and helpful scripts for incorporating new pre-training datasets, various schemes for single node and distributed
training, and importantly, scripts for evaluation and measuring bias.

A Project Mercury Endeavor.

## Contributing

If contributing to this repository, please make sure to do the following:

+ Read the instructions in [`CONTRIBUTING.md`](./CONTRIBUTING.md)

+ Install and activate the Conda Environment using the `QUICKSTART` instructions below.

+ On installing new dependencies (via `pip` or `conda`), please make sure to update the `environment-<ID>.yaml` files
  via the following command (note that you need to separately create the `environment-cpu.yaml` file by exporting from
  your local development environment!):

  `rm environments/environment-<ID>.yaml; conda env export --no-builds |
    grep -v "^prefix: " > environments/environment-<ID>.yaml`

---

## Quickstart

Clones `mistral` to the working directory, then walks through dependency setup, mostly leveraging the
`environment.yaml` files. Note, however, that because most of this work depends on bleeding edge updates to the main
`transformers` repo, you may have to refresh the `transformers` install via `pip install git+https://github.com
/huggingface/transformers`. On any shared resources (NLP Cluster, DGX Boxes) @Sidd will monitor this.

### GPU & Cluster Environments (Shared Resources)

Ensure that you're using the appropriate `environment-<ID>.yaml` file --> if PyTorch doesn't build properly for your
setup, checking the CUDA Toolkit is usually a good place to start. We have `environment-<ID>.yaml` files for CUDA
10.1, 11 (and any additional support can be added -- file an issue if necessary).

---

## Start-Up (from Scratch)

Use these commands if you're starting a repository from scratch (this shouldn't be necessary to use this repo, but is
included for completeness). If you're just trying to run/use this code, look at the Quickstart section above.

### GPU & Cluster Environments (CUDA 10.1, 11.0)

CUDA 10.1 & 11.0 (note only CUDA Toolkit dependency version needs to change for building the below).

```bash
conda create --name mistral-10.1 python=3.8
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch   # CUDA=10.1 on NLP Cluster
conda install ipython jupyter

pip install black datasets flake8 h5py hydra-core hydra_colorlog isort matplotlib pre-commit

# Install Bleeding-Edge Transformers Library!
pip install git+https://github.com/huggingface/transformers
```

```bash
conda create --name mistral-11.0 python=3.8
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch   # CUDA=11.0 on DGX Boxes, GCP/AWS
conda install ipython jupyter

pip install black datasets flake8 h5py hydra-core hydra_colorlog isort matplotlib pre-commit

# Install Bleeding-Edge Transformers Library!
pip install git+https://github.com/huggingface/transformers
```

### CPU Environments (Usually for Local Development -- Geared for Mac OS & Linux)

Similar to the above, but installs the CPU-only versions of Torch and similar dependencies.

```bash
conda create --name mistral-cpu python=3.8
conda install pytorch torchvision torchaudio -c pytorch
conda install ipython jupyter

pip install black datasets flake8 h5py hydra-core hydra_colorlog isort matplotlib pre-commit

# Install Bleeding-Edge Transformers Library!
pip install git+https://github.com/huggingface/transformers
```

### Containerized Setup

Support for running `mistral` inside of a Docker or Singularity container is TBD. If this support is urgently required,
please file an issue!
