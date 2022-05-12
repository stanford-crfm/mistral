#!/bin/sh

conda create -n mistral --file conda-requirements.txt -c pytorch
. $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate mistral
pip install -r pip-requirements.txt
