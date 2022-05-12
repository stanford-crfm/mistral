#!/bin/sh

if  [ "$CONDA_DEFAULT_ENV" != "base" ]; then
    echo "Error: run setup from (base) environment!"
    exit
fi
conda create -n mistral --file conda-requirements.txt -c pytorch
. $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate mistral
pip install -r pip-requirements.txt
echo "Successfully created (mistral) !!"
