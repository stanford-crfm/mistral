#!/bin/sh

if  [ "$CONDA_DEFAULT_ENV" != "base" ]; then
    echo "Error: run setup from base environment!"
    exit
fi
echo "Creating mistral conda environment!"
conda create -y -n mistral --file conda-requirements.txt -c pytorch
. $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate mistral
if [ "$CONDA_DEFAULT_ENV" = "mistral" ]; then
    echo "Installing python dependencies with pip!" 
    pip install -r pip-requirements.txt
fi
echo "Successfully created mistral environment!"
