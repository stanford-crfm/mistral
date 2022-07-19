#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

ENV_NAME="mistral"
# if we have an arg to the script, use it as the env name
if [ $# -eq 1 ]; then
    ENV_NAME=$1
fi

if  [ "$CONDA_DEFAULT_ENV" != "base" ]; then
    echo "Error: run setup from base environment!"
    exit
fi
echo "Creating mistral conda environment '${ENV_NAME}'!"
conda create -y -n "${ENV_NAME}" --file ${SCRIPT_DIR}/conda-requirements.txt -c pytorch
. $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"
if [ "$CONDA_DEFAULT_ENV" = "${ENV_NAME}" ]; then
    echo "Installing python dependencies with pip!"
    pip install -r ${SCRIPT_DIR}/pip-requirements.txt
fi
echo "Successfully created mistral environment '${ENV_NAME}'!"
