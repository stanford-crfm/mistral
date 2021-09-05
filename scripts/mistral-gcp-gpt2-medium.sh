#!/bin/bash
# mistral-gcp-gpt2-medium.sh
#   Mistral GPT-2 Medium Full Run with the DeepSpeed ZeRO-2 Optimizer, Per-Device Batch Size of 4 on Google Cloud with
#   MegaGPU Instances.

# Parse Named Command Arguments::
#   EX: bash mistral-gcp-gpt2-medium.sh MODEL="arwen" RESUME="true"
for ARGUMENT in "$@"
do

    KEY=$(echo "$ARGUMENT" | cut -f1 -d=)
    VALUE=$(echo "$ARGUMENT" | cut -f2 -d=)

    case "$KEY" in
            MODEL)              MODEL=${VALUE} ;;
            RESUME)             RESUME=${VALUE} ;;
            *)
    esac

done

# Set to Default Values if Param is not Set
if [ -z "$MODEL" ]; then MODEL='arwen'; fi
if [ -z "$RESUME" ]; then RESUME='false'; fi

echo "MODEL = $MODEL"
echo "RESUME = $RESUME"

# Constants
GCP_CONFIG="--config conf/gpt2-mistral-medium-gcp-config.yaml";
if [ "$RESUME" == "true" ];
then
  RES="--resume true";
else
  RES="";
fi

INFRA="--nnodes 1 --nproc_per_node 16"

# Batch Size (4 w/o gradient checkpointing, 8 w/ partial gradient checkpointing)
D_BSZ_4="--training_arguments.fp16 true --training_arguments.per_device_train_batch_size 4"

# DeepSpeed Training Configurations
DS_Z2="--training_arguments.deepspeed conf/deepspeed/z2-medium-conf.json"

# Random Seeds -- Arwen :: 21, Beren :: 49, Cerebrimbor :: 81, Durin :: 343, Eowyn :: 777
case $MODEL in
   arwen)
     SEED="--seed 21"
     RUN_ID="--run_id arwen-prime-gpt2-medium-x21"
     ;;
   beren)
     SEED="--seed 49"
     RUN_ID="--run_id beren-prime-gpt2-medium-x49"
     ;;
   cerebrimbor)
     SEED="--seed 81"
     RUN_ID="--run_id cerebrimbor-prime-gpt2-medium-x81"
     ;;
   durin)
     SEED="--seed 343"
     RUN_ID="--run_id durin-prime-gpt2-medium-x343"
     ;;
   eowyn)
     SEED="--seed 777"
     RUN_ID="--run_id eowyn-prime-gpt2-medium-x777"
     ;;
   ?)
     usage
     exit
     ;;
 esac

# Set DeepSpeed Launcher Parameters
DISTRIBUTED_ARGS="--num_gpus 16 --num_nodes 1"

# ---

# Single-Node DS-Z2, Linear LR Schedule, Device BSZ = 4 --> Cleanup --> Seed
echo deepspeed "$DISTRIBUTED_ARGS" train.py "$GCP_CONFIG" "$INFRA" "$D_BSZ_4" "$SEED" "$RES" "$DS_Z2" "$RUN_ID"
deepspeed "$DISTRIBUTED_ARGS" train.py "$GCP_CONFIG" "$INFRA" "$D_BSZ_4" "$SEED" "$RES" "$DS_Z2" "$RUN_ID"
pkill -f "train.py"
sleep 3
