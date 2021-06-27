# mistral-gpt2-medium.sh
#   Mistral GPT-2 Medium Full Run with the DeepSpeed ZeRO-2 Optimizer, Per-Device Batch Size of 4. Runs locally, on
#   Sphinx Cluster.
#
# Parse Named Command Arguments::
#   EX: bash mistral-gpt2-medium.sh MODEL="arwen"
for ARGUMENT in "$@"
do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)

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
SPHINX_CONFIG="--config conf/gpt2-mistral-medium-config.yaml"
if [ "$RESUME" == "true" ];
then
  RES="--resume true";
else
  RES="";
fi
INFRA="--nnodes 2 --nproc_per_node 8"

# Batch Size
D_BSZ_4="--training_arguments.fp16 true --training_arguments.per_device_train_batch_size 4"

# DeepSpeed Training Configuration
DS_Z2="--training_arguments.deepspeed conf/deepspeed/z2-medium-conf.json"

# Random Seeds -- Arwen :: 21, Beren :: 49, Celebrimbor :: 81, Durin :: 343, Eowyn :: 777
case $MODEL in
   arwen)
     SEED="--seed 21"
     RUN_ID="--run_id arwen-prime-gpt2-medium-x21"
     ;;
   beren)
     SEED="--seed 49"
     RUN_ID="--run_id beren-prime-gpt2-medium-x49"
     ;;
   celebrimbor)
     SEED="--seed 81"
     RUN_ID="--run_id celebrimbor-prime-gpt2-medium-x81"
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
DISTRIBUTED_ARGS="--num_gpus 8 --num_nodes 2 --master_addr sphinx1.stanford.edu"

# ---

# Multi-Node DS-Z2, Linear LR Schedule, Device BSZ = 4 --> Cleanup --> Sleep
echo deepspeed $DISTRIBUTED_ARGS train.py $SPHINX_CONFIG $INFRA $D_BSZ_4 $SEED $RES $DS_Z2 $RUN_ID
deepspeed $DISTRIBUTED_ARGS train.py $SPHINX_CONFIG $INFRA $D_BSZ_4 $SEED $RES $DS_Z2 $RUN_ID
pkill -f "train.py"
sleep 3
