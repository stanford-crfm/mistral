# mistral-gcp-gpt2-small.sh
#   Mistral GPT-2 Small Full Run with the DeepSpeed ZeRO-2 Optimizer, Per-Device Batch Size of 16 on Google Cloud with
#   MegaGPU Instances.

# Parse Named Command Arguments::
#   EX: bash mistral-gcp-gpt2-small.sh MODEL="alias" RESUME="true"
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
if [ -z "$MODEL" ]; then MODEL='alias'; fi
if [ -z "$RESUME" ]; then RESUME='false'; fi

echo "MODEL = $MODEL"
echo "RESUME = $RESUME"

# Constants
GCP_CONFIG="--config conf/gpt2-mistral-small-gcp-config.yaml";
if [ "$RESUME" == "true" ];
then
  RES="--resume true";
else
  RES="";
fi

INFRA="--nnodes 1 --nproc_per_node 16"

# Batch Size
D_BSZ_8="--training_arguments.fp16 true --training_arguments.per_device_train_batch_size 8"

# DeepSpeed Training Configuration
DS_Z2="--training_arguments.deepspeed conf/deepspeed/z2-small-conf.json"

# Random Seeds -- Alias :: 21, Battlestar :: 49, Caprica :: 81, Darkmatter :: 343, Expanse :: 777
case $MODEL in
   alias)
     SEED="--seed 21"
     RUN_ID="--run_id alias-prime-gpt2-small-x21"
     ;;
   battlestar)
     SEED="--seed 49"
     RUN_ID="--run_id battlestar-prime-gpt2-small-x49"
     ;;
   caprica)
     SEED="--seed 81"
     RUN_ID="--run_id caprica-prime-gpt2-small-x81"
     ;;
   darkmatter)
     SEED="--seed 343"
     RUN_ID="--run_id darkmatter-prime-gpt2-small-x343"
     ;;
   expanse)
     SEED="--seed 777"
     RUN_ID="--run_id expanse-prime-gpt2-small-x777"
     ;;
   firefly)
     SEED="--seed 801"
     RUN_ID="--run_id firefly-prime-gpt2-small-x801"
     ;;
   gundam)
     SEED="--seed 837"
     RUN_ID="--run_id gundam-prime-gpt2-small-x837"
     ;;
   highlander)
     SEED="--seed 900"
     RUN_ID="--run_id highlander-prime-gpt2-small-x900"
     ;;
   ?)
     usage
     exit
     ;;
 esac

# Set DeepSpeed Launcher Parameters
DISTRIBUTED_ARGS="--num_gpus 16 --num_nodes 1"

# ---

# Single-Node DS-Z2, Linear LR Schedule, Device BSZ = 16 --> Cleanup --> Seed
echo deepspeed $DISTRIBUTED_ARGS train.py $GCP_CONFIG $INFRA $D_BSZ_8 $SEED $RES $DS_Z2 $RUN_ID
deepspeed $DISTRIBUTED_ARGS train.py $GCP_CONFIG $INFRA $D_BSZ_8 $SEED $RES $DS_Z2 $RUN_ID
pkill -f "train.py"
sleep 3
