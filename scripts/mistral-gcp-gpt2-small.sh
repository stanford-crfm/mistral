# mistral-gcp-gpt2-small.sh
#   Mistral GPT-2 Small Full Run with the DeepSpeed ZeRO-2 Optimizer, Per-Device Batch Size of 16 on Google Cloud with
#   MegaGPU Instances.

# Parse Named Command Arguments::
#   EX: bash mistral-gcp-gpt2-small.sh MODEL="cyclone" RESUME="true"
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
if [ -z "$MODEL" ]; then MODEL='downpour'; fi
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
D_BSZ_16="--training_arguments.fp16 true --training_arguments.per_device_train_batch_size 16"

# DeepSpeed Training Configuration
DS_Z2="--training_arguments.deepspeed conf/deepspeed/z2-conf.json"

# Random Seeds -- Aurora :: 21, Blizzard :: 49, Cyclone :: 81, Downpour :: 343, Eddy :: 777
case $MODEL in
   aurora)
     SEED="--seed 21"
     RUN_ID="--run_id aurora-gpt2-small-x21"
     ;;
   blizzard)
     SEED="--seed 49"
     RUN_ID="--run_id blizzard-gpt2-small-x49"
     ;;
   cyclone)
     SEED="--seed 81"
     RUN_ID="--run_id cyclone-gpt2-small-x81"
     ;;
   downpour)
     SEED="--seed 343"
     RUN_ID="--run_id downpour-gpt2-small-x343"
     ;;
   eddy)
     SEED="--seed 777"
     RUN_ID="--run_id eddy-gpt2-small-x777"
     ;;
   flashflood)
     SEED="--seed 801"
     RUN_ID="--run_id flashflood-gpt2-small-x801"
     ;;
   gale)
     SEED="--seed 837"
     RUN_ID="--run_id gale-gpt2-small-x837"
     ;;
   haze)
     SEED="--seed 900"
     RUN_ID="--run_id haze-gpt2-small-x900"
     ;;
   icestorm)
     SEED="--seed 999"
     RUN_ID="--run_id icestorm-gpt2-small-999"
     ;;
   jetstream)
     SEED="--seed 1080"
     RUN_ID="--run_id jetstream-gpt2-small-x1080"
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
echo deepspeed $DISTRIBUTED_ARGS train.py $GCP_CONFIG $INFRA $D_BSZ_16 $SEED $RES $DS_Z2 $RUN_ID
deepspeed $DISTRIBUTED_ARGS train.py $GCP_CONFIG $INFRA $D_BSZ_16 $SEED $RES $DS_Z2 $RUN_ID
pkill -f "train.py"
sleep 3
