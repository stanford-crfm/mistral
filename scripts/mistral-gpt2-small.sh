# mistral-gpt2-small.sh
#   Mistral GPT-2 Small Full Run with the DeepSpeed ZeRO-2 Optimizer, Per-Device Batch Size of 16. Runs locally, on
#   Sphinx Cluster.

# Parse Named Command Arguments::
#   EX: bash mistral-gpt2-small.sh MODEL="firefly" RESUME="true"
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
if [ -z "$MODEL" ]; then MODEL='firefly'; fi
if [ -z "$RESUME" ]; then RESUME='false'; fi

echo "MODEL = $MODEL"
echo "RESUME = $RESUME"

# Constants
SPHINX_CONFIG="--file conf/gpt2-mistral-small-config.yaml"
if [ "$RESUME" == "true" ];
then
  RES="--resume true";
else
  RES="";
fi
INFRA="--nnodes 2 --nproc_per_node 8"

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
DISTRIBUTED_ARGS="--num_gpus 8 --num_nodes 2 --master_addr sphinx1.stanford.edu"

# ---

# Multi-Node DS-Z2, Linear LR Schedule, Device BSZ = 16 --> Cleanup --> Seed
echo deepspeed $DISTRIBUTED_ARGS train.py $SPHINX_CONFIG $INFRA $D_BSZ_8 $SEED $RES $DS_Z2 $RUN_ID
deepspeed $DISTRIBUTED_ARGS train.py $SPHINX_CONFIG $INFRA $D_BSZ_8 $SEED $RES $DS_Z2 $RUN_ID
pkill -f "train.py"
sleep 3
