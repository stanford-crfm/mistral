import json
import os
import subprocess
import sys

from random import shuffle

env_setup_cmd = (
    "task=pubmedqa_hf ; datadir=data/$task ; export WANDB_PROJECT=biomedical-nlp-eval ; export"
    " HF_HOME=/u/scr/nlp/data/mercury/pubmed/artifacts_medqa"
)

checkpoints = ["/u/scr/nlp/data/mercury/pubmed/mistral-abhi/downstream/mc/pubmed-gpt-2.7b-300k-steps"]


settings = json.loads(open(sys.argv[1]).read())
seeds = settings["seeds"]
experiments = []

for checkpoint in checkpoints:
    for lr in settings["lrs"]:
        for num_epochs in settings["epochs"]:
            for batch_size in settings["batch_sizes"]:
                checkpoint_name = os.path.basename(checkpoint)
                experiment_name = f"{checkpoint_name}_lr={lr}_epochs={num_epochs}_batch_size={batch_size}_task=medqa"
                new_experiment = {
                    "checkpoint": checkpoint,
                    "lr": lr,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "name": experiment_name,
                }
                experiments.append(new_experiment)

shuffle(experiments)

for experiment in experiments:
    for seed in seeds:
        lr = experiment["lr"]
        checkpoint = experiment["checkpoint"]
        num_epochs = experiment["num_epochs"]
        batch_size = experiment["batch_size"]
        name = experiment["name"] + f"-seed-{seed}"
        grad_accum = int(int(batch_size) / 8)
        output_dir = f"trash/runs/{name}"
        exp_cmd = (
            "python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 run_multiple_choice.py"
            " --use_flash true --tokenizer_name stanford-crfm/pubmed_gpt_tokenizer --model_name_or_path"
            " pubmed-gpt-2.7b-300k-steps --train_file data/medqa_usmle_hf/train.json --validation_file"
            " data/medqa_usmle_hf/dev.json --test_file data/medqa_usmle_hf/test.json --do_train --do_eval"
            " --do_predict --per_device_train_batch_size 1 --per_device_eval_batch_size 1"
            f" --gradient_accumulation_steps {grad_accum} --learning_rate {lr} --weight_decay 0.0 --warmup_ratio 0.5"
            f" --num_train_epochs {num_epochs} --max_seq_length 512 --bf16 --seed {seed} --logging_first_step"
            " --logging_steps 20 --save_strategy no --evaluation_strategy steps --eval_steps 500 --run_name"
            f" {name} --output_dir {output_dir} --overwrite_output_dir"
        )
        print(exp_cmd)
        try:
            subprocess.call(f"{env_setup_cmd} ; {exp_cmd}", shell=True)
            subprocess.call(
                f"rm -f {output_dir}/pytorch_model*bin",
                shell=True,
            )
        except:
            pass
