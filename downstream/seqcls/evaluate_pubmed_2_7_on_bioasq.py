import json
import os
import subprocess
import sys

from random import shuffle

env_setup_cmd = (
    "task=bioasq_hf ; datadir=data/$task ; export WANDB_PROJECT=biomedical-nlp-eval ; export"
    " HF_HOME=/u/scr/nlp/data/mercury/pubmed/artifacts_bioasq"
)

checkpoints = ["/u/scr/nlp/data/mercury/pubmed/mistral-abhi/downstream/mc/pubmed-gpt-2.7b-300k-steps"]


settings = []
for line in open(sys.argv[1]):
    if not line:
        continue
    settings.append(json.loads(line))

experiments = []

for setting in settings:
    for checkpoint in checkpoints:
        for lr in setting["lrs"]:
            for num_epochs in setting["epochs"]:
                for batch_size in setting["batch_sizes"]:
                    seq_len = seq_len = 1024 if not "seq_len" in setting else setting["seq_len"]
                    checkpoint_name = os.path.basename(checkpoint)
                    experiment_name = (
                        f"{checkpoint_name}_lr={lr}_epochs={num_epochs}_batch_size={batch_size}_seq_len={seq_len}_task=pubmedqa"
                    )
                    new_experiment = {
                        "checkpoint": checkpoint,
                        "lr": lr,
                        "batch_size": batch_size,
                        "num_epochs": num_epochs,
                        "name": experiment_name,
                        "seeds": setting["seeds"],
                        "seq_len": seq_len
                    }
                    experiments.append(new_experiment)

shuffle(experiments)

for experiment in experiments:
    for seed in experiment["seeds"]:
        lr = experiment["lr"]
        checkpoint = experiment["checkpoint"]
        num_epochs = experiment["num_epochs"]
        batch_size = experiment["batch_size"]
        name = experiment["name"] + f"-seed-{seed}"
        grad_accum = int(int(batch_size) / 8)
        exp_cmd = (
            "python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 run_seqcls_gpt.py"
            f" --tokenizer_name stanford-crfm/pubmed_gpt_tokenizer --model_name_or_path {checkpoint} --train_file"
            " $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json --do_train"
            " --do_eval --do_predict --per_device_train_batch_size 1 --gradient_accumulation_steps"
            f" {grad_accum} --learning_rate {lr} --warmup_ratio 0.5 --num_train_epochs {num_epochs}  --max_seq_length"
            f" {seq_len}  --logging_steps 100 --save_strategy no --evaluation_strategy no --output_dir"
            f" /u/scr/nlp/data/mercury/pubmed/pubmed_gpt_2_7_b_eval/bioasq/runs/{name} --overwrite_output_dir --bf16"
            f" --seed {seed} --run_name {name}"
        )
        print(exp_cmd)
        try:
            subprocess.call(f"{env_setup_cmd} ; {exp_cmd}", shell=True)
            subprocess.call(
                f"rm -f /u/scr/nlp/data/mercury/pubmed/pubmed_gpt_2_7_b_eval/bioasq/runs/{name}/pytorch_model.bin",
                shell=True,
            )
        except:
            pass
