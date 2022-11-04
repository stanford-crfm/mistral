import json
import subprocess
import sys

env_setup_cmd = "task=medqa_usmle_hf ; datadir=data/$task ; export WANDB_PROJECT='biomedical-nlp-eval'"

experiments = [json.loads(line) for line in open(sys.argv[1]).read().split("\n")[:-1]]

print(experiments[-1])


for experiment in experiments:
    lr = experiment["lr"]
    checkpoint = experiment["checkpoint"]
    grad_accum = experiment["grad_accum"]
    num_train_epochs = experiment["num_train_epochs"]
    tokenizer_setting = " --tokenizer_name gpt2 "
    if "tokenizer" in experiment:
        if experiment["tokenizer"] == "":
            tokenizer_setting = ""
        else:
            tokenizer_setting = f" --tokenizer_name {experiment['tokenizer']} "
    name = experiment["name"]
    subprocess.call(f"mkdir -p /u/scr/nlp/data/mercury/pubmed/xl-model-eval/medqa/runs/{name}", shell=True)
    exp_cmd = (
        "python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 run_multiple_choice.py"
        f" {tokenizer_setting} --model_name_or_path"
        f" {checkpoint} --train_file $datadir/train.json"
        " --validation_file $datadir/test.json --test_file $datadir/test.json --do_train --do_eval --do_predict"
        f" --per_device_train_batch_size 2 --per_device_eval_batch_size 1 --gradient_accumulation_steps {grad_accum}"
        f" --learning_rate {lr} --warmup_ratio 0.5 --num_train_epochs {num_train_epochs} --max_seq_length 512 --fp16"
        " --logging_first_step --logging_steps 20 --save_strategy no --evaluation_strategy steps "
        " --eval_steps 100 --output_dir"
        f" /u/scr/nlp/data/mercury/pubmed/xl-model-eval/medqa/runs/{name} --overwrite_output_dir"
        f" --sharded_ddp zero_dp_2 --run_name {name}_task=medqa_usmle_hf"
    )
    print(exp_cmd)
    try:
        subprocess.call(f"{env_setup_cmd} ; {exp_cmd}", shell=True)
    except:
        pass
