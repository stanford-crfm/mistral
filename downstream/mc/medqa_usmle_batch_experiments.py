import subprocess

env_setup_cmd = "task=medqa_usmle_hf ; datadir=data/$task ; export WANDB_PROJECT='biomedical-nlp-eval'"

experiments = [
    #{
        #"lr": "3e-5",
        #"checkpoint": "/u/scr/nlp/data/mercury/pubmed/checkpoints/xl/gpt2-xl",
        #"num_train_epochs": 30,
        #"name": "gpt-xl-english-lr=3e-5",
    #},
    #{
        #"lr": "3e-5",
        #"checkpoint": "/u/scr/nlp/data/mercury/pubmed/checkpoints/xl/pubmed_gpt_xl_no_clean_100k",
        #"num_train_epochs": 60,
        #"name": "gpt-xl-pubmed-lr=3e-5-num_epochs=60",
    #},
    {
        "lr": "3e-5",
        "checkpoint": "/u/scr/nlp/data/mercury/pubmed/checkpoints/xl/pubmed_gpt_xl_180k",
        "num_train_epochs": 30,
        "name": "pubmed_gpt_xl_180k_lr=3e-5_epochs=30",
    },
    
]

#checkpoints = ["composer-pubmed-285k", "composer-pubmed-250k", "composer-pubmed-225k", "composer-pubmed-208k", "composer-pubmed-openweb-400k", "composer-pubmed-openweb-350k", "composer-pubmed-openweb-302k", "composer-pubmed-openweb-250k", "composer-pubmed-openweb-200k", "composer-pubmed-openweb-150k", "composer-pubmed-openweb-100k"]

#experiments = [{"lr": "3e-5", "checkpoint": f"/u/scr/nlp/data/mercury/pubmed/checkpoints/small/{checkpoint}", "num_train_epochs": 30, "name": f"gpt-small-{checkpoint}"} for checkpoint in checkpoints] 

for experiment in experiments:
    lr = experiment["lr"]
    checkpoint = experiment["checkpoint"]
    num_train_epochs = experiment["num_train_epochs"]
    name = experiment["name"]
    subprocess.call(f"mkdir -p /u/scr/nlp/data/mercury/pubmed/xl-model-eval/medqa/runs/{name}", shell=True)
    exp_cmd = (
        "python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 run_multiple_choice.py"
        " --tokenizer_name gpt2 --model_name_or_path"
        f" {checkpoint} --train_file $datadir/train.json"
        " --validation_file $datadir/test.json --test_file $datadir/test.json --do_train --do_eval --do_predict"
        " --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 4"
        f" --learning_rate {lr} --warmup_ratio 0.5 --num_train_epochs {num_train_epochs} --max_seq_length 512 --fp16"
        " --logging_first_step --logging_steps 20 --save_strategy steps --evaluation_strategy steps --save_steps 300"
        " --eval_steps 100 --output_dir"
        f" /u/scr/nlp/data/mercury/pubmed/xl-model-eval/medqa/runs/{name} --overwrite_output_dir"
        f" --sharded_ddp zero_dp_2 --run_name {name}_task=medqa_usmle_hf"
    )
    print(exp_cmd)
    try:
        subprocess.call(f"{env_setup_cmd} ; {exp_cmd}", shell=True)
    except:
        pass
