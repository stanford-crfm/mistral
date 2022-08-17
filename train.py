"""
train.py

Run large-scale language model training across various datasets and model types, leveraging Hydra and
`torch.distributed.launch` to support multiple models, datasets, and training configurations. Includes code for
loading, preprocessing, and building `torch.Datasets` from given dataset identifier (via huggingface.Datasets),
initializing models of varying architectures, and training.

Supported Models:
    - GPT-2

Supported Datasets:
    - WikiText-103
    - OpenWebText

Provides additional scripting for logging, interfacing with Weights & Biases, and serializing/saving model checkpoints.

Reference:
    - https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_clm.py

|=>> A Project Mercury Endeavor
"""
import copy
import json
import os
import random
import time
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from quinine import QuinineArgumentParser
from transformers.trainer_utils import get_last_checkpoint

from conf.train_schema import get_schema
from src.args import get_training_arguments
from src.core import CustomCheckpointCallback, CustomWandbCallback, OnlineBenchmarkTrainer
from src.core.trainer import LMDataCollator
from src.corpora import ONLINE_EVAL_DATA_REGISTRY
from src.corpora.auto import build_indexed_dataset
from src.models import get_auto_clm_tokenizer
from src.overwatch import get_overwatch
from src.util import create_paths, set_permissions


def train() -> OnlineBenchmarkTrainer:
    # Parse Quinfig (via Quinine Argparse Binding)
    print("[*] Mercury :: Launching =>>> \N{rocket} \N{see-no-evil monkey} \N{rocket}")
    print('\t=>> "This wind, it is not an ending..." (Robert Jordan - A Memory of Light)')
    quinfig = QuinineArgumentParser(schema=get_schema()).parse_quinfig()

    # Set Distributed Arguments
    quinfig.world_size = int(os.getenv("WORLD_SIZE", quinfig.nproc_per_node))
    quinfig.local_rank = int(os.getenv("LOCAL_RANK", quinfig.local_rank))
    if quinfig.world_size == -1:
        quinfig.world_size = 1

    # Create Unique Run Name (for Logging, Checkpointing, and W&B) :: Initialize all Directories
    run_id = quinfig.run_id
    if run_id is None:
        run_id = (
            f"{quinfig.model.id}-d={quinfig.dataset.id}-n={quinfig.nnodes}-g={quinfig.nproc_per_node}-"
            f"w={quinfig.world_size}+{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
        )
    paths = create_paths(run_id, quinfig.model.id, quinfig.artifacts.run_dir, quinfig.artifacts.cache_dir)

    # Overwatch :: Setup & Configure Console/File Logger --> Handle Process 0 vs. other Process Logging!
    overwatch = get_overwatch(paths["runs"] / f"{run_id}.log", quinfig.log_level, local_rank=quinfig.local_rank)
    overwatch.info(f"Starting Run: {run_id}...")

    # Set Randomness
    overwatch.info(f"Setting Random Seed to {quinfig.seed}!")
    random.seed(quinfig.seed)
    np.random.seed(quinfig.seed)
    torch.manual_seed(quinfig.seed)

    last_checkpoint, resume_run_id = None, None
    if quinfig.resume:
        if quinfig.resume_checkpoint is not None:
            last_checkpoint = quinfig.resume_checkpoint
        else:
            # TODO train.B :: If machine fails while model is saving, checkpoint will be corrupted!
            # We need to verify the last checkpoint is loadable and if not, get the second to last checkpoint
            last_checkpoint = get_last_checkpoint(paths["runs"])
            if last_checkpoint is not None:
                resume_run_id = os.readlink(paths["runs"] / "wandb" / "latest-run").split("-")[-1]
                overwatch.info(f"Checkpoint detected, Resuming Training at `{last_checkpoint}`.")

    # Instantiate Pretrained Tokenizer and Initialize AutoModel (GPT-2) from Arguments
    overwatch.info(f"Building Tokenize and Initializing `{quinfig.model.id}` via AutoModel/AutoConfig...")
    if quinfig.model.config_path:
        overwatch.info(f"Using Configs For Model From: {quinfig.model.config_path} ...")
        with open(quinfig.model.config_path) as f:
            model_configs = json.load(f)
    else:
        model_configs = None
    model, tokenizer = get_auto_clm_tokenizer(
        quinfig.model.id,
        paths,
        model_configs=model_configs,
        use_pretrained_tokenizer=quinfig.model.pretrained_tokenizer,
        use_passthrough_tokenizer=quinfig.model.passthrough_tokenizer,
        reorder_and_upcast_attn=quinfig.model.reorder_and_upcast_attn,
        scale_attn_by_inverse_layer_idx=quinfig.model.scale_attn_by_inverse_layer_idx,
        initial_weights=quinfig.model.initial_weights,
    )

    # Initialize Training Arguments from Quinfig
    overwatch.info("Setting Training Arguments from Quinfig...")
    training_args = get_training_arguments(
        quinfig.training_arguments,
        run_name=run_id,
        output_dir=paths["runs"],
        seed=quinfig.seed,
        local_rank=quinfig.local_rank,
        world_size=quinfig.world_size,
        effective_bsz=quinfig.effective_bsz,
        gradient_checkpointing=quinfig.model.gradient_checkpointing,
    )

    # Load Dataset w/ Preprocessing, Batching, and Collating
    custom_eval_datasets, lm_dataset = load_datasets(quinfig, paths, tokenizer, overwatch)

    # Fix All Dataset Permissions
    set_permissions(paths)

    # Initialize Trainer, with the relevant arguments
    overwatch.info("Initializing Model Trainer...")
    if quinfig.local_rank <= 0:
        overwatch.info(f"Training Arguments: {training_args}")

    # Initialize Checkpoint Frequency Callback
    if quinfig.checkpoint_frequency is None:
        frequencies = [[quinfig.training_arguments.save_steps, quinfig.training_arguments.max_steps]]
    else:
        frequencies = quinfig.checkpoint_frequency

    callbacks = [
        CustomCheckpointCallback(frequencies=frequencies),
    ]
    if os.getenv("WANDB_DISABLED", "false").lower() not in ["true", "1", "yes"]:
        callbacks.append(
            CustomWandbCallback(
                quinfig.wandb,
                json_file=str(paths["runs"] / "metrics.json"),
                group=quinfig.group,
                resume=quinfig.resume,
                resume_run_id=resume_run_id,
                wandb_dir=str(paths["runs"]),
                api_key_path=quinfig.wandb_api_key_path,
            ),
        )

    trainer = OnlineBenchmarkTrainer(
        model=model,
        args=training_args,
        data_collator=LMDataCollator(tokenizer),  # De Facto Collator uses Padding, which we DO NOT want!
        dataset_name=quinfig.dataset.id,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["validation"],
        custom_eval_datasets=custom_eval_datasets,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    if quinfig.local_rank <= 0 and last_checkpoint is None:
        trainer.save_model(output_dir=str(paths["runs"] / "checkpoint-0"))

    # Training Time!
    if quinfig.run_training:
        overwatch.info("Training...")
        trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model()
        overwatch.info("...and that's all folks!")

    # Evaluation Time!
    if quinfig.run_final_eval:
        overwatch.info("Running final evaluation...")
        if quinfig.nproc_per_node > 0:
            trainer.model.to(torch.device("cuda"))
        metrics = trainer.evaluate()
        print(metrics)

    # return trainer as record of training process
    return trainer


def load_datasets(quinfig, paths, tokenizer, overwatch):
    if quinfig.world_size > 1:
        overwatch.info("Distributed Training detected. Forking preprocessing on 0th local rank...")
        _preprocess_once_per_machine(quinfig, paths, tokenizer, overwatch)

    overwatch.info(f"Downloading and Preprocessing Dataset `{quinfig.dataset.id}`...")
    lm_dataset = build_indexed_dataset(
        tokenizer,
        paths,
        dataset_id=quinfig.dataset.id,
        dataset_name=quinfig.dataset.name,
        dataset_dir=quinfig.dataset.dataset_dir,
        seq_len=quinfig.model.seq_len,
        preprocessing_num_proc=quinfig.dataset.num_proc,
        shuffle_seed=quinfig.seed,
    )

    # Load Online Eval Datasets
    custom_eval_datasets = dict()
    for eval_dataset_arg in list(filter(lambda x: x.startswith("do_"), quinfig.online_eval.keys())):
        if getattr(quinfig.online_eval, eval_dataset_arg):
            # Dataset name is in quinfig arg of "do_<dataset>" -> Boolean
            dataset_name = eval_dataset_arg.lstrip("do_")
            overwatch.info(f"Downloading and Preprocessing Online Eval Dataset {dataset_name}")
            custom_eval_datasets[dataset_name] = ONLINE_EVAL_DATA_REGISTRY[dataset_name]["generator"](
                tokenizer,
                paths,
                dataset_id=ONLINE_EVAL_DATA_REGISTRY[dataset_name]["id"],
                dataset_name=ONLINE_EVAL_DATA_REGISTRY[dataset_name]["name"],
                validation_ratio=quinfig.dataset.validation_ratio,
                seq_len=quinfig.model.seq_len,
                stride=quinfig.online_eval.stride,
                preprocessing_num_proc=quinfig.dataset.eval_num_proc,
                ignore_train=True,
            )["validation"]
    return custom_eval_datasets, lm_dataset


def _preprocess_once_per_machine(quinfig, paths, tokenizer, overwatch):
    assert quinfig.world_size > 1, "Shouldn't have forked if world_size is 1"
    import torch.distributed as dist

    # create a group for all ranks on this machine (on the annoying assumption that all machines have the same number of devices running)
    # TODO: this will not work w/ tpus I think?
    cur_group, subgroups = dist.new_subgroups()
    import multiprocessing as mp

    process: Optional[mp.Process] = None
    if cur_group.rank() == 0:
        # fork a process and do a sleep/wait for other processes to finish loading
        cloned_config = copy.deepcopy(quinfig)
        cloned_config.local_rank = 0
        cloned_config.world_size = 1
        process = mp.Process(target=load_datasets, args=(cloned_config, paths, tokenizer, overwatch))
        process.start()

    while True:
        status = [None]
        if cur_group.rank() == 0:
            status = [process.exitcode]

        dist.broadcast_object_list(status, src=0)
        if status[0] is not None:
            break

        time.sleep(10)

    if status[0] != 0:
        raise RuntimeError(f"Forked process exited with status {status[0]}")

    dist.barrier()  # make sure everyone makes it
    for subgroup in subgroups:
        dist.destroy_process_group(subgroup)


if __name__ == "__main__":
    train()
