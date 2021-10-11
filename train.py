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
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
from quinine import QuinineArgumentParser
from transformers.data.data_collator import default_data_collator
from transformers.trainer_utils import get_last_checkpoint

from conf.train_schema import get_schema
from src.args import get_training_arguments
from src.core import CustomCheckpointCallback, CustomWandbCallback, OnlineBenchmarkTrainer
from src.corpora import ONLINE_EVAL_DATA_REGISTRY, get_auto_dataset
from src.models import get_auto_clm_tokenizer
from src.overwatch import get_overwatch
from src.util import create_paths, set_permissions


def train() -> OnlineBenchmarkTrainer:
    # Parse Quinfig (via Quinine Argparse Binding)
    print("[*] Mercury :: Launching =>>> \N{rocket} \N{see-no-evil monkey} \N{rocket}")
    print('\t=>> "This wind, it is not an ending..." (Robert Jordan - A Memory of Light)')
    quinfig = QuinineArgumentParser(schema=get_schema()).parse_quinfig()

    # Set Distributed Arguments
    # TODO train.A :: @Laurel, @Karan -- `local_rank` not in Quinfig w/ torch.distributed.launch?
    quinfig.world_size = int(os.getenv("WORLD_SIZE", quinfig.nproc_per_node))
    quinfig.local_rank = int(os.getenv("LOCAL_RANK", -1))

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
        gradient_checkpointing=quinfig.model.gradient_checkpointing,
        use_pretrained_tokenizer=quinfig.model.pretrained_tokenizer,
        reorder_and_upcast_attn=quinfig.model.reorder_and_upcast_attn,
        scale_attn_by_inverse_layer_idx=quinfig.model.scale_attn_by_inverse_layer_idx,
        initial_weights=quinfig.model.initial_weights,
    )

    # Load Dataset w/ Preprocessing, Batching, and Collating
    overwatch.info(f"Downloading and Preprocessing Dataset `{quinfig.dataset.id}`...")
    lm_dataset = get_auto_dataset(
        tokenizer,
        paths,
        dataset_id=quinfig.dataset.id,
        dataset_name=quinfig.dataset.name,
        validation_ratio=quinfig.dataset.validation_ratio,
        seq_len=quinfig.model.seq_len,
        preprocessing_num_proc=quinfig.dataset.num_proc,
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

    # Fix All Dataset Permissions
    set_permissions(paths)

    # Initialize Training Arguments from Quinfig
    overwatch.info("Setting Training Arguments from Quinfig...")
    training_args = get_training_arguments(
        quinfig.training_arguments,
        run_name=run_id,
        output_dir=paths["runs"],
        seed=quinfig.seed,
        local_rank=quinfig.local_rank,
        effective_bsz=quinfig.effective_bsz,
        nodes=quinfig.nnodes,
        gpus_per_node=quinfig.nproc_per_node,
    )

    # Initialize Trainer, with the relevant arguments
    overwatch.info("Initializing Model Trainer...")
    if quinfig.local_rank <= 0:
        overwatch.info(f"Training Arguments: {training_args}")

    # Initialize Checkpoint Frequency Callback
    if quinfig.checkpoint_frequency is None:
        frequencies = [[quinfig.training_arguments.save_steps, quinfig.training_arguments.max_steps]]
    else:
        frequencies = quinfig.checkpoint_frequency

    trainer = OnlineBenchmarkTrainer(
        model=model,
        args=training_args,
        data_collator=default_data_collator,  # De Facto Collator uses Padding, which we DO NOT want!
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["validation"],
        custom_eval_datasets=custom_eval_datasets,
        tokenizer=tokenizer,
        callbacks=[
            CustomWandbCallback(
                quinfig.wandb,
                json_file=str(paths["runs"] / "metrics.json"),
                group=quinfig.group,
                resume=quinfig.resume,
                resume_run_id=resume_run_id,
                wandb_dir=str(paths["runs"]),
            ),
            CustomCheckpointCallback(frequencies=frequencies),
        ],
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


if __name__ == "__main__":
    train()
