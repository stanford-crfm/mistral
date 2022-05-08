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
from transformers.data.data_collator import default_data_collator
from transformers.trainer_utils import get_last_checkpoint

from src.args import get_training_arguments
from src.core import CustomCheckpointCallback, CustomWandbCallback, OnlineBenchmarkTrainer
from src.corpora import ONLINE_EVAL_DATA_REGISTRY, get_auto_dataset
from src.models import get_auto_clm_tokenizer
from src.overwatch import get_overwatch
from src.train_schema import MistralHparams
from src.util import create_paths, set_permissions


def train() -> OnlineBenchmarkTrainer:
    # Parse config (via Yahp Argparse Binding)
    print("[*] Mercury :: Launching =>>> \N{rocket} \N{see-no-evil monkey} \N{rocket}")
    print('\t=>> "This wind, it is not an ending..." (Robert Jordan - A Memory of Light)')
    hparams = MistralHparams.create()
    print(hparams.dumps(add_docs=True))

    # Set Distributed Arguments
    # TODO train.A :: @Laurel, @Karan -- `local_rank` not in hparams w/ torch.distributed.launch?
    print(hparams)
    hparams.world_size = int(os.getenv("WORLD_SIZE", hparams.nproc_per_node))
    hparams.local_rank = int(os.getenv("LOCAL_RANK", -1))

    # Create Unique Run Name (for Logging, Checkpointing, and W&B) :: Initialize all Directories
    run_id = hparams.run_id
    if run_id is None:
        run_id = (
            f"{hparams.model.id}-d={hparams.dataset.id}-n={hparams.nnodes}-g={hparams.nproc_per_node}-"
            f"w={hparams.world_size}+{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
        )
    paths = create_paths(run_id, hparams.model.id, hparams.artifacts.run_dir, hparams.artifacts.cache_dir)

    # Overwatch :: Setup & Configure Console/File Logger --> Handle Process 0 vs. other Process Logging!
    overwatch = get_overwatch(paths["runs"] / f"{run_id}.log", hparams.log_level, local_rank=hparams.local_rank)
    overwatch.info(f"Starting Run: {run_id}...")

    # Set Randomness
    overwatch.info(f"Setting Random Seed to {hparams.seed}!")
    random.seed(hparams.seed)
    np.random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)

    last_checkpoint, resume_run_id = None, None
    if hparams.resume:
        if hparams.resume_checkpoint is not None:
            last_checkpoint = hparams.resume_checkpoint
        else:
            # TODO train.B :: If machine fails while model is saving, checkpoint will be corrupted!
            # We need to verify the last checkpoint is loadable and if not, get the second to last checkpoint
            last_checkpoint = get_last_checkpoint(paths["runs"])
            if last_checkpoint is not None:
                resume_run_id = os.readlink(paths["runs"] / "wandb" / "latest-run").split("-")[-1]
                overwatch.info(f"Checkpoint detected, Resuming Training at `{last_checkpoint}`.")

    # Instantiate Pretrained Tokenizer and Initialize AutoModel (GPT-2) from Arguments
    overwatch.info(f"Building Tokenize and Initializing `{hparams.model.id}` via AutoModel/AutoConfig...")
    if hparams.model.config_path:
        overwatch.info(f"Using Configs For Model From: {hparams.model.config_path} ...")
        with open(hparams.model.config_path) as f:
            model_configs = json.load(f)
    else:
        model_configs = None
    model, tokenizer = get_auto_clm_tokenizer(
        hparams.model.id,
        paths,
        model_configs=model_configs,
        use_pretrained_tokenizer=hparams.model.pretrained_tokenizer,
        reorder_and_upcast_attn=hparams.model.reorder_and_upcast_attn,
        scale_attn_by_inverse_layer_idx=hparams.model.scale_attn_by_inverse_layer_idx,
        initial_weights=hparams.model.initial_weights,
    )

    # Load Dataset w/ Preprocessing, Batching, and Collating
    overwatch.info(f"Downloading and Preprocessing Dataset `{hparams.dataset.id}`...")
    lm_dataset = get_auto_dataset(
        tokenizer,
        paths,
        dataset_id=hparams.dataset.id,
        dataset_name=hparams.dataset.name,
        validation_ratio=hparams.dataset.validation_ratio,
        seq_len=hparams.model.seq_len,
        preprocessing_num_proc=hparams.dataset.num_proc,
    )

    # Load Online Eval Datasets
    custom_eval_datasets = dict()
    for eval_dataset_arg in list(
        filter(lambda x: x.startswith("do_"), hparams.online_eval.__dataclass_fields__.keys())
    ):
        if getattr(hparams.online_eval, eval_dataset_arg):
            # Dataset name is in hparams arg of "do_<dataset>" -> Boolean
            dataset_name = eval_dataset_arg.lstrip("do_")
            overwatch.info(f"Downloading and Preprocessing Online Eval Dataset {dataset_name}")
            custom_eval_datasets[dataset_name] = ONLINE_EVAL_DATA_REGISTRY[dataset_name]["generator"](
                tokenizer,
                paths,
                dataset_id=ONLINE_EVAL_DATA_REGISTRY[dataset_name]["id"],
                dataset_name=ONLINE_EVAL_DATA_REGISTRY[dataset_name]["name"],
                validation_ratio=hparams.dataset.validation_ratio,
                seq_len=hparams.model.seq_len,
                stride=hparams.online_eval.stride,
                preprocessing_num_proc=hparams.dataset.eval_num_proc,
                ignore_train=True,
            )["validation"]

    # Fix All Dataset Permissions
    set_permissions(paths)

    # Initialize Training Arguments from hparams
    overwatch.info("Setting Training Arguments from hparams...")
    training_args = get_training_arguments(
        hparams.training_arguments,
        run_name=run_id,
        output_dir=paths["runs"],
        seed=hparams.seed,
        local_rank=hparams.local_rank,
        effective_bsz=hparams.effective_bsz,
        nodes=hparams.nnodes,
        gpus_per_node=hparams.nproc_per_node,
    )

    # Initialize Trainer, with the relevant arguments
    overwatch.info("Initializing Model Trainer...")
    if hparams.local_rank <= 0:
        overwatch.info(f"Training Arguments: {training_args}")

    # Initialize Checkpoint Frequency Callback
    if hparams.checkpoint_frequency is None:
        frequencies = [[hparams.training_arguments.save_steps, hparams.training_arguments.max_steps]]
    else:
        frequencies = hparams.checkpoint_frequency

    callbacks = [
        CustomCheckpointCallback(frequencies=frequencies),
    ]
    if os.getenv("WANDB_DISABLED", "false").lower() != "true":
        callbacks.append(
            CustomWandbCallback(
                hparams.wandb,
                json_file=str(paths["runs"] / "metrics.json"),
                group=hparams.group,
                resume=hparams.resume,
                resume_run_id=resume_run_id,
                wandb_dir=str(paths["runs"]),
                api_key_path=hparams.wandb_api_key_path,
            ),
        )

    trainer = OnlineBenchmarkTrainer(
        model=model,
        args=training_args,
        data_collator=default_data_collator,  # De Facto Collator uses Padding, which we DO NOT want!
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["validation"],
        custom_eval_datasets=custom_eval_datasets,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    if hparams.local_rank <= 0 and last_checkpoint is None:
        trainer.save_model(output_dir=str(paths["runs"] / "checkpoint-0"))

    # Training Time!
    if hparams.run_training:
        overwatch.info("Training...")
        trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model()
        overwatch.info("...and that's all folks!")

    # Evaluation Time!
    if hparams.run_final_eval:
        overwatch.info("Running final evaluation...")
        if hparams.nproc_per_node > 0:
            trainer.model.to(torch.device("cuda"))
        metrics = trainer.evaluate()
        print(metrics)

    # return trainer as record of training process
    return trainer


if __name__ == "__main__":
    train()
