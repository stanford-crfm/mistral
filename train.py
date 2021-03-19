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
import os
import random
from datetime import datetime

import numpy as np
import torch
from quinine import QuinineArgumentParser
from transformers import default_data_collator
from transformers.trainer_utils import get_last_checkpoint

from conf.train_schema import get_schema
from src.args import get_training_arguments
from src.corpora import get_auto_dataset
from src.models import get_auto_clm_tokenizer
from src.overwatch import get_overwatch
from src.util import create_paths, set_permissions
from src.util.callbacks import CustomWandbCallback, compute_metrics
from src.util.registry import ONLINE_EVAL_DATA_REGISTRY
from src.util.trainer import OnlineBenchmarkTrainer


def train() -> None:
    # Parse Quinfig (via Quinine Argparse Binding)
    print("[*] Mercury :: Launching the Bastard =>>> \N{rocket} \N{see-no-evil monkey} \N{rocket}")
    print('\t=>> "This wind, it is not an ending..." (Robert Jordan - A Memory of Light)')
    quinfig = QuinineArgumentParser(schema=get_schema()).parse_quinfig()

    # Create Unique Run Name (for Logging, Checkpointing, and W&B) :: Initialize all Directories
    run_id = quinfig.run_id
    if run_id is None:
        run_id = (
            f"{quinfig.model.id}-d={quinfig.dataset.id}-n={quinfig.infra.nodes}-g={quinfig.infra.gpus}+"
            f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
        )
    paths = create_paths(run_id, quinfig.model.id, quinfig.artifacts.run_dir, quinfig.artifacts.cache_dir)

    # Overwatch :: Setup & Configure Console/File Logger --> Handle Process 0 vs. other Process Logging!
    overwatch = get_overwatch(paths["runs"] / f"{run_id}.log", quinfig.log_level, rank=quinfig.infra.rank)
    overwatch.info(f"Starting Run: {run_id}...")

    # Set Randomness
    overwatch.info(f"Setting Random Seed to {quinfig.seed}!")
    random.seed(quinfig.seed)
    np.random.seed(quinfig.seed)
    torch.manual_seed(quinfig.seed)

    last_checkpoint, resume_run_id = None, None
    if quinfig.resume:
        last_checkpoint = get_last_checkpoint(paths["runs"])
        resume_run_id = os.readlink(paths["runs"] / "wandb" / "latest-run").split("-")[-1]
        assert last_checkpoint is not None, "Cannot detect checkpoint in run_dir -- Resuming Failed!"
        overwatch.info(f"Checkpoint detected, Resuming Training at `{last_checkpoint}`.")

    # Instantiate Pretrained Tokenizer and Initialize AutoModel (GPT-2) from Arguments
    overwatch.info(f"Building Tokenize and Initializing `{quinfig.model.id}` via AutoModel/AutoConfig...")
    model, tokenizer = get_auto_clm_tokenizer(
        quinfig.model.id,
        paths,
        gradient_checkpointing=quinfig.model.gradient_checkpointing,
        use_pretrained_tokenizer=quinfig.model.pretrained_tokenizer,
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
            overwatch.info(f"Downloading and Preprocessing Online Eval Datset {dataset_name}")
            custom_eval_datasets[dataset_name] = ONLINE_EVAL_DATA_REGISTRY[dataset_name]["generator"](
                tokenizer,
                paths,
                dataset_id=ONLINE_EVAL_DATA_REGISTRY[dataset_name]["id"],
                dataset_name=ONLINE_EVAL_DATA_REGISTRY[dataset_name]["name"],
                validation_ratio=quinfig.dataset.validation_ratio,
                seq_len=quinfig.model.seq_len,
                stride=quinfig.online_eval.stride,
                preprocessing_num_proc=quinfig.dataset.num_proc,
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
        local_rank=quinfig.infra.rank,
        effective_bsz=quinfig.effective_bsz,
        nodes=quinfig.infra.nodes,
        gpus_per_node=quinfig.infra.gpus,
    )

    # Important - Note that by default if multiple GPUs available on node, HF.Trainer defaults to `torch.DataParallel`
    #   which is almost always worse in efficiency than the DDP equivalent. So basically, always run with DDP!
    # TODO 21 :: Set up DDP (Single-Node), DDP (Multi-Node) Training + Mixed Precision Training
    # TODO 22 :: Setup DeepSpeed Training
    # TODO 23 :: Setup FairScale Training

    # Initialize Trainer, with the relevant arguments
    # TODO 32 :: Make sure we're using the right opt/schedule... should be configured by `training_args` so check!
    # TODO 33 :: Pass in `compute_metrics` for correct evaluation metrics --> Perplexity! Do during train as well?
    overwatch.info("Initializing Model Trainer...")
    trainer = OnlineBenchmarkTrainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["validation"],
        custom_eval_datasets=custom_eval_datasets,
        tokenizer=tokenizer,
        data_collator=default_data_collator,  # De Facto Collator uses Padding, which we DO NOT want!
        compute_metrics=compute_metrics,
        callbacks=[
            CustomWandbCallback(
                quinfig.wandb,
                json_file=str(paths["runs"] / "metrics.json"),
                resume=quinfig.resume,
                resume_run_id=resume_run_id,
                wandb_dir=str(paths["runs"]),
            )
        ],
    )

    # Training Time!
    overwatch.info("Training...")
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()

    overwatch.info("...and that's all folks!")


if __name__ == "__main__":
    train()
