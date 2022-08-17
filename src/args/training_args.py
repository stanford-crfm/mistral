"""
training_args.py

Utility script for unloading Quinfigs into full set of Training Arguments, as well as for handling any argument
overrides (e.g., paths that are defined at runtime, parameters that are dynamically computed such as gradient
accumulation).
"""
import logging
from pathlib import Path
from typing import Optional

from munch import Munch
from transformers import TrainingArguments


# Nest Overwatch under root `mistral` logger, inheriting formatting!
overwatch = logging.getLogger("mistral.args.training")


def get_training_arguments(
    quinfig_args: Munch,
    run_name: str,
    output_dir: Path,
    seed: int,
    local_rank: int,
    world_size: int,
    effective_bsz: int,
    gradient_checkpointing: Optional[bool] = None,
) -> TrainingArguments:
    """Initialize Training Arguments from Quinfig and Runtime-Defined Variables."""

    # `quinfig_args` already contains some default training arguments --> we'll be overwriting/adding to the Dict
    #   =>> a `Munch` is a subclass of Dictionary that supports attribute style access
    training_args = quinfig_args
    training_args.run_name = run_name
    training_args.output_dir = output_dir
    training_args.seed = seed
    training_args.data_seed = seed
    training_args.local_rank = local_rank

    # Since we Implement a Custom W&B / JSON Logging Callback, we don't report to anyone -- we've gone rogue!
    training_args.report_to = "none"

    # do it this way so we start supporting gradient_checkpointing in training_args à la Transformers
    if gradient_checkpointing is not None:
        training_args.gradient_checkpointing = gradient_checkpointing

    # If "sharded_ddp" is None --> replace with False
    if training_args.sharded_ddp is None:
        training_args.sharded_ddp = False
    else:
        assert isinstance(training_args.sharded_ddp, str) and training_args.sharded_ddp in [
            "simple",
            "zero_dp_2+auto_wrap",
            "zero_dp_2+auto_wrap+offload",
            "zero_dp_3+auto_wrap",
            "zero_dp_3+auto_wrap+offload",
        ]

        # If "+" in `sharded_ddp` --> Split, and then join... this is kinda hacky (TODO training_args.A :: Fix!)
        if "+" in training_args.sharded_ddp:
            training_args.sharded_ddp = " ".join(training_args.sharded_ddp.split("+"))

    # Compute Gradient Accumulation Dynamically
    training_args.gradient_accumulation_steps = effective_bsz // (
        quinfig_args.per_device_train_batch_size * world_size
    )
    overwatch.info(
        f"Setting Gradient Accumulation Steps = `{training_args.gradient_accumulation_steps}` [BSZ: {effective_bsz} "
        f"World Size: {world_size} Device BSZ: {quinfig_args.per_device_train_batch_size}]"
    )
    if (
        training_args.gradient_accumulation_steps <= 0
        or effective_bsz % training_args.gradient_accumulation_steps != 0
    ):
        raise ValueError("Incompatible sizes for gradient accumulation!")

    args = TrainingArguments(**training_args)

    # TODO(dlwh): report this bug to transformers
    assert (
        args.dataloader_num_workers == 0 or world_size == 1
    ), "dataloader_num_workers must be 0 for multi-gpu training in HF right now"

    return args
