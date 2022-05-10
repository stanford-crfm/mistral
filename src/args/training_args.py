"""
training_args.py

Utility script for massaging TrainingArgumentHparams into full set of Training Arguments, as well as for handling any
argument overrides (e.g., paths that are defined at runtime, parameters that are dynamically computed such as gradient
accumulation).
"""
import logging
from pathlib import Path

from transformers import TrainingArguments

from src.train_schema import TrainingArgumentsHparams


# Nest Overwatch under root `mistral` logger, inheriting formatting!
overwatch = logging.getLogger("mistral.args.training")


def get_training_arguments(
    args: TrainingArgumentsHparams,
    run_name: str,
    output_dir: Path,
    seed: int = 21,
    local_rank: int = 0,
    effective_bsz: int = 512,
    nodes: int = 1,
    gpus_per_node: int = 8,
) -> TrainingArguments:
    """Initialize Training Arguments from Hparams and Runtime-Defined Variables."""

    # `args` already contains some default training arguments, but we want to add the following:
    training_args = args
    training_args.run_name = run_name
    training_args.output_dir = output_dir
    training_args.seed = seed
    training_args.data_seed = seed
    training_args.local_rank = local_rank

    # Since we Implement a Custom W&B / JSON Logging Callback, we don't report to anyone -- we've gone rogue!
    training_args.report_to = "none"

    # If "sharded_ddp" is None --> replace with False
    if training_args.sharded_ddp is None:
        training_args.sharded_ddp = False
    else:
        assert isinstance(training_args.sharded_ddp, str) and training_args.sharded_ddp in [
            "",
            "simple",
            "zero_dp_2+auto_wrap",
            "zero_dp_2+auto_wrap+offload",
            "zero_dp_3+auto_wrap",
            "zero_dp_3+auto_wrap+offload",
        ], (
            "sharded_ddp must be one of: '', 'simple', 'zero_dp_2+auto_wrap', 'zero_dp_2+auto_wrap+offload',"
            f" 'zero_dp_3+auto_wrap', 'zero_dp_3+auto_wrap+offload', but was {training_args.sharded_ddp}"
        )

        # If "+" in `sharded_ddp` --> Split, and then join... this is kinda hacky (TODO training_args.A :: Fix!)
        if "+" in training_args.sharded_ddp:
            training_args.sharded_ddp = " ".join(training_args.sharded_ddp.split("+"))

    # Compute Gradient Accumulation Dynamically
    training_args.gradient_accumulation_steps = effective_bsz // (
        args.per_device_train_batch_size * gpus_per_node * nodes
    )
    overwatch.info(
        f"Setting Gradient Accumulation Steps = `{training_args.gradient_accumulation_steps}` [Node(s): {nodes} - "
        f"GPU(s): {gpus_per_node} - Device BSZ: {args.per_device_train_batch_size}]"
    )

    return TrainingArguments(**training_args.__dict__)
