"""
train_schema.py

Cerberus schema used by Quinine for train.py.
"""
from typing import Any, Dict

from quinine.common.cerberus import default, merge, nullable, required, stdict, tboolean, tfloat, tinteger, tstring


def get_schema() -> Dict[str, Any]:
    """ Get the Cerberus schema for the Quinine config used in train.py. """

    # Schema for Dataset
    data_schema = {
        "name": merge(tstring, nullable, default(None)),
        "id": merge(tstring, required),
        "num_proc": merge(tinteger, default(4)),
        "validation_ratio": merge(tfloat, default(0.0005)),
    }

    # Schema for Model
    model_schema = {
        "id": merge(tstring, required),
        "pretrained_tokenizer": merge(tboolean, default(True)),
        "seq_len": merge(tinteger, default(1024)),
    }

    # Schema for Run Information
    run_schema = {
        "seed": merge(tinteger, default(1234)),
        "log_level": merge(tstring, default("INFO")),
    }

    # Schema for Huggingface Trainer and Training Arguments
    trainer_schema = {
        "output_dir": merge(tstring, nullable, default(None)),
        "do_train": merge(tboolean, default(True)),
        "evaluation_strategy": merge(tstring, default("steps")),
        "per_device_train_batch_size": merge(tinteger, default(32)),
        "per_device_eval_batch_size": merge(tinteger, default(32)),
        "gradient_accumulation_steps": merge(tinteger, default(1)),
        "prediction_loss_only": merge(tboolean, default(True)),
        "learning_rate": merge(tfloat, default(2.0e-5)),
        "weight_decay": merge(tfloat, default(0.01)),
        "adam_epsilon": merge(tfloat, default(1.0e-8)),
        "adam_beta1": merge(tfloat, default(0.9)),
        "adam_beta2": merge(tfloat, default(0.999)),
        "max_grad_norm": merge(tfloat, default(1.0)),
        "max_steps": merge(tinteger, default(-1)),
        "lr_scheduler_type": merge(tstring, default("cosine")),
        "warmup_steps": merge(tinteger, default(1000)),
        "run_name": merge(tstring, nullable, default(None)),
        "logging_dir": merge(tstring, default("logs")),
        "logging_first_step": merge(tboolean, default(True)),
        "logging_steps": merge(tinteger, default(100)),
        "eval_steps": merge(tinteger, default(1000)),
        "save_steps": merge(tinteger, default(1000)),
        "seed": merge(tinteger, default(42)),
        "fp16": merge(tboolean, default(True)),
        "local_rank": merge(tboolean, nullable, default(None)),
    }

    # Schema for Training Infrastructure
    infra_schema = {
        "rank": merge(tinteger, default(-1)),
        "nodes": merge(tinteger, default(1)),
        "gpus": merge(tinteger, default(1)),
    }

    # Schema for Storing Training and Data Artifacts
    artifacts_schema = {
        "cache_dir": merge(tstring, default("/u/scr/nlp/mercury/mistral/artifacts")),
        "run_dir": merge(tstring, default("/u/scr/nlp/mercury/mistral/runs")),
    }

    # Combined Schema for `train.py`
    schema = {
        "dataset": stdict(data_schema),
        "run": stdict(run_schema),
        "model": stdict(model_schema),
        "training_arguments": stdict(trainer_schema),
        "artifacts": stdict(artifacts_schema),
        "infra": stdict(infra_schema),
        "bsz": merge(tinteger, default(2)),
        "resume": merge(tboolean, default(False)),
        "log_level": merge(tinteger, default(20)),
        "run_id": merge(tstring, nullable, default(None)),
        "wandb": merge(tstring, nullable, default(None)),
        "seed": merge(tinteger, default(21)),
        "local_rank": merge(tinteger, nullable),
    }

    return schema
