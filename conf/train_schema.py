"""
train_schema.py

Cerberus schema used by Quinine for train.py.
"""
import logging
from typing import Any, Dict

from quinine.common.cerberus import (
    default,
    merge,
    nullable,
    required,
    schema,
    stdict,
    tboolean,
    tfloat,
    tinteger,
    tlist,
    tstring,
)


def deprecated_field(msg):
    """Can be used in a schema to indicate that a field has been deprecated."""
    def _deprecated_field(field, value, _error):
        if value is not None:
            logging.warning(f"{field} is deprecated and will be removed in a future release.")
            if msg:
                logging.warning(msg)
    return {"check_with": _deprecated_field}


def get_schema() -> Dict[str, Any]:
    """Get the Cerberus schema for the Quinine config used in train.py."""

    # Schema for Dataset
    data_schema = {
        "id": merge(tstring, required),
        "name": merge(tstring, nullable, default(None)),
        "validation_ratio": merge(tfloat, default(0.0005)),
        "num_proc": merge(tinteger, default(64)),
        "eval_num_proc": merge(tinteger, default(4)),
    }

    # Schema for Model
    model_schema = {
        "id": merge(tstring, required),
        "gradient_checkpointing": merge(tboolean, nullable, default(None),
                                        deprecated_field("This config is now in training_arguments to better match HF.")),
        "pretrained_tokenizer": merge(tboolean, default(True)),
        "seq_len": merge(tinteger, default(1024)),
        "reorder_and_upcast_attn": merge(tboolean, nullable, default(True)),
        "scale_attn_by_inverse_layer_idx": merge(tboolean, nullable, default(True)),
        "initial_weights": merge(tstring, nullable, default(None)),
        "config_path": merge(tstring, nullable, default(None)),
    }

    # Schema for Huggingface Trainer and Training Arguments
    trainer_schema = {
        "output_dir": merge(tstring, nullable, default(None)),
        "do_train": merge(tboolean, default(True)),
        "evaluation_strategy": merge(tstring, default("steps")),
        "per_device_train_batch_size": merge(tinteger, default(2)),
        "per_device_eval_batch_size": merge(tinteger, default(8)),
        "gradient_accumulation_steps": merge(tinteger, default(1)),
        "prediction_loss_only": merge(tboolean, default(True)),
        "learning_rate": merge(tfloat, default(5.0e-5)),
        "weight_decay": merge(tfloat, default(0.01)),
        "adam_beta1": merge(tfloat, default(0.9)),
        "adam_beta2": merge(tfloat, default(0.999)),
        "adam_epsilon": merge(tfloat, default(1.0e-8)),
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
        "ignore_data_skip": merge(tboolean, default(False)),
        "seed": merge(tinteger, default(42)),
        "fp16": merge(tboolean, default(True)),
        "fp16_backend": merge(tstring, default("auto")),
        "sharded_ddp": merge(tstring, nullable, default(None)),
        "deepspeed": merge(tstring, nullable, default(None)),
        "dataloader_num_workers": merge(tinteger, default(4)),
        "local_rank": merge(tinteger, nullable, default(None)),
        "gradient_checkpointing": merge(tboolean, default(False)),
    }

    # Schema for Online Custom Evaluation Datasets (e.g. LAMBADA)
    online_eval_schema = {
        "do_wikitext": merge(tboolean, default(True)),
        "do_lambada": merge(tboolean, default(True)),
        "stride": merge(tinteger, default(512)),
    }

    # Schema for Storing Training and Data Artifacts
    artifacts_schema = {
        "cache_dir": merge(tstring, default("/u/scr/nlp/mercury/mistral/artifacts")),
        "run_dir": merge(tstring, default("/u/scr/nlp/mercury/mistral/runs")),
    }

    # Combined Schema for `train.py`
    mistral_schema = {
        "dataset": stdict(data_schema),
        "model": stdict(model_schema),
        "training_arguments": stdict(trainer_schema),
        "online_eval": stdict(online_eval_schema),
        "artifacts": stdict(artifacts_schema),
        "effective_bsz": merge(tinteger, default(512)),
        "resume": merge(tboolean, default(False)),
        "resume_checkpoint": merge(tstring, nullable, default(None)),
        "checkpoint_frequency": merge(merge(tlist, schema(merge(tlist, schema(tinteger)))), nullable, default(None)),
        "log_level": merge(tinteger, default(20)),
        "run_id": merge(tstring, nullable, default(None)),
        "wandb_api_key_path": merge(tstring, nullable, default(None)),
        "wandb": merge(tstring, nullable, default(None)),
        "group": merge(tstring, nullable, default(None)),
        "seed": merge(tinteger, default(42)),
        "run_training": merge(tboolean, default(True)),
        "run_final_eval": merge(tboolean, default(True)),
        "use_gpu": merge(tboolean, default(True)),
        # Infra Params - Passed in from `torch.distributed`
        "local_rank": merge(tinteger, default(-1)),
        "nnodes": merge(tinteger, default(-1)),
        "nproc_per_node": merge(tinteger, default(-1)),
        # Infra Params - Passed in from DeepSpeed
        "num_gpus": merge(tinteger, default(-1)),
        "num_nodes": merge(tinteger, default(-1)),
        "world_size": merge(tinteger, default(-1)),
    }

    return mistral_schema
