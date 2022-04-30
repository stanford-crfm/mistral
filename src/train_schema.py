"""
train_schema.py

Defines dataclass hparams for use with yahp
"""
import dataclasses
from dataclasses import dataclass
import typing
from typing import Optional, List, Union, Dict, Any

import yahp as hp
from transformers import TrainingArguments


@dataclass
class DatasetSourceHparams(hp.Hparams):
    urls: List[str] = hp.required("urls of the dataset. Supports braceexpand")
    json_text_key: str = hp.optional("key of the json text", default="text")
    extra_fsspec_args: Dict[str, Any] = hp.optional("fsspec args. Use for s3, gs, etc.", default_factory=lambda: {})

    def validate(self):
        pass

    def initialize_object(self):
        return self


@dataclass
class DatasetHparams(hp.Hparams):
    train_sources: Dict[str, DatasetSourceHparams] = hp.optional("source hparams", default_factory=lambda: {})
    train_weights: Optional[Dict[str, float]] = hp.optional("dict of [str, float] for weights. If None, then a strict "
                                                      "alternation is used.", default=None)

    validation_sources: Dict[str, DatasetSourceHparams] = hp.optional(
        "validation source hparams. Perplexities will be represented separately per-source", default_factory=lambda: {})

    cycle: bool = hp.optional(
        "Whether to cycle the dataset during training. This is useful for training.",
        default=False,
    )
    shuffle: bool = hp.optional(
        "Whether to shuffle the samples in the train dataset. Currently, shards are assigned and consumed with deterministic "
        "per-device shard order, but shuffling affects the order of samples via (per-device) shuffle buffers.",
        default=True,
    )
    shuffle_buffer_size: int = hp.optional(
        "If `shuffle=True`, samples are read into a buffer of this size (per-device), and randomly sampled from there "
        "to produce shuffled samples.",
        default=10000,
    )
    drop_last: bool = hp.optional(
        "Whether to drop the last samples for the last batch.", default=True
    )

    def validate(self):
        # Until https://github.com/mosaicml/yahp/issues/115 is fixed, we need to coerce what should be
        # DataSourceHparams but is actually "json" (a dict) to a DatasetSourceHparams.
        self.train_sources = {k: DatasetSourceHparams.create(data=v, cli_args=False)
                              for k, v in self.train_sources.items()} # type: ignore
        self.validation_sources = {k: DatasetSourceHparams.create(data=v, cli_args=False)
                                   for k, v in self.validation_sources.items()} # type: ignore

        super().validate()


@dataclass
class ModelHparams(hp.Hparams):
    """Hparams for Model."""
    id: str = hp.required("model id for config")
    pretrained_tokenizer: bool = hp.optional("use pretrained tokenizer", default=True)
    seq_len: int = hp.optional("sequence length for the model", default=1024)
    reorder_and_upcast_attn: bool = hp.optional("reorder and upcast attention", default=True)
    scale_attn_by_inverse_layer_idx: bool = hp.optional("scale attention by inverse layer idx", default=True)
    initial_weights: Optional[str] = hp.optional("initial weights", default=None)
    config_path: Optional[str] = hp.optional("transformer config", default=None)


# Schema for Huggingface Trainer and Training Arguments
# only overrides things that are different from transformers.TrainingArguments
training_argument_defaults = {
    "do_train": True,
    "evaluation_strategy": "steps",
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 8,
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 1000,
    "logging_dir": "logs",
    "logging_first_step": True,
    "logging_steps": 100,
    "eval_steps": 1000,
    "save_steps": 1000,
    "fp16": True,
    "dataloop_num_workers": 4,
    "gradient_checkpointing": False,
}

# these are set via other config
exclude_fields = {"output_dir", "gradient_accumulation_steps", "seed", "data_seed", "_n_gpu"}


def is_optional(field):
    return typing.get_origin(field) is Union and \
           type(None) in typing.get_args(field)


def convert_hf_to_hparam_field(field: dataclasses.Field, overridden_default=None):
    """ Converts a field from HuggingFace's TrainingArguments to an hparam dataclass field.
     This function does two things:
     * It renames "help" to "doc" in the metadata for fields
     * It wraps the type annotation in a Union if it is optional. (HF doesn't bother with optional in type annotations...)
    """
    help = field.metadata.get("help", "")
    default = field.default
    if overridden_default is not None:
        default = overridden_default

    if default is dataclasses.MISSING:
        is_required = True
        field_info = hp.required(doc=help)
    else:
        is_required = False
        field_info = hp.optional(doc=help, default=default)

    tpe = field.type
    if not is_required and not is_optional(tpe) and default is None:
        tpe = Optional[tpe]

    return field.name, tpe, field_info


TrainingArgumentsHparams = dataclasses.make_dataclass("TrainingArgumentsHparams", [
    convert_hf_to_hparam_field(field, training_argument_defaults.get(field.name, None))
    for field in dataclasses.fields(TrainingArguments)
    if field.name not in exclude_fields
], bases=(hp.Hparams,))


# Schema for Online Custom Evaluation Datasets (e.g. LAMBADA)
@dataclass
class OnlineEvalHparams(hp.Hparams):
    do_wikitext: bool = hp.optional(doc="Whether to run Wikitext-2 evaluation", default=True)
    do_lambada: bool = hp.optional(doc="Whether to run Lambada evaluation", default=True)
    stride: int = hp.optional(doc="Stride for evaluation", default=512)


# Schema for Storing Training and Data Artifacts
@dataclass
class ArtifactsHparams(hp.Hparams):
    cache_dir: str = hp.optional(doc="Cache directory", default="/u/scr/nlp/mercury/mistral/artifacts")
    run_dir: str = hp.optional(doc="Run directory", default="/u/scr/nlp/mercury/mistral/runs")


@dataclass
class MistralHparams(hp.Hparams):
    dataset: DatasetHparams = hp.required("Dataset parameters")
    model: ModelHparams = hp.required("Model parameters")
    training_arguments: TrainingArgumentsHparams = hp.required("Training arguments")
    online_eval: OnlineEvalHparams = hp.required("Online evaluation parameters")
    artifacts: ArtifactsHparams = hp.required("Artifacts parameters")
    effective_bsz: int = hp.optional(doc="Effective batch size", default=512)
    resume: bool = hp.optional(doc="Whether to resume training", default=False)
    resume_checkpoint: Optional[str] = hp.optional(doc="Checkpoint to resume from", default=None)
    checkpoint_frequency: List[List[int]] = hp.optional(doc="Checkpoint frequency", default_factory=lambda:[])
    log_level: int = hp.optional(doc="Logging level", default=20)
    run_id: Optional[str] = hp.optional(doc="Run ID", default=None)
    wandb_api_key_path: Optional[str] = hp.optional(doc="Path to wandb api key", default=None)
    wandb: Optional[str] = hp.optional(doc="Wandb project name", default=None)
    group: Optional[str] = hp.optional(doc="Wandb group name", default=None)
    seed: int = hp.optional(doc="Random seed", default=42)
    run_training: bool = hp.optional(doc="Whether to run training", default=True)
    run_final_eval: bool = hp.optional(doc="Whether to run final evaluation", default=True)
    use_gpu: bool = hp.optional(doc="Whether to use GPU", default=True)
    # Infra Params - Passed in from `torch.distributed`
    local_rank: int = hp.optional(doc="Local rank", default=-1)
    nnodes: int = hp.optional(doc="Number of nodes", default=-1)
    nproc_per_node: int = hp.optional(doc="Number of processes per node", default=-1)
    # Infra Params - Passed in from DeepSpeed
    num_gpus: int = hp.optional(doc="Number of GPUs", default=-1)
    num_nodes: int = hp.optional(doc="Number of nodes", default=-1)
    world_size: int = hp.optional(doc="World size", default=-1)
