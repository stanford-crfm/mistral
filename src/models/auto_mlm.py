"""
auto_mlm.py

Default Masked Language Model (MLM) & Tokenizer Specification and Initialization. Downloads Model Configuration (if
necessary) from the  Hugging Face `transformers` Hub, instantiates pretrained Tokenizer, and initializes model using
the necessary AutoModel class.
"""
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, PreTrainedTokenizer
from transformers.models.bert import BertConfig

from ..util import REGISTRY


# Nest Overwatch under root `mistral` logger, inheriting formatting!
overwatch = logging.getLogger("mistral.models.auto")


def get_auto_mlm_tokenizer(
    model_id: str,
    paths: Dict[str, Path],
    model_configs: dict = None,
    gradient_checkpointing: bool = True,
    use_pretrained_tokenizer: bool = True,
    reorder_and_upcast_attn: bool = True,
    scale_attn_by_inverse_layer_idx: bool = True,
    initial_weights: str = None,
) -> Tuple[AutoModelForMaskedLM, PreTrainedTokenizer]:
    """Download/Load AutoConfig and Instantiate Corresponding Model and Tokenizer."""

    # Create Configuration
    if "bert" in model_id and model_configs:
        overwatch.info(f"Building Hugging Face BERTConfig from provided configs: {model_configs} ...")
        config = BertConfig.from_dict(model_configs)
    else:
        overwatch.info(f"Fetching Hugging Face AutoConfig for Model: `{REGISTRY[model_id]}`...")
        config = AutoConfig.from_pretrained(REGISTRY[model_id], cache_dir=paths["configs"])

    # IMPORTANT :: Set `use_cache` to False -- we don't need it ever and it conflicts with gradient checkpointing!
    config.use_cache = False

    # Create Tokenizer
    overwatch.info(f"Fetching Hugging Face [Fast] AutoTokenizer for Model: `{REGISTRY[model_id]}`...")
    if use_pretrained_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(REGISTRY[model_id], config=config, cache_dir=paths["tokenizer"])
    else:
        overwatch.error("Tokenizer Training/Initialization (from Scratch) not yet implemented!")
        raise NotImplementedError()

    overwatch.info(f"Initializing Tabula Rasa Model from Configuration: `{REGISTRY[model_id]}`...")
    model = AutoModelForMaskedLM.from_config(config)

    # Run GPT-Specific Initialization, if applicable
    model.resize_token_embeddings(len(tokenizer))

    # If `initial_weights` is not None, load weights from path!
    if initial_weights is not None:
        overwatch.info(f"Initializing Weights from File: `{initial_weights}`...")
        model.load_state_dict(torch.load(initial_weights, map_location=torch.device("cpu")))

    return model, tokenizer
