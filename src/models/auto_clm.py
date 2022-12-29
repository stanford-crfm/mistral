"""
auto_clm.py

Default Causal Language Model (CLM) & Tokenizer Specification and Initialization. Downloads Model Configuration (if
necessary) from the  Hugging Face `transformers` Hub, instantiates pretrained Tokenizer, and initializes model using
the necessary AutoModel class.
"""
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel

from ..util import REGISTRY


# Nest Overwatch under root `mistral` logger, inheriting formatting!
overwatch = logging.getLogger("mistral.models.auto")


def get_auto_clm_tokenizer(
    model_id: str,
    paths: Dict[str, Path],
    model_configs: dict = None,
    gradient_checkpointing: bool = True,
    use_pretrained_tokenizer: bool = True,
    reorder_and_upcast_attn: bool = True,
    scale_attn_by_inverse_layer_idx: bool = True,
    initial_weights: str = None,
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    """Download/Load AutoConfig and Instantiate Corresponding Model and Tokenizer."""

    # Create Configuration
    if "gpt2" in model_id and model_configs:
        overwatch.info(f"Building Hugging Face GPT2Config from provided configs: {model_configs} ...")
        config = GPT2Config.from_dict(model_configs)
        config.reorder_and_upcast_attn = reorder_and_upcast_attn
        config.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
    else:
        overwatch.info(f"Fetching Hugging Face AutoConfig for Model: `{REGISTRY[model_id]}`...")
        config = AutoConfig.from_pretrained(REGISTRY[model_id], cache_dir=paths["configs"])

    # IMPORTANT :: Set `use_cache` to False -- we don't need it ever and it conflicts with gradient checkpointing!
    config.use_cache = False

    # Use faster gelu
    config.activation_function = "gelu"

    # Create Tokenizer
    overwatch.info(f"Fetching Hugging Face [Fast] AutoTokenizer for Model: `{REGISTRY[model_id]}`...")
    if use_pretrained_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(REGISTRY[model_id], config=config, cache_dir=paths["tokenizer"])
    else:
        overwatch.error("Tokenizer Training/Initialization (from Scratch) not yet implemented!")
        raise NotImplementedError()

    # Partial Gradient Checkpointing (currently only supported for GPT-2 models)
    if "gpt2" in model_id:
        overwatch.info(f"Initializing Custom GPT-2 Model from Configuration: `{REGISTRY[model_id]}`...")
        model = GPT2LMHeadModel(config)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()

    # No Adaptive Gradient Checkpointing
    else:
        # Initialize Model
        overwatch.info(f"Initializing Tabula Rasa Model from Configuration: `{REGISTRY[model_id]}`...")
        model = AutoModelForCausalLM.from_config(config)

    # Run GPT-Specific Initialization, if applicable
    model.resize_token_embeddings(len(tokenizer))

    # If `initial_weights` is not None, load weights from path!
    if initial_weights is not None:
        overwatch.info(f"Initializing Weights from File: `{initial_weights}`...")
        model.load_state_dict(torch.load(initial_weights, map_location=torch.device("cpu")))
    
    overwatch.info("Model config: ")
    overwatch.info(model.config)
    return model, tokenizer
