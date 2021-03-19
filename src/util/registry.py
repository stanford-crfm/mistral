"""
registry.py

Model/Data Registry :: Human-Readable Identifier --> Huggingface.co ID. Ideally will be expanded upon as we introduce
more model configurations, different types of architectures, etc.
"""
from src.corpora.auto import get_auto_dataset, get_lambada
from src.corpora.detokenization import wikitext_detokenize


REGISTRY = {"gpt2-small": "gpt2", "gpt2-medium": "gpt2-medium", "gpt2-large": "gpt2-large", "gpt2-xl": "gpt2-xl"}

# Mapping of eval dataset name -> HF Dataset IDs
ONLINE_EVAL_DATA_REGISTRY = {
    "wikitext": {"id": "wikitext", "name": "wikitext-103-raw-v1", "generator": get_auto_dataset},
    "lambada": {"id": "lambada", "name": None, "generator": get_lambada},
}

DATASET_TOKENIZATION_STRATEGY = {"wikitext": wikitext_detokenize}
