"""
registry.py

Model/Data Registry :: Human-Readable Identifier --> Huggingface.co ID. Ideally will be expanded upon as we introduce
more model configurations, different types of architectures, etc.
"""

REGISTRY = {"gpt2-small": "gpt2", "gpt2-medium": "gpt2-medium", "gpt2-large": "gpt2-large", "gpt2-xl": "gpt2-xl"}

# Mapping of eval dataset name -> HF Dataset IDs
ONLINE_EVAL_DATA_REGISTRY = {
    "wikitext": {"id": "wikitext", "name": "wikitext-103-raw-v1"},
    "lambada": {"id": "lambada", "name": None},
}
