"""
registry.py

Model/Data Registry :: Human-Readable Identifier --> Huggingface.co ID. Ideally will be expanded upon as we introduce
more model configurations, different types of architectures, etc.
"""

# Model Names
REGISTRY = {
    "gpt2-small": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "gpt2-xl": "gpt2-xl",
    "bert-base-cased": "bert-base-cased",
}

# Absolute Paths
PATH_REGISTRY = {
    "gpt2-small": "gpt2",
    "gpt2-medium": "gpt2",
    "gpt2-large": "gpt2",
    "gpt2-xl": "gpt2",
    "bert-base-cased": "bert-cased",
}
