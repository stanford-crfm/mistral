"""
paths.py

Utility function for initializing the appropriate directories/sub-directories on the start of each run. Decoupled from
main code in case we want separate directory structures/artifact storage based on infrastructure (e.g., NLP Cluster vs.
GCP).
"""
import os
from pathlib import Path
from typing import Dict

from .registry import REGISTRY


def create_paths(run_id: str, model: str, run_dir: str, cache_dir: str, energy_dir: str) -> Dict[str, Path]:
    """
    Create the necessary directories and sub-directories conditioned on the `run_id`, checkpoint directory, and cache
    directories.

    :param run_id: Unique Run Identifier.
    :param model: Huggingface.Transformers Model ID for specifying the desired configuration.
    :param run_dir: Path to run directory to save model checkpoints and run metrics.
    :param cache_dir: Path to artifacts/cache directory to store any intermediate values, configurations, etc.
    :param energy_dir: Path to energy logging directory for writing energy usage for climate-responsible AI.

    :return: Dictionary mapping str ids --> paths on the filesystem.
    """
    paths = {
        # Top-Level Checkpoint Directory for Given Run
        "runs": Path(run_dir) / run_id,
        # Logging Directory (HF defaults to Tensorboard -- TODO 19 :: Remove Tensorboard and just use W&B and Custom?
        "logs": Path(run_dir) / run_id / "logs",
        # WandB Save Directory
        "wandb": Path(run_dir) / run_id / "wandb",
        # Energy Directory to save Carbon Metrics
        "energy": Path(energy_dir) / run_id / "energy",
        # Cache Directories for various components
        "configs": Path(cache_dir) / f"{REGISTRY[model]}-configs",
        "tokenizer": Path(cache_dir) / f"{REGISTRY[model]}-tokenizer",
        "dataset": Path(cache_dir) / "datasets",
        "preprocessed": Path(cache_dir) / f"{REGISTRY[model]}-processed",
    }

    # Programatically Create Paths for each Directory
    for p in paths:
        paths[p].mkdir(parents=True, exist_ok=True)

    return paths


def set_permissions(paths: Dict[str, Path]) -> None:
    """ Recursively call `os.chmod(775) recursively for the given paths. """
    for p in paths:
        os.system(f"chmod -R 775 {paths[p]}")
