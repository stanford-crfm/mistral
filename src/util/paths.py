"""
paths.py

Utility function for initializing the appropriate directories/sub-directories on the start of each run. Decoupled from
main code in case we want separate directory structures/artifact storage based on infrastructure (e.g., NLP Cluster vs.
GCP).
"""
import os
import re
import shutil
from pathlib import Path
from typing import Dict

from .registry import PATH_REGISTRY


PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")


def create_paths(run_id: str, model: str, run_dir: str, cache_dir: str) -> Dict[str, Path]:
    """
    Create the necessary directories and sub-directories conditioned on the `run_id`, checkpoint directory, and cache
    directories.

    :param run_id: Unique Run Identifier.
    :param model: Huggingface.Transformers Model ID for specifying the desired configuration.
    :param run_dir: Path to run directory to save model checkpoints and run metrics.
    :param cache_dir: Path to artifacts/cache directory to store any intermediate values, configurations, etc.

    :return: Dictionary mapping str ids --> paths on the filesystem.
    """
    paths = {
        # Top-Level Checkpoint Directory for Given Run
        "runs": Path(run_dir) / run_id,
        # Cache Directories for various components
        "configs": Path(cache_dir) / f"{PATH_REGISTRY[model]}-configs",
        "tokenizer": Path(cache_dir) / f"{PATH_REGISTRY[model]}-tokenizer",
        "dataset": Path(cache_dir) / "datasets",
        "preprocessed": Path(cache_dir) / f"{PATH_REGISTRY[model]}-processed",
    }

    # Programatically Create Paths for each Directory
    for p in paths:
        paths[p].mkdir(parents=True, exist_ok=True)

    return paths


def set_permissions(paths: Dict[str, Path]) -> None:
    """ Recursively call `os.chmod(775) recursively for the given paths. """
    for p in paths:
        os.system(f"chmod -R 775 {paths[p]} >/dev/null 2>&1")


def get_nearest_checkpoint(folder: Path, desired_checkpoint_step: str) -> str:
    """ Given the set of checkpoints in folder, find the maximum closest smaller checkpoint"""
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return None
    checkpoint_steps = list(map(lambda x: int(_re_checkpoint.search(x).groups()[0]), checkpoints))
    # Sort Smallest First
    checkpoint_steps, checkpoints = list(zip(*sorted(zip(checkpoint_steps, checkpoints), key=lambda x: x[0])))

    # Get largest checkpoint such that desired_checkpoint_step < checkpoint+1
    i = 0
    while i < len(checkpoint_steps):
        if desired_checkpoint_step < checkpoint_steps[i]:
            break
        i += 1
    i = max(0, i - 1)
    return folder / checkpoints[i]


def crash_latest_checkpoint(folder: Path, crash_checkpoint_step: str) -> bool:
    """ Checks if the crash checkpoint is after the lastest saved checkpoint in Path """
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    checkpoint_steps = list(map(lambda x: int(_re_checkpoint.search(x).groups()[0]), checkpoints))
    return max(checkpoint_steps) <= crash_checkpoint_step


def remove_runs_after(folder: Path, resume_checkpoint: str) -> None:
    """ Given a checkpoint. remove all later checkpoints in folder """
    content = os.listdir(folder)
    resume_checkpoint_step = int(_re_checkpoint.search(resume_checkpoint).groups()[0])
    to_remove_checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None
        and os.path.isdir(os.path.join(folder, path))
        and int(_re_checkpoint.search(path).groups()[0]) > resume_checkpoint_step
    ]
    for path in to_remove_checkpoints:
        shutil.rmtree(folder / path, ignore_errors=True)
    return
