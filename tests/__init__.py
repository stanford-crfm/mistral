import os
import shutil
import sys

from src.core.trainer import OnlineBenchmarkTrainer
from train import train
from unittest.mock import patch

MISTRAL_TEST_DIR = os.getenv("MISTRAL_TEST_DIR")


def run_train_process(cl_args, runs_dir, run_id) -> OnlineBenchmarkTrainer:
    """
    Run training with given cl args and run dir.
    """
    # clear training dir
    run_id_dir = f"{runs_dir}/{run_id}"
    cl_args += ["--artifacts.run_dir", f"{runs_dir}"]
    cl_args += ["--run_id", f"{run_id}"]
    shutil.rmtree(run_id_dir) if os.path.exists(run_id_dir) else None
    with patch.object(sys, "argv", cl_args):
        # run main training process
        trainer = train()
    return trainer
