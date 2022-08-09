from copy import copy

import numpy

from src.core import OnlineBenchmarkTrainer
from tests import MISTRAL_TEST_DIR, run_tests, run_train_process


# paths
CACHE_DIR = f"{MISTRAL_TEST_DIR}/artifacts"
RUNS_DIR = f"{MISTRAL_TEST_DIR}/runs"

TRAIN_ARGS = {
    "config": "conf/train.yaml",
    "training_arguments.fp16": "false",
    "training_arguments.per_device_train_batch_size": "1",
    "artifacts.cache_dir": CACHE_DIR,
    "log_level": "50",
    "run_training": "false",
    "run_final_eval": "false",
}

trainer: OnlineBenchmarkTrainer = None


def setup_module() -> None:
    global trainer
    trainer = run_train_process(cl_args_dict=TRAIN_ARGS, runs_dir=RUNS_DIR, run_id="train_args_test")


def test_train_args() -> None:
    metrics = trainer.evaluate()
    assert numpy.isfinite(metrics["eval_loss"])


if __name__ == "__main__":
    run_tests()
