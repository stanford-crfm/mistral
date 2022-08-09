import re

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
    "run_training": "true",
    "training_arguments.max_steps": "2",  # just enough steps so HF doesn't complain about using zero 2 for inference
    "run_final_eval": "false",
}

trainer: OnlineBenchmarkTrainer = None
metrics: dict = None


def setup_module() -> None:
    global trainer, metrics
    trainer, metrics = run_train_process(
        cl_args_dict=TRAIN_ARGS, runs_dir=RUNS_DIR, run_id="train_eval_loss_is_defined", also_evaluate=True
    )


def test_train_args() -> None:
    assert any(numpy.isfinite(v) and re.match("eval.*loss", k) for k, v in metrics.items())


if __name__ == "__main__":
    run_tests()
