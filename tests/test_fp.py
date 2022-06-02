import pytest
import torch.cuda

from tests import MISTRAL_TEST_DIR, run_tests, run_train_process


# common paths and resources for tests

# paths
CACHE_DIR = f"{MISTRAL_TEST_DIR}/artifacts"
RUNS_DIR = f"{MISTRAL_TEST_DIR}/runs"
RUN_ID = "upcasting_test"
RUN_ID_DIR = f"{RUNS_DIR}/{RUN_ID}"

# run training processes for tests
TRAIN_ARGS = {
    "nnodes": "1",
    "nproc_per_node": "1",
    "config": "conf/train.yaml",
    "training_arguments.fp16": "true",
    "training_arguments.max_steps": "4",
    "artifacts.cache_dir": CACHE_DIR,
    "run_training": "true",
    "run_final_eval": "false",
    "log_level": "50",
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="need cuda for fp16")
def test_upcasting() -> None:
    """
    Run training with upcasting
    """
    run_train_process(cl_args_dict=TRAIN_ARGS, runs_dir=RUNS_DIR, run_id=RUN_ID)


if __name__ == "__main__":
    run_tests()
