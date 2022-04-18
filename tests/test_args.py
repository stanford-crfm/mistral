from copy import copy

from tests import MISTRAL_TEST_DIR, run_tests, run_train_process


# paths
CACHE_DIR = f"{MISTRAL_TEST_DIR}/artifacts"
RUNS_DIR = f"{MISTRAL_TEST_DIR}/runs"

TRAIN_ARGS = {
    "nnodes": "1",
    "nproc_per_node": "1",
    "config": "conf/train.yaml",
    "training_arguments.fp16": "false",
    "training_arguments.per_device_train_batch_size": "1",
    "artifacts.cache_dir": CACHE_DIR,
    "log_level": "50",
    "run_training": "false",
    "run_final_eval": "false",
}

trainer_w_train = run_train_process(cl_args_dict=TRAIN_ARGS, runs_dir=RUNS_DIR, run_id="train_args_test")

TRAIN_ARGS_DIFF = copy(TRAIN_ARGS)
TRAIN_ARGS_DIFF["config"] = "conf/train-diff.yaml"

trainer_w_train_diff = run_train_process(
    cl_args_dict=TRAIN_ARGS_DIFF, runs_dir=RUNS_DIR, run_id="train_args_diff_test"
)


def test_train_args() -> None:
    assert trainer_w_train.args.weight_decay == 0.1
    assert trainer_w_train.args.adam_beta1 == 0.9
    assert trainer_w_train.args.adam_beta2 == 0.95
    assert trainer_w_train.args.max_grad_norm == 1.0
    assert trainer_w_train_diff.args.weight_decay == 0.2
    assert trainer_w_train_diff.args.adam_beta1 == 0.7
    assert trainer_w_train_diff.args.adam_beta2 == 0.3
    assert trainer_w_train_diff.args.max_grad_norm == 2.0


if __name__ == "__main__":
    run_tests()
