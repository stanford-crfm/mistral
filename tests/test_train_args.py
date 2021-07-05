import os
import pytest
import shutil
import sys
import torch

from src.models.mistral_gpt2 import MistralGPT2LMHeadModel
from tests import MISTRAL_TEST_DIR, run_train_process
from train import train


# paths
CACHE_DIR = f"{MISTRAL_TEST_DIR}/artifacts"
RUNS_DIR = f"{MISTRAL_TEST_DIR}/runs"
RUN_ID = "train_args_test"
RUN_ID_DIR = f"{RUNS_DIR}/{RUN_ID}"
LAST_CHECKPOINT = "checkpoint-2"

# run training process for tests
TRAIN_ARGS = [
    "test_train_args.py",
    "--nnodes",
    "1",
    "--nproc_per_node",
    "1",
    "--config",
    "conf/train.yaml",
    "--training_arguments.fp16",
    "true",
    "--training_arguments.per_device_train_batch_size",
    "1",
    "--training_arguments.max_steps",
    "1",
    "--artifacts.cache_dir",
    CACHE_DIR,
    "--log_level",
    "50",
]
trainer_w_train = run_train_process(cl_args=TRAIN_ARGS, runs_dir=RUNS_DIR, run_id=RUN_ID)

TRAIN_ARGS_DIFF = [
    "test_train_args.py",
    "--nnodes",
    "1",
    "--nproc_per_node",
    "1",
    "--config",
    "conf/train-diff.yaml",
    "--training_arguments.fp16",
    "true",
    "--training_arguments.per_device_train_batch_size",
    "1",
    "--training_arguments.max_steps",
    "1",
    "--artifacts.cache_dir",
    CACHE_DIR,
    "--log_level",
    "50",
]
RUN_ID = "train_args_diff_test"
trainer_w_train_diff = run_train_process(cl_args=TRAIN_ARGS_DIFF, runs_dir=RUNS_DIR, run_id=RUN_ID)


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
    test_train_args()
