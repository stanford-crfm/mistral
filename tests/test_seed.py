from itertools import islice
from typing import List, Dict

import torch

from tests import MISTRAL_TEST_DIR, run_tests, run_train_process


# paths
CACHE_DIR = f"{MISTRAL_TEST_DIR}/artifacts"
RUNS_DIR = f"{MISTRAL_TEST_DIR}/runs"
RUN_ID = "train_args_test"
RUN_ID_DIR = f"{RUNS_DIR}/{RUN_ID}"

# set up different trainers to see initialization differences
TRAIN_ARGS_SEED_7 = {
    "nnodes": "1",
    "nproc_per_node": "1",
    "file": "conf/train.yaml",
    "training_arguments.fp16": "false",
    "training_arguments.per_device_train_batch_size": "1",
    "artifacts.cache_dir": CACHE_DIR,
    "seed": "7",
    "log_level": "50",
    "run_training": "false",
    "run_final_eval": "false",
}

TRAIN_ARGS_SEED_10 = dict(TRAIN_ARGS_SEED_7)
TRAIN_ARGS_SEED_10["seed"] = "10"


def is_randomized(key):
    """
    Helper to determine if the key in the state_dict() is a set of parameters that is randomly initialized.
    Some weights are not randomly initalized and won't be afffected by seed, particularly layer norm
    weights and biases, and bias terms in general.
    """
    # regexes for components that are not randomized
    if key.endswith("bias") or "ln" in key:
        return False
    else:
        return True


def test_weight_initializations() -> None:
    trainer_seed_7 = run_train_process(cl_args_dict=TRAIN_ARGS_SEED_7, runs_dir=RUNS_DIR, run_id="trainer_seed_7")
    trainer_seed_10 = run_train_process(cl_args_dict=TRAIN_ARGS_SEED_10, runs_dir=RUNS_DIR, run_id="trainer_seed_10")

    assert trainer_seed_7.model.state_dict().keys() == trainer_seed_10.model.state_dict().keys()
    for key in trainer_seed_7.model.state_dict().keys():
        if is_randomized(key):
            assert not torch.equal(
                trainer_seed_7.model.state_dict()[key], trainer_seed_10.model.state_dict()[key]
            ), f"weights are the same for {key}"


def test_data_order() -> None:
    trainer_seed_7 = run_train_process(cl_args_dict=TRAIN_ARGS_SEED_7, runs_dir=RUNS_DIR, run_id="trainer_seed_7")
    trainer_seed_10 = run_train_process(cl_args_dict=TRAIN_ARGS_SEED_10, runs_dir=RUNS_DIR, run_id="trainer_seed_10")

    seed_7_dataloader = trainer_seed_7.get_train_dataloader()
    seed_10_dataloader = trainer_seed_10.get_train_dataloader()

    def check_equal(data1: List[Dict[str, torch.Tensor]], data2: List[Dict[str, torch.Tensor]]) -> bool:
        # enough to check that the input_ids are the same
        # remember to use torch.equal() for tensors
        if len(data1) != len(data2):
            return False
        for i in range(len(data1)):
            if not torch.equal(data1[i]["input_ids"], data2[i]["input_ids"]):
                return False
        return True

    seed_7_data, seed_10_data = list(islice(iter(seed_7_dataloader), 20)),  list(islice(iter(seed_10_dataloader), 20))

    trainer_seed_7_copy = run_train_process(cl_args_dict=TRAIN_ARGS_SEED_7, runs_dir=RUNS_DIR,
                                            run_id="trainer_seed_7_copy")
    seed_7_copy_dataloader = trainer_seed_7_copy.get_train_dataloader()
    seed_7_copy_data = list(islice(iter(seed_7_copy_dataloader), 20))

    assert check_equal(seed_7_copy_data, seed_7_data), "data is not the same"
    assert not check_equal(seed_10_data, seed_7_data), "data order should be different for different seeds"


if __name__ == "__main__":
    run_tests()
