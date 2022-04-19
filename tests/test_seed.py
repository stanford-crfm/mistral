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
    "config": "conf/train.yaml",
    "training_arguments.fp16": "false",
    "training_arguments.per_device_train_batch_size": "1",
    "artifacts.cache_dir": CACHE_DIR,
    "seed": "7",
    "log_level": "50",
    "run_training": "false",
    "run_final_eval": "false",
}

trainer_seed_7 = run_train_process(cl_args_dict=TRAIN_ARGS_SEED_7, runs_dir=RUNS_DIR, run_id="trainer_seed_7")

TRAIN_ARGS_SEED_10 = dict(TRAIN_ARGS_SEED_7)
TRAIN_ARGS_SEED_10["seed"] = "10"
trainer_seed_10 = run_train_process(cl_args_dict=TRAIN_ARGS_SEED_10, runs_dir=RUNS_DIR, run_id="trainer_seed_10")


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
    assert trainer_seed_7.model.state_dict().keys() == trainer_seed_10.model.state_dict().keys()
    for key in trainer_seed_7.model.state_dict().keys():
        if is_randomized(key):
            assert not torch.equal(
                trainer_seed_7.model.state_dict()[key], trainer_seed_10.model.state_dict()[key]
            ), f"weights are the same for {key}"


def test_data_order() -> None:
    seed_7_dataloader = trainer_seed_7.get_train_dataloader()
    seed_10_dataloader = trainer_seed_10.get_train_dataloader()
    seed_7_indices, seed_10_indices = list(iter(seed_7_dataloader.sampler)), list(iter(seed_10_dataloader.sampler))

    trainer_seed_7_copy = run_train_process(cl_args_dict=TRAIN_ARGS_SEED_7, runs_dir=RUNS_DIR,
                                            run_id="trainer_seed_7_copy")
    seed_7_copy_dataloader = trainer_seed_7_copy.get_train_dataloader()
    seed_7_copy_indices = list(iter(seed_7_copy_dataloader.sampler))

    assert seed_7_copy_indices == seed_7_indices, "data order is different"
    assert seed_10_indices != seed_7_indices, "data order should be different for different seeds"


if __name__ == "__main__":
    run_tests()
