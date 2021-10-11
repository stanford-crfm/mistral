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
    "training_arguments.fp16": "true",
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
    print(key)
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
    expected_indices = [
        7750,
        834,
        2173,
        1965,
        8353,
        6936,
        63,
        801,
        8120,
        5126,
        545,
        4399,
        1870,
        5742,
        1479,
        8300,
        2992,
        5862,
        4228,
        7721,
        427,
        2686,
        6852,
        6476,
        5063,
        1162,
        3629,
        6828,
        5911,
        6000,
        4749,
        7991,
        4956,
        3906,
        5102,
        5656,
        1819,
        2048,
        3421,
        3459,
        187,
        6226,
        3936,
        34,
        4773,
        5253,
        5129,
        870,
        8856,
        3452,
        8169,
        8412,
        480,
        6471,
        1781,
        5759,
        6117,
        7573,
        5958,
        2147,
        6067,
        5012,
        4554,
        6166,
        4864,
        2953,
        8328,
        2556,
        8428,
        6649,
        1337,
        7188,
        500,
        7584,
        3712,
        399,
        5224,
        2324,
        445,
        5117,
        559,
        7899,
        1175,
        8707,
        6184,
        1214,
        1954,
        4491,
        4756,
        3144,
        4684,
        4508,
        2956,
        7459,
        1431,
        3665,
        474,
        3485,
        7773,
        1874,
    ]
    actual_indices = [seed_10_indices.index(seed_7_indices[i]) for i in range(0, 100)]
    assert expected_indices == actual_indices, actual_indices


if __name__ == "__main__":
    run_tests()
