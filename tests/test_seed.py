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
    expected_indices = [
        7485,
        6448,
        8289,
        7940,
        6492,
        4866,
        1722,
        1303,
        3568,
        7713,
        4597,
        3294,
        7178,
        2517,
        8770,
        8208,
        90,
        4594,
        4487,
        5002,
        2784,
        4846,
        6457,
        4210,
        1510,
        2230,
        8074,
        1846,
        753,
        3613,
        3354,
        8174,
        6577,
        6422,
        2463,
        670,
        8784,
        8659,
        2515,
        647,
        6654,
        5255,
        8623,
        7172,
        679,
        4060,
        4177,
        2159,
        7638,
        3163,
        468,
        2689,
        5817,
        8100,
        5736,
        8081,
        3993,
        7968,
        3549,
        7995,
        596,
        370,
        6044,
        1640,
        1693,
        7685,
        3544,
        5806,
        1887,
        692,
        5526,
        4601,
        3042,
        8700,
        222,
        1601,
        4908,
        5576,
        4823,
        7853,
        6892,
        5932,
        7890,
        2599,
        6431,
        2136,
        8601,
        964,
        2214,
        3320,
        1593,
        5543,
        5599,
        1694,
        3991,
        3595,
        4128,
        5573,
        4720,
        4600,
    ]
    actual_indices = [seed_10_indices.index(seed_7_indices[i]) for i in range(0, 100)]
    assert expected_indices == actual_indices, actual_indices


if __name__ == "__main__":
    run_tests()
