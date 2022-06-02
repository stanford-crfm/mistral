import os

import torch

from tests import MISTRAL_TEST_DIR, check_samples_equal, get_samples, run_tests, run_train_process


# common paths and resources for tests

# paths
CACHE_DIR = f"{MISTRAL_TEST_DIR}/artifacts"
RUNS_DIR = f"{MISTRAL_TEST_DIR}/runs"
RUN_ID = "checkpoint_test"
RUN_ID_DIR = f"{RUNS_DIR}/{RUN_ID}"
LAST_CHECKPOINT = "checkpoint-2"

# run training processes for tests
TRAIN_ARGS = {
    "nnodes": "1",
    "nproc_per_node": "1",
    "config": "conf/train.yaml",
    "training_arguments.fp16": "false",
    "training_arguments.max_steps": "3",
    "training_arguments.per_device_train_batch_size": "1",
    "artifacts.cache_dir": CACHE_DIR,
    "log_level": "20",
    "effective_bsz": "16",
    "run_final_eval": "false",
    "training_arguments.dataloader_num_workers": "0",
}


RESTART_ARGS = {
    "nnodes": "1",
    "nproc_per_node": "1",
    "config": "conf/train.yaml",
    "training_arguments.fp16": "false",
    "training_arguments.max_steps": "3",
    "training_arguments.per_device_train_batch_size": "1",
    "resume": "True",
    "resume_checkpoint": f"{RUN_ID_DIR}/{LAST_CHECKPOINT}",
    "artifacts.cache_dir": CACHE_DIR,
    "log_level": "20",
    "effective_bsz": "16",
    "run_final_eval": "false",
    "training_arguments.dataloader_num_workers": "0",
}


trainer_after_training = None
trainer_after_restart = None


def setup_module() -> None:
    global trainer_after_training, trainer_after_restart
    trainer_after_training = run_train_process(cl_args_dict=TRAIN_ARGS, runs_dir=RUNS_DIR, run_id=RUN_ID)
    trainer_after_restart = run_train_process(cl_args_dict=RESTART_ARGS, runs_dir=RUNS_DIR, run_id=RUN_ID + "-restart")


def test_checkpoint_weights() -> None:
    """
    Test weights of a checkpointed model match the true weights.
    """
    model = trainer_after_training.model
    loaded_model = trainer_after_restart.model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loaded_model.to(device)
    assert model.state_dict().keys() == loaded_model.state_dict().keys()
    for key in model.state_dict().keys():
        assert torch.equal(model.state_dict()[key], loaded_model.state_dict()[key])


def test_checkpoint_forward_pass() -> None:
    """
    Test that loaded model correctly calculate forward pass
    """
    model = trainer_after_training.model
    loaded_model = trainer_after_restart.model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loaded_model.to(device)
    train_dataloader = trainer_after_training.get_train_dataloader()
    inputs = next(iter(train_dataloader))
    inputs = trainer_after_training._prepare_inputs(inputs)
    assert model.state_dict().keys() == loaded_model.state_dict().keys()
    for key in model.state_dict().keys():
        assert torch.equal(model.state_dict()[key], loaded_model.state_dict()[key])
    # run forward with loaded model
    loaded_model.eval()
    outputs_loaded = loaded_model(**inputs)
    # run forward with original model
    model.eval()
    outputs = model(**inputs)
    assert torch.equal(outputs["logits"], outputs_loaded["logits"]), (
        f"original: {outputs['logits']} dtype: {outputs['logits'].dtype}, loaded: {outputs_loaded['logits']} dtype:"
        f" {outputs['logits'].dtype}"
    )


def test_checkpoint_frequency() -> None:
    """
    Test checkpointing happening at expected frequency
    """
    assert not os.path.exists(f"{RUN_ID_DIR}/checkpoint-1")
    assert os.path.exists(f"{RUN_ID_DIR}/checkpoint-2")
    assert not os.path.exists(f"{RUN_ID_DIR}/checkpoint-3")


def test_restart_batch_order() -> None:
    """
    Test batch order is consistent when restarting
    """
    original_data = get_samples(trainer_after_training.get_train_dataloader())
    after_restart_data = get_samples(trainer_after_restart.get_train_dataloader())
    assert check_samples_equal(original_data, after_restart_data)


if __name__ == "__main__":
    run_tests()
