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
RUN_ID = "train_test"
RUN_ID_DIR = f"{RUNS_DIR}/{RUN_ID}"
LAST_CHECKPOINT = "checkpoint-2"

# run training process for tests
CL_ARGS = [
    "test_train.py",
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
    "2",
    "--artifacts.cache_dir",
    CACHE_DIR,
    "--log_level",
    "50",
]
trainer_after_training = run_train_process(cl_args=CL_ARGS, runs_dir=RUNS_DIR, run_id=RUN_ID)


def test_checkpoint_weights() -> None:
    """
    Test weights of a checkpointed model match the true weights.
    """
    model = trainer_after_training.model
    loaded_model = MistralGPT2LMHeadModel.from_pretrained(f"{RUN_ID_DIR}/{LAST_CHECKPOINT}")
    loaded_model.to(torch.device("cuda"))
    assert model.state_dict().keys() == loaded_model.state_dict().keys()
    for key in model.state_dict().keys():
        assert torch.equal(model.state_dict()[key], loaded_model.state_dict()[key])
    loaded_model.to(torch.device("cpu"))


def test_checkpoint_forward_pass() -> None:
    """
    Test that loaded model correctly calculate forward pass
    """
    model = trainer_after_training.model
    loaded_model = MistralGPT2LMHeadModel.from_pretrained(f"{RUN_ID_DIR}/{LAST_CHECKPOINT}")
    loaded_model.to(torch.device("cuda"))
    train_dataloader = trainer_after_training.get_train_dataloader()
    inputs = next(iter(train_dataloader))
    inputs = trainer_after_training._prepare_inputs(inputs)
    # run forward with loaded model
    loaded_model.eval()
    outputs_loaded = loaded_model(**inputs)
    #loaded_model.to(torch.device("cpu"))
    # run forward with original model
    #torch.cuda.empty_cache()
    model.eval()
    outputs = model(**inputs)
    assert torch.equal(outputs["logits"], outputs_loaded["logits"])
    assert torch.equal(outputs["loss"], outputs_loaded["loss"])


def test_checkpoint_frequency() -> None:
    """
    Test checkpointing happening at expected frequency
    """
    assert not os.path.exists(f"{RUN_ID_DIR}/checkpoint-1")
    assert os.path.exists(f"{RUN_ID_DIR}/checkpoint-2")


if __name__ == "__main__":
    test_checkpoint_weights()
    test_checkpoint_forward_pass()
    test_checkpoint_frequency()
