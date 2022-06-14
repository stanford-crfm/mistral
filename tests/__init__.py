import inspect
import os
import re
import shutil
import subprocess
import sys
import traceback
from unittest.mock import patch

import psutil

from src.core.trainer import OnlineBenchmarkTrainer
from train import train


MISTRAL_TEST_DIR = os.getenv("MISTRAL_TEST_DIR")

# standard utils


def to_cl_args(args_dict):
    """
    Create a list of cl args from a dictionary
    """
    args_list = []
    for k, v in args_dict.items():
        args_list.append(f"--{k}")
        args_list.append(v)
    return args_list


# deepspeed utils


def launched_by_deepspeed():
    """
    Determine if this process has been launched by deepspeed.
    """
    parent = psutil.Process(os.getppid())
    return "deepspeed.launcher.launch" in parent.cmdline()


DEEPSPEED_MODE = launched_by_deepspeed()


def am_first_deepspeed_child():
    """
    Check if this is the first deepspeed child.
    """
    if DEEPSPEED_MODE:
        parent = psutil.Process(os.getppid())
        children = parent.children()
        return os.getpid() == children[0].pid if children else False
    else:
        return False


def deepspeed_launch_info():
    """
    Get info about number of nodes/gpus used by deepspeed.
    """
    grandparent = psutil.Process(os.getppid()).parent()
    num_nodes = grandparent.cmdline()[grandparent.cmdline().index("--num_nodes") + 1]
    num_gpus = grandparent.cmdline()[grandparent.cmdline().index("--num_gpus") + 1]
    return {"nodes": int(num_nodes), "gpus": int(num_gpus)}


def deepspeedify(cl_args_dict):
    """
    Alter standard test args to have deepspeed info
    """
    info = deepspeed_launch_info()
    cl_args_dict["nproc_per_node"] = str(info["gpus"])
    cl_args_dict["nnodes"] = str(info["nodes"])
    # cl_args_dict["training_arguments.deepspeed"] = "conf/deepspeed/z2-small-conf.json"
    cl_args_dict["training_arguments.deepspeed"] = "conf/deepspeed/z2-small-conf.json"


def run_train_process(cl_args_dict, runs_dir, run_id, use_deepspeed=DEEPSPEED_MODE) -> OnlineBenchmarkTrainer:
    """
    Run training with given cl args and run dir.
    """
    # clear training dir
    cl_args_dict["artifacts.run_dir"] = runs_dir
    cl_args_dict["run_id"] = run_id
    if use_deepspeed:
        deepspeedify(cl_args_dict)
    cl_args = [""] + to_cl_args(cl_args_dict)
    run_id_dir = f"{runs_dir}/{run_id}"
    if not use_deepspeed or am_first_deepspeed_child():
        print(f"Removing {run_id_dir}...")
        shutil.rmtree(run_id_dir) if os.path.exists(run_id_dir) else None
    # log cl args used
    print(f"Using following command line args for training: {cl_args}")
    with patch.object(sys, "argv", cl_args):
        # run main training process
        trainer = train()
    return trainer


def get_test_functions():
    """
    Return all test functions in this module.
    """
    all_test_functions = [
        (name, obj)
        for name, obj in inspect.getmembers(sys.modules["__main__"])
        if (inspect.isfunction(obj) and name.startswith("test") and obj.__module__ == "__main__")
    ]
    return all_test_functions


def run_tests():
    """
    Run each function, catch and report AssertionError's
    """
    if DEEPSPEED_MODE and not am_first_deepspeed_child():
        return
    test_functions = get_test_functions()
    passing_tests = []
    failing_tests = []
    assertion_errors = []
    print("Running tests:")
    for (name, test_function) in test_functions:
        print("")
        print(name)
        try:
            test_function()
            passing_tests.append(name)
        except AssertionError as e:
            failing_tests.append(name)
            assertion_errors.append((e, traceback.format_exc()))
    print("")
    print("Test report:")
    print(f"{len(passing_tests)} passed, {len(failing_tests)} failed")
    print("")
    print("Failing tests:")
    for test, error in zip(failing_tests, assertion_errors):
        print("")
        print(f"{test}")
        print(error[1])
        print(error[0])
    if len(failing_tests) > 0:
        sys.exit(1)
