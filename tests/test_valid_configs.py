import pathlib
import sys
from unittest.mock import patch

from quinine.common.argparse import QuinineArgumentParser

import tests
from conf.train_schema import get_schema


expected_manual_args = [
    "--artifacts.cache_dir",
    "artifacts/cache",
    "--artifacts.run_dir",
    "artifacts/runs",
]


def validate_config(config_file):
    try:
        cl_args = ["--config", str(config_file)] + expected_manual_args
        with patch.object(sys, "argv", ["foo.py"] + cl_args):
            QuinineArgumentParser(schema=get_schema()).parse_quinfig()
    except Exception as e:
        raise Exception(f"{config_file} is not valid: {e}") from e


def test_all_test_configs_are_valid():
    # test all the yaml files in the main mistral conf directory, and in the mistral tests/conf directory
    test_root = pathlib.Path(tests.__file__).parent.absolute()

    for path in pathlib.Path(test_root).glob("conf/*.yaml"):
        validate_config(path)


def test_all_real_configs_are_valid():
    # test all the yaml files in the main mistral conf directory, and in the mistral tests/conf directory
    mistral_root = pathlib.Path(tests.__file__).parent.parent.absolute()

    for path in pathlib.Path(mistral_root).glob("conf/*.yaml"):
        validate_config(path)


if __name__ == "__main__":
    tests.run_tests()
