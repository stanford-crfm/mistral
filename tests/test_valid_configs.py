import tests
import pathlib

from src.train_schema import MistralHparams

expected_manual_args = [
    "--artifacts.cache_dir", "artifacts/cache",
    "--artifacts.run_dir", "artifacts/runs",
]


def test_all_test_configs_are_valid():
    # test all the yaml files in the main mistral conf directory, and in the mistral tests/conf directory
    test_root = pathlib.Path(tests.__file__).parent.absolute()

    for path in pathlib.Path(test_root).glob('conf/*.yaml'):
        hparams: MistralHparams = MistralHparams.create(f=path, cli_args=expected_manual_args)
        # obnoxiously, yahp doesn't validate by default
        hparams.validate()

def test_all_real_configs_are_valid():
    # test all the yaml files in the main mistral conf directory, and in the mistral tests/conf directory
    mistral_root = pathlib.Path(tests.__file__).parent.parent.absolute()

    for path in pathlib.Path(mistral_root).glob('conf/*.yaml'):
        try:
            hparams: MistralHparams = MistralHparams.create(f=path, cli_args=expected_manual_args)
            # obnoxiously, yahp doesn't validate by default
            hparams.validate()
        except Exception as e:
            raise Exception(f"{path} is not valid: {e}") from e



if __name__ == "__main__":
    tests.run_tests()