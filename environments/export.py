"""
export.py

Utility script for taking an existing `conda` environment (Note: assumes that you are running this script from WITHIN
the given environment), dumping it to a `.yaml` file, stripping the "pip" requirements, and replacing it with the
output of `pip freeze > requirements.txt`.
"""
import argparse
import subprocess
from pathlib import Path

import yaml


MAP = {
    # We always want the latest version of Transformers -- TODO :: Maybe lock this to a specific version?
    "transformers": "git+https://github.com/huggingface/transformers",
}


def export() -> None:
    # Default & Simple Argparse --> Just takes one argument :: `arch` in < cpu | gpu >
    parser = argparse.ArgumentParser(description="Export Conda Environment for the Given Architecture.")
    parser.add_argument("-a", "--arch", default="gpu", type=str, help="Architecture in < cpu | gpu >.")
    args = parser.parse_args()
    assert args.arch in ["cpu", "gpu"], f"Architecture `{args.arch}` not defined - try one of < cpu | gpu >!"

    # Remove existing environment.yaml
    environment_yaml = Path("environments", f"environment-{args.arch}.yaml")
    Path.unlink(environment_yaml, missing_ok=True)

    # Run a call to dump the environment.yaml file, and a call to pip freeze to dump `requirements.txt`
    subprocess.call(f'conda env export --no-builds | grep -v "^prefix: " > {environment_yaml}', shell=True)

    # Read and Edit YAML File on the Fly...
    with open(environment_yaml, "r") as f:
        spec = yaml.load(f, Loader=yaml.FullLoader)

    # Iterate through spec["dependencies"] until `dict with "pip" as key!`
    for i in reversed(range(len(spec["dependencies"]))):
        if isinstance(spec["dependencies"][i], dict) and "pip" in spec["dependencies"][i]:
            pip_dependencies = spec["dependencies"][i]["pip"]

            # Edit in Place --> Replace Occurrences of MAP Libraries with corresponding links
            for j, pd in enumerate(pip_dependencies):
                key = pd.split("==")[0]
                if key in MAP:
                    pip_dependencies[j] = MAP[key]

            break

    # Dump YAML back to File
    with open(environment_yaml, "w") as f:
        yaml.dump(spec, f, sort_keys=False)


if __name__ == "__main__":
    export()
