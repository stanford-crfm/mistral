"""
data_args.py

Utility script for unloading Quinfigs into data args.
"""
import logging
from typing import Any, Dict

from munch import Munch


# Nest Overwatch under root `mistral` logger, inheriting formatting!
overwatch = logging.getLogger("mistral.args.data")


def get_data_arguments(quinfig_args: Munch) -> Dict[Any, Any]:
    """Initialize Data Arguments from Quinfig."""
    data_args: Dict[Any, Any] = {}
    # set up dataset loading arguments
    if quinfig_args.datasets:
        data_args["dataset_list"] = [dict(ds) for ds in quinfig_args.datasets]
        # set up data_files
        for ds in data_args["dataset_list"]:
            data_files = {"train": ds.pop("train", None), "validation": ds.pop("validation", None)}
            if data_files["train"] or data_files["validation"]:
                data_args["data_files"] = data_files
    else:
        data_args["dataset_list"] = [{"path": quinfig_args.id, "name": quinfig_args.name}]
    # get ratios of datasets
    data_args["dataset_ratios"] = quinfig_args.dataset_ratios
    return data_args
