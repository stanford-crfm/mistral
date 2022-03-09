"""
data_args.py

Utility script for unloading Quinfigs into data args.
"""
import logging
from typing import Dict

from munch import Munch


# Nest Overwatch under root `mistral` logger, inheriting formatting!
overwatch = logging.getLogger("mistral.args.data")


def get_data_arguments(quinfig_args: Munch) -> Dict:
    """Initialize Data Arguments from Quinfig."""
    data_args = {}
    if quinfig_args.datasets:
        data_args["dataset_list"] = [dict(ds) for ds in quinfig_args.datasets]
    else:
        data_args["dataset_list"] = [{"path": quinfig_args.id, "name": quinfig_args.name}]
    data_args["dataset_ratios"] = quinfig_args.dataset_ratios
    return data_args
