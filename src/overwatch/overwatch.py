"""
overwatch.py

Utility class for creating a centralized/standardized Python logger, with the Mercury format, at the appropriate
logging level.
"""
import logging
from pathlib import Path

import datasets
import transformers


# Constants - for Formatting
LOG_FORMAT = "|=>> %(asctime)s - %(name)s - %(levelname)s :: %(message)s"
DATE_FORMAT = "%m/%d [%H:%M:%S]"


def get_overwatch(path: Path, level: int, local_rank: int = 0) -> logging.Logger:
    """
    Initialize logging.Logger with the appropriate name, console, and file handlers.

    :param path: Path for writing log file --> should be identical to run_name (inherited from `train.py`)
    :param level: Default logging level --> should usually be INFO (inherited from `train.py`).
    :param local_rank: Process Rank (default = -1). Only log to `level` on rank <= 0, otherwise default level is WARN.

    :return: Default "mistral" root logger object :: logging.Logger
    """
    # Create Root Logger w/ Base Formatting
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=DATE_FORMAT)

    # Suppress Hugging Face Loggers --> propagate up to Root!
    transformers.logging._get_library_root_logger().handlers = []
    transformers.logging._get_library_root_logger().setLevel(level=level)
    datasets.logging._get_library_root_logger().handlers = []

    # Create Default Logger & add File Handler
    logger = logging.getLogger()
    logger.setLevel(level if local_rank <= 0 else logging.WARNING)

    # Only Log to File w/ Rank 0 on each Node
    if local_rank <= 0:
        # Create File Handler --> Set mode to "a" to append to logs (ok, since each run will be uniquely named)
        file_handler = logging.FileHandler(path, mode="a")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(file_handler)

    return logging.getLogger("mistral")
