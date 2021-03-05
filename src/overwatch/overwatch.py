"""
overwatch.py

Utility class for creating a centralized/standardized Python logger, with the Mercury format, at the appropriate
logging level.
"""
import logging
import sys


# Constants - for Formatting
FORMATTER = logging.Formatter("[*] %(asctime)s - %(name)s - %(levelname)s :: %(message)s", datefmt="%m/%d [%H:%M:%S]")


def get_overwatch(path: str, level: int, rank: int = 0, name: str = "mistral") -> logging.Logger:
    """
    Initialize logging.Logger with the appropriate name, console, and file handlers.

    TODO 1 - Initialize from YAML? -- see: https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/
    TODO 2 - Wrap all external code with a context manager? -- see: https://johnpaton.net/posts/redirect-logging/

    :param path: Path for writing log file --> should be identical to run_name (inherited from `train.py`)
    :param level: Default logging level --> should usually be INFO (inherited from `train.py`).
    :param rank: Process Rank (default = -1). Only log to `level` on rank <= 0, otherwise default level is WARN.
    :param name: Name of the top-level logger --> should usually be `mistral`.

    :return: Default "mistral" logger object :: logging.Logger
    """
    # Create Default Logger & add Handlers
    logger = logging.getLogger(name)
    logger.handlers = []
    logger.setLevel(level if rank <= 0 else logging.WARNING)

    # Create Console Handler --> Write to stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    logger.addHandler(console_handler)

    # Only Log to File w/ Rank 0 on each Node
    if rank <= 0:
        # Create File Handler --> Set mode to "w" to overwrite logs (ok, since each run will be uniquely named)
        file_handler = logging.FileHandler(path, mode="w")
        file_handler.setFormatter(FORMATTER)
        logger.addHandler(file_handler)

    # Do not propagate by default...
    logger.propagate = False
    return logger
