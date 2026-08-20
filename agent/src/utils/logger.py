"""
src/utils/logger.py
--------------------
Centralised logger for the entire project.
Import `logger` anywhere instead of using bare print() statements.

Usage:
    from src.utils.logger import logger
    logger.info("Graph compiled successfully")
    logger.error("LLM call failed: %s", str(e))
"""

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = "matz_agent") -> logging.Logger:
    """Return a configured logger. Call once per module."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    return logger


# Default logger — import this directly in most cases
logger = get_logger()
