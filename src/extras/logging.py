import sys
import logging


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
