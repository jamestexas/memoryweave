import logging
import logging.config
from pathlib import Path

import yaml


def configure_logging(config_path="config/logging.yaml"):
    """Configures logging from a YAML file."""

    config_path = (
        Path(__file__).parent / config_path
    )  # adjust the path to be relative to the current file.
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text())
        logging.config.dictConfig(config)
    else:
        print(f"Logging config file not found at: {config_path}")


configure_logging()  # configure the logging.

logger = logging.getLogger(__name__)  # create a base logger for the library.


def main() -> None:
    print("Hello from memoryweave!")
