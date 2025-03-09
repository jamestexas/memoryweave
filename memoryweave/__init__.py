import logging
import logging.config
import os
from pathlib import Path

import yaml


def configure_logging(config_path="config/logging.yaml"):
    """Configures logging from a YAML file, handling HF_HUB_OFFLINE."""

    config_path = Path(__file__).parent / config_path

    if config_path.exists():
        config = yaml.safe_load(config_path.read_text())

        # Check for HF_HUB_OFFLINE environment variable (hugging face is noisy ok)
        if os.environ.get("HF_HUB_OFFLINE") == "1":
            # Disable logging for 'huggingface_hub.connection'
            if "loggers" not in config:
                config["loggers"] = {}
            config["loggers"]["huggingface_hub.connection"] = {
                "level": "CRITICAL",  # Or "ERROR" or "WARNING" depending on your needs.
                "propagate": False,
            }

        logging.config.dictConfig(config)
    else:
        print(f"Logging config file not found at: {config_path}")


configure_logging()  # configure the logging.

logger = logging.getLogger(__name__)  # create a base logger for the library.


def main() -> None:
    print("Hello from memoryweave!")
