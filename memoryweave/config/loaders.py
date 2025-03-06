"""Configuration loading utilities for MemoryWeave.

This module provides utilities for loading configurations from various sources,
such as JSON files, YAML files, and environment variables.
"""

import json
import logging
import os
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Optional, Union

from memoryweave.config.options import get_default_config
from memoryweave.config.validation import ConfigValidationError, validate_config


class ConfigLoader:
    """Loader for component configurations."""

    def __init__(self):
        """Initialize the config loader."""
        self._logger = logging.getLogger(__name__)

    def load_from_file(
        self, file_path: Union[str, Path], component_type: Optional[str] = None
    ) -> dict[str, Any]:
        """Load configuration from a file.

        Args:
            file_path: Path to the configuration file
            component_type: Optional component type for validation

        Returns:
            Loaded configuration

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid
            ConfigValidationError: If the configuration is invalid
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        # Determine file format from extension
        if path.suffix.lower() in (".json", ".jsonc"):
            config = self._load_json(path)
        elif path.suffix.lower() in (".yaml", ".yml"):
            config = self._load_yaml(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        # Validate if component type provided
        if component_type:
            valid, errors = validate_config(config, component_type)
            if not valid:
                raise ConfigValidationError(errors, component_type)

        return config

    def load_with_defaults(self, config: dict[str, Any], component_type: str) -> dict[str, Any]:
        """Load configuration with default values for missing options.

        Args:
            config: User-provided configuration
            component_type: Component type for defaults

        Returns:
            Configuration with defaults applied
        """
        defaults = get_default_config(component_type)

        # Merge defaults with user config (user config takes precedence)
        merged = {**defaults, **config}

        return merged

    def load_from_env(self, prefix: str, component_type: Optional[str] = None) -> dict[str, Any]:
        """Load configuration from environment variables.

        Args:
            prefix: Prefix for environment variables (e.g., 'MEMORYWEAVE_')
            component_type: Optional component type for validation

        Returns:
            Loaded configuration

        Raises:
            ConfigValidationError: If the configuration is invalid
        """
        config = {}

        # Find all environment variables with the prefix
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix) :].lower()

                # Convert value to appropriate type
                config[config_key] = self._convert_env_value(value)

        # Validate if component type provided
        if component_type:
            valid, errors = validate_config(config, component_type)
            if not valid:
                raise ConfigValidationError(errors, component_type)

        return config

    def _load_json(self, file_path: Path) -> dict[str, Any]:
        """Load configuration from a JSON file."""
        with open(file_path) as f:
            return json.load(f)

    def _load_yaml(self, file_path: Path) -> dict[str, Any]:
        """Load configuration from a YAML file."""
        if find_spec("yaml") is None:
            self._logger.error(
                "PyYAML is required to load YAML files. Install with 'pip install pyyaml'"
            )
            raise ImportError("PyYAML is required to load YAML files")
        import yaml

        with open(file_path) as f:
            return yaml.safe_load(f)

    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Check for boolean values
        if value.lower() in ("true", "yes", "1"):
            return True
        elif value.lower() in ("false", "no", "0"):
            return False

        # Check for numeric values
        try:
            # Try as int first
            return int(value)
        except ValueError:
            try:
                # Then as float
                return float(value)
            except ValueError:
                # Otherwise, keep as string
                return value
