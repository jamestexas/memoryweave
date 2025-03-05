"""Configuration validation for MemoryWeave.

This module provides validation functionality for MemoryWeave configurations,
ensuring that configuration values meet the required constraints.
"""

from typing import Any, Optional

from memoryweave.config.options import ConfigOption, ConfigValueType, get_component_config


class ConfigValidationError(Exception):
    """Error raised when configuration validation fails."""

    def __init__(self, errors: dict[str, list[str]], component_type: Optional[str] = None):
        """Initialize the error.

        Args:
            errors: Dictionary mapping option names to lists of error messages
            component_type: Optional component type that was being validated
        """
        self.errors = errors
        self.component_type = component_type

        # Format error message
        message = "Configuration validation failed"
        if component_type:
            message += f" for component type '{component_type}'"

        if errors:
            message += ":\\n"
            for option_name, option_errors in errors.items():
                for error in option_errors:
                    message += f"  - {option_name}: {error}\\n"

        super().__init__(message)


def validate_config(
    config: dict[str, Any], component_type: str
) -> tuple[bool, dict[str, list[str]]]:
    """Validate a configuration against the component's configuration options.

    Args:
        config: Configuration to validate
        component_type: Type of component

    Returns:
        Tuple of (is_valid, errors) where errors is a dictionary mapping
        option names to lists of error messages
    """
    component_config = get_component_config(component_type)
    if not component_config:
        raise ValueError(f"Unknown component type: {component_type}")

    errors: dict[str, list[str]] = {}

    # Check required options
    for option in component_config.options:
        if option.required and option.name not in config:
            errors.setdefault(option.name, []).append("Required option is missing")

    # Validate each option in the config
    for name, value in config.items():
        # Find the corresponding option definition
        option = next((o for o in component_config.options if o.name == name), None)
        if option is None:
            errors.setdefault(name, []).append(
                f"Unknown option for component type '{component_type}'"
            )
            continue

        # Validate the option value
        option_errors = validate_option(value, option)
        if option_errors:
            errors[name] = option_errors

    # Return validation result
    return len(errors) == 0, errors


def validate_option(value: Any, option: ConfigOption) -> list[str]:
    """Validate a single configuration option.

    Args:
        value: Value to validate
        option: Option definition

    Returns:
        List of error messages, empty if valid
    """
    errors = []

    # Check value type
    if not is_correct_type(value, option.value_type):
        errors.append(f"Expected type {option.value_type.name}, got {type(value).__name__}")
        # If type is wrong, no need to check other constraints
        return errors

    # Check min/max constraints
    if option.min_value is not None and value < option.min_value:
        errors.append(f"Value must be at least {option.min_value}")

    if option.max_value is not None and value > option.max_value:
        errors.append(f"Value must be at most {option.max_value}")

    # Check allowed values
    if option.allowed_values is not None and value not in option.allowed_values:
        errors.append(f"Value must be one of: {', '.join(str(v) for v in option.allowed_values)}")

    # Check enum values
    if option.value_type == ConfigValueType.ENUM and option.enum_type is not None:
        if not isinstance(value, option.enum_type):
            errors.append(f"Value must be a {option.enum_type.__name__} enum")

    # Check nested options
    if option.value_type == ConfigValueType.DICT and option.nested_options is not None:
        for nested_option in option.nested_options:
            if nested_option.name in value:
                nested_errors = validate_option(value[nested_option.name], nested_option)
                if nested_errors:
                    for error in nested_errors:
                        errors.append(f"{nested_option.name}: {error}")

    return errors


def is_correct_type(value: Any, value_type: ConfigValueType) -> bool:
    """Check if a value is of the correct type.

    Args:
        value: Value to check
        value_type: Expected value type

    Returns:
        True if the value is of the correct type, False otherwise
    """
    if value_type == ConfigValueType.STRING:
        return isinstance(value, str)
    elif value_type == ConfigValueType.INTEGER:
        return isinstance(value, int) and not isinstance(value, bool)
    elif value_type == ConfigValueType.FLOAT:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    elif value_type == ConfigValueType.BOOLEAN:
        return isinstance(value, bool)
    elif value_type == ConfigValueType.LIST:
        return isinstance(value, list)
    elif value_type == ConfigValueType.DICT:
        return isinstance(value, dict)
    elif value_type == ConfigValueType.ENUM:
        # Enum type checked separately
        return True
    else:
        return False
