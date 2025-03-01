"""Memory component factory for MemoryWeave.

This module provides factories for creating memory-related components,
such as memory stores, vector stores, and activation managers.
"""

from typing import Any, Dict, Optional

from memoryweave.config.options import get_default_config
from memoryweave.config.validation import ConfigValidationError, validate_config
from memoryweave.interfaces.memory import (
    IActivationManager,
    ICategoryManager,
    IMemoryStore,
    IVectorStore,
)
from memoryweave.storage.activation import ActivationManager, TemporalActivationManager
from memoryweave.storage.category import CategoryManager
from memoryweave.storage.memory_store import MemoryStore
from memoryweave.storage.vector_store import ActivationVectorStore, SimpleVectorStore


class MemoryFactory:
    """Factory for creating memory-related components."""

    @staticmethod
    def create_memory_store(config: Optional[Dict[str, Any]] = None) -> IMemoryStore:
        """Create a memory store component.

        Args:
            config: Optional configuration for the memory store

        Returns:
            Configured memory store

        Raises:
            ConfigValidationError: If the configuration is invalid
        """
        # Use default config if none provided
        if config is None:
            config = {}

        # Merge with defaults
        defaults = get_default_config("memory_store")
        merged_config = {**defaults, **config}

        # Validate config
        is_valid, errors = validate_config(merged_config, "memory_store")
        if not is_valid:
            raise ConfigValidationError(errors, "memory_store")

        # Create memory store
        return MemoryStore()

    @staticmethod
    def create_vector_store(config: Optional[Dict[str, Any]] = None) -> IVectorStore:
        """Create a vector store component.

        Args:
            config: Optional configuration for the vector store

        Returns:
            Configured vector store

        Raises:
            ConfigValidationError: If the configuration is invalid
        """
        # Use default config if none provided
        if config is None:
            config = {}

        # Merge with defaults
        defaults = get_default_config("vector_store")
        merged_config = {**defaults, **config}

        # Validate config
        is_valid, errors = validate_config(merged_config, "vector_store")
        if not is_valid:
            raise ConfigValidationError(errors, "vector_store")

        # Check if we should use activation-weighted store
        if merged_config.get("activation_weight", 0.0) > 0:
            return ActivationVectorStore(activation_weight=merged_config["activation_weight"])
        else:
            return SimpleVectorStore()

    @staticmethod
    def create_activation_manager(config: Optional[Dict[str, Any]] = None) -> IActivationManager:
        """Create an activation manager component.

        Args:
            config: Optional configuration for the activation manager

        Returns:
            Configured activation manager
        """
        # Use default config if none provided
        if config is None:
            config = {}

        # Check if we should use temporal decay
        if config.get("use_temporal_decay", False):
            return TemporalActivationManager(
                initial_activation=config.get("initial_activation", 0.0),
                max_activation=config.get("max_activation", 10.0),
                min_activation=config.get("min_activation", -10.0),
                half_life_days=config.get("half_life_days", 7.0),
            )
        else:
            return ActivationManager(
                initial_activation=config.get("initial_activation", 0.0),
                max_activation=config.get("max_activation", 10.0),
                min_activation=config.get("min_activation", -10.0),
            )

    @staticmethod
    def create_category_manager(config: Optional[Dict[str, Any]] = None) -> ICategoryManager:
        """Create a category manager component.

        Args:
            config: Optional configuration for the category manager

        Returns:
            Configured category manager
        """
        # Use default config if none provided
        if config is None:
            config = {}

        # Create category manager
        vigilance = config.get("vigilance", 0.85)
        return CategoryManager(vigilance=vigilance)
