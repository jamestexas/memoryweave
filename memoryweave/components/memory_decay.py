"""
Memory decay component for MemoryWeave.

This component implements memory activation decay over time,
gradually reducing the activation levels of memories that haven't
been recently accessed.
"""

from typing import Any, Dict, Optional

import numpy as np

from memoryweave.components.base import Component


class MemoryDecayComponent(Component):
    """
    Implements memory decay by periodically reducing activation levels.

    This component applies an exponential decay to memory activation levels
    at specified intervals, simulating the natural fading of memories
    over time.
    """

    def __init__(self):
        """Initialize the memory decay component."""
        self.memory_decay_enabled = True
        self.memory_decay_rate = 0.99  # Default: Retain 99% of activation per decay
        self.memory_decay_interval = 10  # Default: Apply decay every 10 interactions
        self.interaction_count = 0
        self.art_clustering_enabled = False
        self.memory = None

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.memory_decay_enabled = config.get("memory_decay_enabled", True)
        self.memory_decay_rate = config.get("memory_decay_rate", 0.99)
        self.memory_decay_interval = config.get("memory_decay_interval", 10)
        self.art_clustering_enabled = config.get("art_clustering_enabled", False)
        self.memory = config.get("memory", None)

    def process(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data by applying memory decay as needed.

        Args:
            data: Input data dictionary
            context: Processing context

        Returns:
            Updated data with decay applied
        """
        if not self.memory_decay_enabled or not self.memory:
            return data

        # Get memory from context if not already set
        if not self.memory and "memory" in context:
            self.memory = context["memory"]

        # Apply decay
        self._apply_memory_decay()

        # Add decay info to context
        data["memory_decay_applied"] = self.interaction_count % self.memory_decay_interval == 0
        data["memory_decay_rate"] = self.memory_decay_rate
        data["interaction_count"] = self.interaction_count

        return data

    def process_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a query by applying memory decay and returning updated context.

        Args:
            query: The query string
            context: The processing context

        Returns:
            Updated context with decay information
        """
        # Get memory from context if not already set
        if not self.memory and "memory" in context:
            self.memory = context["memory"]

        # Apply decay
        self._apply_memory_decay()

        # Add decay info to context
        context["memory_decay_applied"] = self.interaction_count % self.memory_decay_interval == 0
        context["memory_decay_rate"] = self.memory_decay_rate
        context["interaction_count"] = self.interaction_count

        return context

    def _apply_memory_decay(self) -> None:
        """Apply decay to memory activations based on configured parameters."""
        # Increment interaction counter
        self.interaction_count += 1

        # Check if we should apply decay this interaction
        if self.interaction_count % self.memory_decay_interval != 0:
            return

        # Apply decay to memory activations
        try:
            # For component architecture memory
            if hasattr(self.memory, "get_activation_manager"):
                activation_manager = self.memory.get_activation_manager()
                if activation_manager:
                    activation_manager.decay_activations(self.memory_decay_rate)

            # For legacy memory
            elif hasattr(self.memory, "activation_levels") and isinstance(
                self.memory.activation_levels, np.ndarray
            ):
                # Apply decay - multiply by (1 - decay_rate) as per the legacy implementation
                # For example, with decay_rate = 0.5, multiply by 0.5 to retain half the value
                self.memory.activation_levels = self.memory.activation_levels * (
                    1.0 - self.memory_decay_rate
                )

            # If ART clustering is enabled and category activations exist
            if (
                self.art_clustering_enabled
                and hasattr(self.memory, "category_activations")
                and isinstance(self.memory.category_activations, np.ndarray)
            ):
                # Apply the same decay to category activations
                self.memory.category_activations = self.memory.category_activations * (
                    1.0 - self.memory_decay_rate
                )

        except Exception as e:
            # Log error but don't crash
            print(f"Error applying memory decay: {str(e)}")
