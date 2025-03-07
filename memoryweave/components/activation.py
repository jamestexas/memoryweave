# memoryweave/components/activation.py
"""
Activation mechanisms for MemoryWeave.

This module implements biologically-inspired activation patterns for memory retrieval,
enabling spreading activation through the associative memory fabric.
"""

import time
from collections import defaultdict
from typing import Any, Optional

import numpy as np

from memoryweave.components.associative_linking import AssociativeMemoryLinker
from memoryweave.components.base import Component
from memoryweave.components.component_names import ComponentName
from memoryweave.interfaces.memory import MemoryID
from memoryweave.storage.base_store import StandardMemoryStore


class ActivationManager(Component):
    """
    Component that manages activation levels for memories in the system.

    This component implements:
    1. Memory activation tracking
    2. Spreading activation through associative links
    3. Decay functions for activation over time
    4. Visualization of activation patterns

    The activation mechanisms simulate how real memory systems access and
    retrieve information through associative patterns.
    """

    def __init__(
        self,
        memory_store: Optional[StandardMemoryStore] = None,
        associative_linker: Optional[AssociativeMemoryLinker] = None,
    ):
        """
        Initialize the activation manager.

        Args:
            memory_store: Optional memory store to manage activations for
            associative_linker: Optional associative memory linker for spreading activation
        """
        self.memory_store = memory_store
        self.associative_linker = associative_linker
        self.component_id = ComponentName.ACTIVATION_MANAGER

        # Activation parameters
        self.base_activation = 0.1
        self.activation_threshold = 0.2
        self.spreading_factor = 0.5
        self.max_spreading_hops = 2
        self.decay_rate = 0.9  # Per update
        self.long_term_decay_rate = 0.99  # Slower decay for long-term memory
        self.long_term_threshold = 7 * 86400  # 7 days

        # Activation state
        self.activation_levels: dict[MemoryID, float] = defaultdict(float)
        self.activation_history: dict[MemoryID, list[tuple[float, float]]] = defaultdict(list)
        self.last_update_time = time.time()
        self.activation_timestamps: dict[MemoryID, float] = {}

    def initialize(self, config: dict[str, Any]) -> None:
        """
        Initialize the component with configuration.

        Args:
            config: Configuration dictionary with parameters:
                - base_activation: Base activation level for memories (default: 0.1)
                - activation_threshold: Threshold for considering memories activated (default: 0.2)
                - spreading_factor: Factor for spreading activation (default: 0.5)
                - max_spreading_hops: Maximum hops for spreading activation (default: 2)
                - decay_rate: Rate of activation decay (default: 0.9)
                - long_term_decay_rate: Decay rate for long-term memories (default: 0.99)
                - long_term_threshold: Time threshold for long-term memory (default: 7 days)
        """
        self.base_activation = config.get("base_activation", 0.1)
        self.activation_threshold = config.get("activation_threshold", 0.2)
        self.spreading_factor = config.get("spreading_factor", 0.5)
        self.max_spreading_hops = config.get("max_spreading_hops", 2)
        self.decay_rate = config.get("decay_rate", 0.9)
        self.long_term_decay_rate = config.get("long_term_decay_rate", 0.99)
        self.long_term_threshold = config.get("long_term_threshold", 7 * 86400)

        # Set memory store and linker if provided
        if "memory_store" in config:
            self.memory_store = config["memory_store"]

        if "associative_linker" in config:
            self.associative_linker = config["associative_linker"]

        # Initialize activation levels if memory store is available
        if self.memory_store is not None:
            self._initialize_activations()

    def _initialize_activations(self) -> None:
        """Initialize activation levels for all memories."""
        if self.memory_store is None:
            return

        # Get all memories
        all_memories = self.memory_store.get_all()
        current_time = time.time()

        # Initialize activation levels
        for memory in all_memories:
            # Use existing activation if available in metadata
            if "activation" in memory.metadata:
                self.activation_levels[memory.id] = memory.metadata["activation"]
            else:
                # Otherwise set to base activation
                self.activation_levels[memory.id] = self.base_activation

            # Track activation time
            self.activation_timestamps[memory.id] = memory.metadata.get(
                "last_accessed", current_time
            )

            # Initialize history
            self.activation_history[memory.id] = [(current_time, self.activation_levels[memory.id])]

        # Set last update time
        self.last_update_time = current_time

    def activate_memory(
        self, memory_id: MemoryID, activation_level: float = 1.0, spread: bool = True
    ) -> dict[MemoryID, float]:
        """
        Activate a specific memory and optionally spread activation.

        Args:
            memory_id: ID of the memory to activate
            activation_level: Activation level to set (default: 1.0)
            spread: Whether to spread activation to connected memories (default: True)

        Returns:
            Dictionary mapping memory IDs to their updated activation levels
        """
        # Update activation level
        self.activation_levels[memory_id] = min(1.0, activation_level)
        current_time = time.time()

        # Update timestamp
        self.activation_timestamps[memory_id] = current_time

        # Add to history
        self.activation_history[memory_id].append((current_time, activation_level))

        # Limit history size
        if len(self.activation_history[memory_id]) > 100:
            self.activation_history[memory_id] = self.activation_history[memory_id][-100:]

        # Update memory store if available
        if self.memory_store is not None:
            try:
                self.memory_store.update_metadata(
                    memory_id, {"activation": activation_level, "last_accessed": current_time}
                )
            except KeyError:
                # Memory might not exist in store
                pass

        # Spread activation if requested
        activated_memories = {memory_id: activation_level}

        if spread and self.associative_linker is not None:
            spread_activations = self._spread_activation(memory_id, activation_level)
            activated_memories.update(spread_activations)

        return activated_memories

    def _spread_activation(
        self, source_id: MemoryID, source_activation: float
    ) -> dict[MemoryID, float]:
        """
        Spread activation from a source memory through associative links.

        Args:
            source_id: ID of the source memory
            source_activation: Activation level of the source memory

        Returns:
            Dictionary mapping memory IDs to their updated activation levels
        """
        if self.associative_linker is None:
            return {}

        # Use associative linker to traverse network
        activations = self.associative_linker.traverse_associative_network(
            start_id=source_id, max_hops=self.max_spreading_hops, min_strength=0.1
        )

        # Remove source memory from results
        if source_id in activations:
            del activations[source_id]

        # Apply spreading factor
        spread_activations = {}
        current_time = time.time()

        for memory_id, link_strength in activations.items():
            # Calculate spread activation level
            spread_activation = source_activation * link_strength * self.spreading_factor

            # Combine with existing activation (don't decrease existing activation)
            current_activation = self.activation_levels.get(memory_id, self.base_activation)
            new_activation = max(current_activation, spread_activation)

            # Update activation level
            self.activation_levels[memory_id] = new_activation

            # Add to history
            self.activation_history[memory_id].append((current_time, new_activation))

            # Limit history size
            if len(self.activation_history[memory_id]) > 100:
                self.activation_history[memory_id] = self.activation_history[memory_id][-100:]

            # Update timestamp if activation increased
            if new_activation > current_activation:
                self.activation_timestamps[memory_id] = current_time

            # Update memory store if available
            if self.memory_store is not None:
                try:
                    self.memory_store.update_metadata(
                        memory_id,
                        {
                            "activation": new_activation,
                            "activation_source": source_id,
                            "last_accessed": current_time,
                        },
                    )
                except KeyError:
                    # Memory might not exist in store
                    pass

            # Add to result
            spread_activations[memory_id] = new_activation

        return spread_activations

    def apply_activation_decay(self, current_time: Optional[float] = None) -> None:
        """
        Apply activation decay to all memories.

        Args:
            current_time: Current timestamp for decay calculation (default: current time)
        """
        if current_time is None:
            current_time = time.time()

        # Skip if no significant time has passed
        time_since_update = current_time - self.last_update_time
        if time_since_update < 1.0:  # Less than 1 second
            return

        # Update all activation levels
        for memory_id, activation in list(self.activation_levels.items()):
            # Get activation timestamp
            last_activated = self.activation_timestamps.get(memory_id, self.last_update_time)

            # Calculate time since last activation
            time_since_activation = current_time - last_activated

            # Choose decay rate based on time since activation
            if time_since_activation > self.long_term_threshold:
                # Long-term decay
                decay_rate = self.long_term_decay_rate
            else:
                # Short-term decay
                decay_rate = self.decay_rate

            # Apply exponential decay
            decay_factor = decay_rate ** (time_since_update / 3600)  # Scale by hour
            decayed_activation = max(self.base_activation, activation * decay_factor)

            # Update activation level
            self.activation_levels[memory_id] = decayed_activation

            # Add to history (but less frequently for decay)
            if time_since_update > 3600:  # Only record hourly for decay
                self.activation_history[memory_id].append((current_time, decayed_activation))

            # Update memory store if available
            if self.memory_store is not None:
                try:
                    self.memory_store.update_metadata(memory_id, {"activation": decayed_activation})
                except KeyError:
                    # Memory might not exist in store
                    pass

        # Update last update time
        self.last_update_time = current_time

    def get_activation_level(self, memory_id: MemoryID) -> float:
        """
        Get the current activation level for a memory.

        Args:
            memory_id: ID of the memory

        Returns:
            Current activation level (0.0-1.0)
        """
        return self.activation_levels.get(memory_id, self.base_activation)

    def get_activated_memories(self, threshold: Optional[float] = None) -> dict[MemoryID, float]:
        """
        Get all memories with activation above a threshold.

        Args:
            threshold: Activation threshold (default: self.activation_threshold)

        Returns:
            Dictionary mapping memory IDs to activation levels
        """
        if threshold is None:
            threshold = self.activation_threshold

        # Apply decay to ensure current activations
        self.apply_activation_decay()

        # Filter by threshold
        return {
            memory_id: activation
            for memory_id, activation in self.activation_levels.items()
            if activation >= threshold
        }

    def get_activation_history(self, memory_id: MemoryID) -> list[tuple[float, float]]:
        """
        Get activation history for a memory.

        Args:
            memory_id: ID of the memory

        Returns:
            List of (timestamp, activation) tuples
        """
        return self.activation_history.get(memory_id, [])

    def boost_by_recency(
        self, results: list[dict[str, Any]], boost_factor: float = 2.0
    ) -> list[dict[str, Any]]:
        """
        Boost retrieval results by activation level.

        Args:
            results: Retrieval results to boost
            boost_factor: Maximum boost factor (default: 2.0)

        Returns:
            Boosted results with updated relevance scores
        """
        if not results:
            return results

        # Apply decay to ensure current activations
        self.apply_activation_decay()

        # Create a copy of results to avoid modifying originals
        boosted_results = []

        for result in results:
            # Deep copy
            boosted_result = dict(result)

            # Get memory ID
            memory_id = result.get("memory_id")
            if memory_id is None:
                boosted_results.append(boosted_result)
                continue

            # Get activation level
            activation = self.get_activation_level(memory_id)

            # Calculate boost based on activation (linear scaling)
            # activation=0 -> boost=1.0, activation=1 -> boost=boost_factor
            activation_boost = 1.0 + (boost_factor - 1.0) * activation

            # Apply boost to relevance score
            original_score = boosted_result.get("relevance_score", 0.0)
            boosted_result["relevance_score"] = original_score * activation_boost
            boosted_result["activation_boost"] = activation_boost
            boosted_result["activation_level"] = activation

            boosted_results.append(boosted_result)

        # Sort by boosted score
        boosted_results.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)

        return boosted_results

    def generate_activation_heatmap(self, top_k: int = 20) -> dict[str, Any]:
        """
        Generate data for visualizing activation heatmap.

        Args:
            top_k: Number of top activated memories to include

        Returns:
            Dictionary with heatmap data
        """
        # Apply decay to ensure current activations
        self.apply_activation_decay()

        # Get top activated memories
        activations = sorted(self.activation_levels.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        # Format for visualization
        memory_ids = [memory_id for memory_id, _ in activations]
        activation_values = [activation for _, activation in activations]

        # Get memory content if store is available
        memory_contents = {}
        if self.memory_store is not None:
            for memory_id in memory_ids:
                try:
                    memory = self.memory_store.get(memory_id)
                    if hasattr(memory, "content") and isinstance(memory.content, dict):
                        text = memory.content.get("text", "")
                    else:
                        text = str(memory.content) if hasattr(memory, "content") else ""

                    # Truncate long text
                    if len(text) > 100:
                        text = text[:97] + "..."

                    memory_contents[memory_id] = text
                except KeyError:
                    memory_contents[memory_id] = "Unknown"

        return {
            "memory_ids": memory_ids,
            "activation_values": activation_values,
            "memory_contents": memory_contents,
            "timestamp": time.time(),
        }

    def get_activation_pattern(self) -> dict[str, Any]:
        """
        Get the current activation pattern across all memories.

        Returns:
            Dictionary with activation pattern data
        """
        # Apply decay to ensure current activations
        self.apply_activation_decay()

        # Calculate activation statistics
        total_memories = len(self.activation_levels)
        activated_memories = sum(
            1 for a in self.activation_levels.values() if a >= self.activation_threshold
        )
        average_activation = sum(self.activation_levels.values()) / max(1, total_memories)

        # Find activation clusters (memories that are similar in activation)
        activation_values = list(self.activation_levels.values())

        # Simplistic clustering by binning
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        hist, _ = np.histogram(activation_values, bins=bins)

        return {
            "total_memories": total_memories,
            "activated_memories": activated_memories,
            "average_activation": average_activation,
            "activation_distribution": hist.tolist(),
            "activation_bins": bins,
            "timestamp": time.time(),
        }
