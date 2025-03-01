"""Activation management for MemoryWeave.

This module provides implementations for memory activation tracking,
including activation updates, decay, and retrieval based on activation level.
"""

from typing import Dict, List, Tuple, Optional
import time
import math
import numpy as np

from memoryweave.interfaces.memory import IActivationManager, MemoryID


class ActivationManager(IActivationManager):
    """Implementation of memory activation management."""
    
    def __init__(self, 
                 initial_activation: float = 0.0,
                 max_activation: float = 10.0,
                 min_activation: float = -10.0):
        """Initialize the activation manager.
        
        Args:
            initial_activation: Default activation level for new memories
            max_activation: Maximum activation level
            min_activation: Minimum activation level
        """
        self._activations: Dict[MemoryID, float] = {}
        self._last_updated: Dict[MemoryID, float] = {}
        self._initial_activation = initial_activation
        self._max_activation = max_activation
        self._min_activation = min_activation
    
    def update_activation(self, memory_id: MemoryID, activation_delta: float) -> None:
        """Update the activation level of a memory."""
        # Initialize if not present
        if memory_id not in self._activations:
            self._activations[memory_id] = self._initial_activation
        
        # Update activation level
        current_activation = self._activations[memory_id]
        new_activation = current_activation + activation_delta
        
        # Clamp activation to min/max
        new_activation = max(self._min_activation, min(self._max_activation, new_activation))
        
        # Store updated activation
        self._activations[memory_id] = new_activation
        self._last_updated[memory_id] = time.time()
    
    def get_activation(self, memory_id: MemoryID) -> float:
        """Get the current activation level of a memory."""
        return self._activations.get(memory_id, self._initial_activation)
    
    def decay_activations(self, decay_factor: float) -> None:
        """Apply decay to all memory activations."""
        for memory_id in list(self._activations.keys()):
            current_activation = self._activations[memory_id]
            
            # Decay toward zero
            if current_activation > 0:
                new_activation = current_activation * (1.0 - decay_factor)
            else:
                new_activation = current_activation * (1.0 - decay_factor)
            
            # Store updated activation
            self._activations[memory_id] = new_activation
    
    def get_most_active(self, k: int) -> List[Tuple[MemoryID, float]]:
        """Get the k most active memories."""
        if not self._activations:
            return []
        
        # Sort by activation (descending)
        sorted_activations = sorted(
            self._activations.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Return top k
        return sorted_activations[:k]


class TemporalActivationManager(IActivationManager):
    """Implementation of memory activation with temporal decay.
    
    This implementation applies continuous decay to activations based on time,
    making memories gradually less active as time passes without access.
    """
    
    def __init__(self, 
                 initial_activation: float = 0.0,
                 max_activation: float = 10.0,
                 min_activation: float = -10.0,
                 half_life_days: float = 7.0):
        """Initialize the temporal activation manager.
        
        Args:
            initial_activation: Default activation level for new memories
            max_activation: Maximum activation level
            min_activation: Minimum activation level
            half_life_days: Number of days for activation to decay by half
        """
        self._activations: Dict[MemoryID, float] = {}
        self._last_updated: Dict[MemoryID, float] = {}
        self._initial_activation = initial_activation
        self._max_activation = max_activation
        self._min_activation = min_activation
        
        # Calculate decay rate from half-life
        # If half_life_days = 7, then after 7 days, activation should be 0.5 of original
        # decay_rate = -ln(0.5) / half_life_seconds
        half_life_seconds = half_life_days * 24 * 60 * 60
        self._decay_rate = math.log(2) / half_life_seconds
    
    def update_activation(self, memory_id: MemoryID, activation_delta: float) -> None:
        """Update the activation level of a memory."""
        current_time = time.time()
        
        # Get current activation with decay applied
        current_activation = self._get_decayed_activation(memory_id, current_time)
        
        # Update activation level
        new_activation = current_activation + activation_delta
        
        # Clamp activation to min/max
        new_activation = max(self._min_activation, min(self._max_activation, new_activation))
        
        # Store updated activation
        self._activations[memory_id] = new_activation
        self._last_updated[memory_id] = current_time
    
    def get_activation(self, memory_id: MemoryID) -> float:
        """Get the current activation level of a memory with decay applied."""
        return self._get_decayed_activation(memory_id, time.time())
    
    def decay_activations(self, decay_factor: float) -> None:
        """Apply additional manual decay to all memory activations.
        
        This is in addition to the continuous temporal decay.
        """
        current_time = time.time()
        for memory_id in list(self._activations.keys()):
            # Get current activation with decay applied
            current_activation = self._get_decayed_activation(memory_id, current_time)
            
            # Apply additional decay
            if current_activation > 0:
                new_activation = current_activation * (1.0 - decay_factor)
            else:
                new_activation = current_activation * (1.0 - decay_factor)
            
            # Store updated activation
            self._activations[memory_id] = new_activation
            self._last_updated[memory_id] = current_time
    
    def get_most_active(self, k: int) -> List[Tuple[MemoryID, float]]:
        """Get the k most active memories with decay applied."""
        if not self._activations:
            return []
        
        # Calculate current activations with decay
        current_time = time.time()
        current_activations = {
            memory_id: self._get_decayed_activation(memory_id, current_time)
            for memory_id in self._activations.keys()
        }
        
        # Sort by activation (descending)
        sorted_activations = sorted(
            current_activations.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Return top k
        return sorted_activations[:k]
    
    def _get_decayed_activation(self, memory_id: MemoryID, current_time: float) -> float:
        """Get activation with temporal decay applied."""
        # If memory not present, return initial activation
        if memory_id not in self._activations:
            return self._initial_activation
        
        # Calculate time since last update
        last_updated = self._last_updated.get(memory_id, current_time)
        time_elapsed = current_time - last_updated
        
        # Apply exponential decay
        original_activation = self._activations[memory_id]
        decay_factor = math.exp(-self._decay_rate * time_elapsed)
        
        # Decay toward zero
        if original_activation > 0:
            decayed_activation = original_activation * decay_factor
        else:
            decayed_activation = original_activation * decay_factor
        
        return decayed_activation