"""
Unit tests for the MemoryDecayComponent.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from memoryweave.components.memory_decay import MemoryDecayComponent
from memoryweave.storage.activation import ActivationManager


class MemoryDecayComponentTest(unittest.TestCase):
    """
    Unit tests for the MemoryDecayComponent.
    """

    def setUp(self):
        """Set up test environment before each test."""
        self.decay_component = MemoryDecayComponent()
        
        # Create mock memory with activation support
        self.mock_memory = MagicMock()
        self.activation_manager = ActivationManager()
        
        # Add some activations
        self.activation_manager.update_activation("memory1", 1.0)
        self.activation_manager.update_activation("memory2", 2.0)
        self.activation_manager.update_activation("memory3", 0.5)
        
        # Setup mock to return our activation manager
        self.mock_memory.get_activation_manager.return_value = self.activation_manager
        
    def test_initialization(self):
        """Test initialization with configuration."""
        config = {
            "memory_decay_enabled": True,
            "memory_decay_rate": 0.95,
            "memory_decay_interval": 5,
            "memory": self.mock_memory
        }
        
        self.decay_component.initialize(config)
        
        self.assertTrue(self.decay_component.memory_decay_enabled)
        self.assertEqual(self.decay_component.memory_decay_rate, 0.95)
        self.assertEqual(self.decay_component.memory_decay_interval, 5)
        self.assertEqual(self.decay_component.memory, self.mock_memory)
        
    def test_process_without_decay(self):
        """Test processing without applying decay (not at interval)."""
        self.decay_component.initialize({
            "memory_decay_enabled": True,
            "memory_decay_rate": 0.9,
            "memory_decay_interval": 10,
            "memory": self.mock_memory
        })
        
        # Get initial activations
        initial_activation1 = self.activation_manager.get_activation("memory1")
        initial_activation2 = self.activation_manager.get_activation("memory2")
        
        # Process once (interaction count = 1)
        result = self.decay_component.process({}, {})
        
        # Verify that no decay was applied
        self.assertEqual(self.activation_manager.get_activation("memory1"), initial_activation1)
        self.assertEqual(self.activation_manager.get_activation("memory2"), initial_activation2)
        self.assertFalse(result["memory_decay_applied"])
        self.assertEqual(result["interaction_count"], 1)
        
    def test_process_with_decay(self):
        """Test processing with decay applied (at interval)."""
        self.decay_component.initialize({
            "memory_decay_enabled": True,
            "memory_decay_rate": 0.5,  # 50% decay for easy testing
            "memory_decay_interval": 1,  # Decay every interaction
            "memory": self.mock_memory
        })
        
        # Get initial activations
        initial_activation1 = self.activation_manager.get_activation("memory1")
        initial_activation2 = self.activation_manager.get_activation("memory2")
        
        # Process once (interaction count = 1)
        result = self.decay_component.process({}, {})
        
        # Verify that decay was applied
        # Expect 50% decay (memory_decay_rate = 0.5)
        self.assertLess(self.activation_manager.get_activation("memory1"), initial_activation1)
        self.assertLess(self.activation_manager.get_activation("memory2"), initial_activation2)
        self.assertTrue(result["memory_decay_applied"])
        self.assertEqual(result["interaction_count"], 1)
        
    def test_decay_disabled(self):
        """Test that no decay is applied when disabled."""
        self.decay_component.initialize({
            "memory_decay_enabled": False,
            "memory_decay_rate": 0.5,
            "memory_decay_interval": 1,
            "memory": self.mock_memory
        })
        
        # Get initial activations
        initial_activation1 = self.activation_manager.get_activation("memory1")
        
        # Process once
        self.decay_component.process({}, {})
        
        # Verify that no decay was applied
        self.assertEqual(self.activation_manager.get_activation("memory1"), initial_activation1)
        
    def test_legacy_memory_support(self):
        """Test decay with legacy memory format."""
        # Create a class with the expected attributes for better mocking
        class MockLegacyMemory:
            def __init__(self):
                self.activation_levels = np.array([1.0, 2.0, 0.5])

        mock_legacy_memory = MockLegacyMemory()
        
        # Initialize with legacy memory
        self.decay_component.initialize({
            "memory_decay_enabled": True,
            "memory_decay_rate": 0.5,
            "memory_decay_interval": 1,
            "memory": mock_legacy_memory
        })
        
        # Process once
        self.decay_component.process({}, {})
        
        # Verify decay was applied to legacy format
        np.testing.assert_array_almost_equal(
            mock_legacy_memory.activation_levels,
            np.array([0.5, 1.0, 0.25])  # 50% of original values
        )
    
    def test_art_clustering_decay(self):
        """Test decay of category activations with ART clustering."""
        # Create a class with the expected attributes for better mocking
        class MockLegacyMemory:
            def __init__(self):
                self.activation_levels = np.array([1.0, 2.0, 0.5])
                self.category_activations = np.array([0.8, 1.2, 0.3])

        mock_legacy_memory = MockLegacyMemory()
        
        # Initialize with legacy memory and ART clustering enabled
        self.decay_component.initialize({
            "memory_decay_enabled": True,
            "memory_decay_rate": 0.5,
            "memory_decay_interval": 1,
            "memory": mock_legacy_memory,
            "art_clustering_enabled": True
        })
        
        # Process once
        self.decay_component.process({}, {})
        
        # Verify decay was applied to both activation levels and category activations
        np.testing.assert_array_almost_equal(
            mock_legacy_memory.activation_levels,
            np.array([0.5, 1.0, 0.25])  # 50% of original values
        )
        
        np.testing.assert_array_almost_equal(
            mock_legacy_memory.category_activations,
            np.array([0.4, 0.6, 0.15])  # 50% of original values
        )


if __name__ == "__main__":
    unittest.main()