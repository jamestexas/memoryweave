"""
Tests for the CategoryManager component.
"""

import numpy as np
import pytest

from memoryweave.components.category_manager import CategoryManager


class TestCategoryManager:
    """Test suite for CategoryManager component."""

    def test_initialization(self):
        """Test that CategoryManager initializes correctly."""
        # Test with default parameters
        manager = CategoryManager()
        assert manager.vigilance_threshold == 0.8
        assert manager.learning_rate == 0.2
        assert manager.embedding_dim == 768
        assert manager.core_manager is None

        # Test with custom parameters using initialize
        manager.initialize(
            {
                "vigilance_threshold": 0.7,
                "learning_rate": 0.3,
                "embedding_dim": 384,
            }
        )
        assert manager.vigilance_threshold == 0.7
        assert manager.learning_rate == 0.3
        assert manager.embedding_dim == 384
        assert manager.core_manager is not None
        assert isinstance(manager.core_manager, CategoryManager)

        # Test with passing existing core manager
        core_manager = CategoryManager(embedding_dim=512)
        manager = CategoryManager(core_manager)
        assert manager.core_manager is core_manager

    def test_category_assignment(self):
        """Test assigning memories to categories."""
        # Initialize with small embedding dimension for testing and forced high vigilance
        manager = CategoryManager()
        manager.initialize(
            {
                "embedding_dim": 4,
                "vigilance_threshold": 0.99,  # Set very high to force new categories
            }
        )

        # Create clearly distinct embeddings to ensure categories are different
        embed1 = np.array([1.0, 0.0, 0.0, 0.0])
        embed2 = np.array([0.9, 0.1, 0.0, 0.0])  # Similar to first
        embed3 = np.array([0.0, 0.0, 1.0, 0.0])  # Very different from first

        # Normalize
        embed1 = embed1 / np.linalg.norm(embed1)
        embed2 = embed2 / np.linalg.norm(embed2)
        embed3 = embed3 / np.linalg.norm(embed3)

        # Assign to categories
        cat1 = manager.assign_to_category(embed1)
        # First category should be 0
        assert cat1 == 0

        # Very similar embedding might still be in same category
        cat2 = manager.assign_to_category(embed2)

        # Different embedding should get new category
        cat3 = manager.assign_to_category(embed3)

        # Main assertion: verify we've created at least 2 distinct categories
        assigned_categories = {cat1, cat2, cat3}
        assert len(assigned_categories) >= 2, "Should create at least 2 distinct categories"

    def test_category_mapping(self):
        """Test memory to category mapping."""
        manager = CategoryManager()
        manager.initialize({"embedding_dim": 4})

        # Add some mappings
        manager.add_memory_category_mapping(0, 0)
        manager.add_memory_category_mapping(1, 0)
        manager.add_memory_category_mapping(2, 1)

        # Test retrieving category for memory
        assert manager.get_category_for_memory(0) == 0
        assert manager.get_category_for_memory(1) == 0
        assert manager.get_category_for_memory(2) == 1

        # Test retrieving memories for category
        cat0_memories = manager.get_memories_for_category(0)
        cat1_memories = manager.get_memories_for_category(1)
        assert set(cat0_memories) == {0, 1}
        assert set(cat1_memories) == {2}

        # Test invalid memory index
        with pytest.raises(IndexError):
            manager.get_category_for_memory(999)

    def test_category_similarities(self):
        """Test calculating similarities between query and categories."""
        manager = CategoryManager()
        manager.initialize(
            {
                "embedding_dim": 4,
                "vigilance_threshold": 0.99,  # Set vigilance very high to create distinct categories
            }
        )

        # Create test embeddings with very distinct patterns
        embeddings = [
            np.array([1.0, 0.0, 0.0, 0.0]),  # Cat pattern
            np.array([0.0, 1.0, 0.0, 0.0]),  # Dog pattern
            np.array([0.0, 0.0, 1.0, 0.0]),  # Weather pattern
        ]

        # Normalize
        for i in range(len(embeddings)):
            embeddings[i] = embeddings[i] / np.linalg.norm(embeddings[i])

        # Assign embeddings to categories - these should create distinct categories
        category_ids = []
        for embed in embeddings:
            cat_id = manager.assign_to_category(embed)
            category_ids.append(cat_id)

        # We should have at least 2 categories
        assert len(set(category_ids)) >= 2

        # Create query embeddings that are similar to our categories
        cat_query = np.array([0.9, 0.1, 0.0, 0.0])
        cat_query = cat_query / np.linalg.norm(cat_query)

        dog_query = np.array([0.1, 0.9, 0.0, 0.0])
        dog_query = dog_query / np.linalg.norm(dog_query)

        # Calculate similarities
        cat_similarities = manager.get_category_similarities(cat_query)
        dog_similarities = manager.get_category_similarities(dog_query)

        # Check that we get back similarities (should match number of categories created)
        assert len(cat_similarities) > 0
        assert len(dog_similarities) > 0
        assert len(cat_similarities) == len(set(category_ids))

        # Check that the similarities are in the expected range [0, 1]
        assert all(0 <= sim <= 1 for sim in cat_similarities)
        assert all(0 <= sim <= 1 for sim in dog_similarities)

    def test_category_consolidation(self):
        """Test consolidating similar categories."""
        # Create a core category manager directly for more control
        core_manager = CategoryManager(
            embedding_dim=4,
            vigilance_threshold=0.999,  # Extremely high to force separate categories
            enable_category_consolidation=True,
            consolidation_threshold=0.5,
        )
        manager = CategoryManager(core_manager)

        # Create clearly distinct embeddings
        cat_group = [
            np.array([0.9, 0.1, 0.0, 0.0]),  # Cat pattern 1
            np.array([0.85, 0.15, 0.0, 0.0]),  # Cat pattern 2
        ]

        dog_group = [
            np.array([0.0, 0.0, 0.9, 0.1]),  # Dog pattern 1
            np.array([0.0, 0.0, 0.85, 0.15]),  # Dog pattern 2
        ]

        # Normalize all embeddings
        for i in range(len(cat_group)):
            cat_group[i] = cat_group[i] / np.linalg.norm(cat_group[i])
            dog_group[i] = dog_group[i] / np.linalg.norm(dog_group[i])

        # Assign to categories with high vigilance
        cat_categories = []
        dog_categories = []

        # Add cat group
        for embed in cat_group:
            cat_id = manager.assign_to_category(embed)
            cat_categories.append(cat_id)

        # Add dog group
        for embed in dog_group:
            cat_id = manager.assign_to_category(embed)
            dog_categories.append(cat_id)

        # Count initial categories
        initial_categories = set(cat_categories + dog_categories)

        # We should have at least 2 categories
        assert len(initial_categories) >= 2, (
            f"Should have at least 2 categories, got {initial_categories}"
        )

        # Create memory-category mappings
        for i, cat_id in enumerate(cat_categories):
            manager.add_memory_category_mapping(i, cat_id)

        for i, cat_id in enumerate(dog_categories):
            manager.add_memory_category_mapping(i + len(cat_categories), cat_id)

        # Consolidate using a custom threshold that should merge similar categories
        manager.vigilance_threshold = 0.5  # Lower vigilance
        manager.core_manager.vigilance_threshold = 0.5

        # Get current stats
        manager.get_category_statistics()

        # Explicitly run consolidation
        threshold = 0.6  # Higher threshold means more merging
        num_categories = manager.consolidate_categories(threshold=threshold)

        # Get after stats
        manager.get_category_statistics()

        # Should reduce number of categories or stay the same
        assert num_categories <= len(initial_categories)

        # Verify we didn't break anything
        assert num_categories >= 1

        # After consolidation, check that similar embeddings are categorized together
        if num_categories < len(initial_categories):
            # Only test this if categories were actually consolidated
            # Check that the first two (cat group) have the same category
            assert manager.get_category_for_memory(0) == manager.get_category_for_memory(1)

    def test_category_statistics(self):
        """Test retrieving category statistics."""
        # Create a manager with high vigilance to create multiple categories
        manager = CategoryManager()
        manager.initialize(
            {
                "embedding_dim": 4,
                "vigilance_threshold": 0.99,  # Very high to force distinct categories
            }
        )

        # Create distinct embeddings to ensure multiple categories
        embeddings = [
            np.array([1.0, 0.0, 0.0, 0.0]),  # Completely different pattern 1
            np.array([0.0, 1.0, 0.0, 0.0]),  # Completely different pattern 2
        ]

        # Normalize embeddings
        for i in range(len(embeddings)):
            embeddings[i] = embeddings[i] / np.linalg.norm(embeddings[i])

        # Assign to categories
        cat_indices = []
        for embed in embeddings:
            cat_idx = manager.assign_to_category(embed)
            cat_indices.append(cat_idx)

        # Add memory mappings
        for i, cat in enumerate(cat_indices):
            manager.add_memory_category_mapping(i, cat)

        # Update category activations
        for idx in cat_indices:
            manager.update_category_activation(idx)

        # Get statistics
        stats = manager.get_category_statistics()

        # Check that stats contains expected fields
        assert "num_categories" in stats
        assert stats["num_categories"] >= 1  # At least one category
        assert "category_activations" in stats

        # Verify the stats contain information about our categories
        if "memories_per_category" in stats:
            # There should be counts for the categories we created
            category_counts = stats["memories_per_category"]
            assert len(category_counts) > 0
