# tests/unit/components/test_associative_linking.py
import time

import numpy as np
import pytest

from memoryweave.components.associative_linking import AssociativeMemoryLinker
from memoryweave.storage.base_store import StandardMemoryStore


class TestAssociativeMemoryLinker:
    """Test the AssociativeMemoryLinker component."""

    @pytest.fixture
    def memory_store(self):
        """Create a memory store with some test data."""
        store = StandardMemoryStore()

        # Add test memories
        embedding1 = np.array([0.8, 0.1, 0.1])
        embedding2 = np.array([0.7, 0.2, 0.1])  # Similar to first
        embedding3 = np.array([0.1, 0.8, 0.1])  # Different
        embedding4 = np.array([0.1, 0.7, 0.2])  # Similar to third

        current_time = time.time()

        # Add with sequential timestamps for predictable temporal relationships
        store.add(embedding1, "Python programming language", {"created_at": current_time})  # id=0
        store.add(embedding2, "Programming in Python", {"created_at": current_time + 60})  # id=1
        store.add(
            embedding3, "Machine learning algorithms", {"created_at": current_time + 120}
        )  # id=2
        store.add(embedding4, "AI and ML concepts", {"created_at": current_time + 180})  # id=3

        return store

    @pytest.fixture
    def linker(self, memory_store):
        """Create an associative memory linker for testing."""
        linker = AssociativeMemoryLinker(memory_store)
        linker.initialize(
            {
                "similarity_threshold": 0.5,
                "temporal_weight": 0.3,
                "semantic_weight": 0.7,
            }
        )
        return linker

    def test_calculate_similarity(self, linker):
        """Test similarity calculation between embeddings."""
        # Same vectors should have similarity 1.0
        vec1 = np.array([0.6, 0.8, 0.0])
        vec2 = np.array([0.6, 0.8, 0.0])
        similarity = linker._calculate_similarity(vec1, vec2)
        assert similarity == 1.0

        # Orthogonal vectors should have similarity 0.0
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        similarity = linker._calculate_similarity(vec1, vec2)
        assert similarity == 0.0

        # Similar vectors should have similarity between 0 and 1
        vec1 = np.array([0.8, 0.2, 0.0])
        vec2 = np.array([0.7, 0.3, 0.0])
        similarity = linker._calculate_similarity(vec1, vec2)
        assert 0.9 < similarity < 1.0

    def test_establish_links_for_memory(self, linker, memory_store):
        """Test establishing links for a single memory."""
        # Get memory data for the first memory
        memory = memory_store.get(0)
        memory_data = {
            "id": memory.id,
            "embedding": memory.embedding,
            "created_at": memory.metadata.get("created_at"),
        }

        # Establish links
        linker._establish_links_for_memory(0, memory_data, {})

        # Verify links were created
        links = linker.associative_links.get(0, [])
        assert len(links) > 0

        # First memory should link strongly to second memory (similar embedding)
        assert any(mem_id == 1 for mem_id, _ in links)

        # Links should be sorted by strength (highest first)
        strengths = [strength for _, strength in links]
        assert all(strengths[i] >= strengths[i + 1] for i in range(len(strengths) - 1))

        # Verify bidirectional links were created
        reverse_links = linker.associative_links.get(1, [])
        assert any(mem_id == 0 for mem_id, _ in reverse_links)

    def test_rebuild_all_links(self, linker):
        """Test rebuilding all associative links."""
        # Clear existing links
        linker.associative_links.clear()

        # Rebuild links
        linker._rebuild_all_links()

        # Verify all memories have links
        for memory_id in range(4):  # 4 test memories
            assert memory_id in linker.associative_links

        # Check specific linking patterns:
        # Memory 0 should link to memory 1 (similar)
        links_0 = linker.associative_links.get(0, [])
        assert any(mem_id == 1 for mem_id, _ in links_0)

        # Memory 2 should link to memory 3 (similar)
        links_2 = linker.associative_links.get(2, [])
        assert any(mem_id == 3 for mem_id, _ in links_2)

        # Memory 0 should not strongly link to memory 2 (very different)
        links_0 = linker.associative_links.get(0, [])
        assert not any(mem_id == 2 for mem_id, _ in links_0)

    def test_traverse_associative_network(self, linker):
        """Test traversing the associative network."""
        # Ensure links are built
        if not linker.associative_links:
            linker._rebuild_all_links()

        # Traverse from memory 0
        activations = linker.traverse_associative_network(0, max_hops=2, min_strength=0.1)

        # Should activate at least itself and memory 1
        assert 0 in activations
        assert 1 in activations

        # Starting node should have highest activation
        assert activations[0] == 1.0

        # With 2 hops, we might reach memory 3 through memory 1->2->3 path
        # But the activation would be lower due to hop decay
        if 3 in activations:
            assert activations[3] < activations[1]

    def test_find_path_between_memories(self, linker):
        """Test finding paths between memories."""
        # Ensure links are built
        if not linker.associative_links:
            linker._rebuild_all_links()

        # Find path from memory 0 to memory 1 (direct neighbors)
        path = linker.find_path_between_memories(0, 1)
        assert len(path) > 0
        assert path[0][0] == 0  # First node is starting point
        assert path[-1][0] == 1  # Last node is target

        # Try to find path between less related memories
        # From memory 0 to memory 3 (might require multiple hops)
        path = linker.find_path_between_memories(0, 3, max_hops=3)

        # If path found, verify it follows connected nodes
        if path:
            # Path should be in order: 0 -> some intermediate nodes -> 3
            assert path[0][0] == 0
            assert path[-1][0] == 3

            # Verify each step in path is connected
            for i in range(len(path) - 1):
                curr_id = path[i][0]
                next_id = path[i + 1][0]
                curr_links = linker.associative_links.get(curr_id, [])
                assert any(link_id == next_id for link_id, _ in curr_links)

    def test_create_associative_link(self, linker):
        """Test manually creating an associative link."""
        # Create a link that doesn't naturally exist
        linker.create_associative_link("0", "2", 0.75)

        # Verify the link was created in both directions
        links_0 = linker.associative_links.get("0", [])
        assert any(mem_id == "2" and strength == 0.75 for mem_id, strength in links_0)

        links_2 = linker.associative_links.get("2", [])
        assert any(mem_id == "0" for mem_id, _ in links_2)

        # The reverse link should have slightly lower strength
        reverse_strength = next(strength for mem_id, strength in links_2 if mem_id == "0")
        assert reverse_strength < 0.75
        assert reverse_strength >= 0.75 * 0.8  # 80% of original

        # Update the link with higher strength
        linker.create_associative_link("0", "2", 0.9)

        # Verify the link was updated
        links_0 = linker.associative_links.get("0", [])
        updated_strength = next(strength for mem_id, strength in links_0 if mem_id == "2")
        assert updated_strength == 0.9

    def test_process_query(self, linker):
        """Test processing a query through associative memory."""
        # Set up context with some initial results
        context = {
            "results": [
                {"memory_id": 0, "relevance_score": 0.9},
                {"memory_id": 2, "relevance_score": 0.7},
            ]
        }

        # Ensure we have associative links established
        if not linker.associative_links:
            linker._rebuild_all_links()

        # Process the query
        result = linker.process_query("Python programming", context)

        # Should add associative_memories to context
        assert "associative_memories" in result

        # Should contain at least the seed memories and linked memories
        activations = result["associative_memories"]
        assert 0 in activations
        assert 2 in activations

        # Should contain memory 1 (linked to 0)
        assert 1 in activations

        # Memory 1's activation should be strong due to link to memory 0
        assert activations[1] > 0.5

        # Memory 0 should have highest activation (seed memory)
        assert activations[0] == 1.0
