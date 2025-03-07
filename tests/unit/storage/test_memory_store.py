import unittest

import numpy as np

from memoryweave.interfaces.memory import Memory
from memoryweave.storage.memory_store import StandardMemoryStore


class TestStandardMemoryStore(unittest.TestCase):
    def setUp(self):
        self.store = StandardMemoryStore()
        self.test_embedding = np.array([0.1, 0.2, 0.3])
        self.test_content = "Test memory content"
        self.test_metadata = {"type": "test", "importance": 0.7}

    def test_add_and_get(self):
        # Test basic add and get functionality
        memory_id = self.store.add(self.test_embedding, self.test_content, self.test_metadata)

        # Test that ID is returned
        self.assertIsNotNone(memory_id)
        self.assertTrue(isinstance(memory_id, str))

        # Test retrieval
        memory = self.store.get(memory_id)

        # Test memory properties
        self.assertEqual(memory.id, memory_id)
        self.assertTrue(np.array_equal(memory.embedding, self.test_embedding))
        self.assertEqual(memory.content, self.test_content)
        self.assertEqual(memory.metadata, self.test_metadata)

    def test_add_with_id(self):
        # Test adding with a specific ID
        custom_id = "custom_memory_id"
        memory_id = self.store.add_with_id(
            custom_id, self.test_embedding, self.test_content, self.test_metadata
        )

        # Test that the custom ID is used
        self.assertEqual(memory_id, custom_id)

        # Test retrieval with the custom ID
        memory = self.store.get(custom_id)
        self.assertEqual(memory.id, custom_id)

    @staticmethod
    def _get_memory_content(memory):
        # Check if content is a dict with 'text' field
        print(f"MEMORY IS: {memory}")
        if isinstance(memory.content, dict) and "text" in memory.content:
            return memory.content["text"]
        return memory.content

    def test_get_all(self):
        # Add multiple memories
        _id1 = self.store.add(self.test_embedding, "Content 1", {"index": 1})
        _id2 = self.store.add(self.test_embedding, "Content 2", {"index": 2})
        _id3 = self.store.add(self.test_embedding, "Content 3", {"index": 3})

        # Get all memories
        memories = self.store.get_all()

        # Test that all memories are returned
        self.assertEqual(len(memories), 3)

        # Test that the memories have the correct content

        memory_contents = [self._get_memory_content(m) for m in memories]

        self.assertIn("Content 1", memory_contents)
        self.assertIn("Content 2", memory_contents)
        self.assertIn("Content 3", memory_contents)

        # Test that the memories have the correct metadata
        memory_indices = [m.metadata.get("index") for m in memories]
        self.assertIn(1, memory_indices)
        self.assertIn(2, memory_indices)
        self.assertIn(3, memory_indices)

    def test_update_metadata(self):
        # Add a memory
        memory_id = self.store.add(self.test_embedding, self.test_content, self.test_metadata)

        # Update metadata
        new_metadata = {"type": "updated", "importance": 0.9, "new_field": "new_value"}
        self.store.update_metadata(memory_id, new_metadata)

        # Get the memory and check the metadata
        memory = self.store.get(memory_id)

        # The metadata should be updated
        self.assertEqual(memory.metadata["type"], "updated")
        self.assertEqual(memory.metadata["importance"], 0.9)
        self.assertEqual(memory.metadata["new_field"], "new_value")

    def test_remove(self):
        # Add a memory
        memory_id = self.store.add(self.test_embedding, self.test_content, self.test_metadata)

        # Verify it was added
        memory = self.store.get(memory_id)
        self.assertEqual(memory.content, self.test_content)

        # Remove the memory
        self.store.remove(memory_id)

        # Verify it was removed
        with self.assertRaises(KeyError):
            self.store.get(memory_id)

    def test_clear(self):
        # Add multiple memories
        self.store.add(self.test_embedding, "Content 1", {"index": 1})
        self.store.add(self.test_embedding, "Content 2", {"index": 2})

        # Verify they were added
        memories = self.store.get_all()
        self.assertEqual(len(memories), 2)

        # Clear the store
        self.store.clear()

        # Verify it's empty
        memories = self.store.get_all()
        self.assertEqual(len(memories), 0)

    def test_consolidate(self):
        # Add multiple memories with different activation levels
        self.store.add(self.test_embedding, "Content 1", {"importance": 0.1})
        self.store.add(self.test_embedding, "Content 2", {"importance": 0.5})
        self.store.add(self.test_embedding, "Content 3", {"importance": 0.9})

        # Set different activation levels
        memories = self.store.get_all()
        for memory in memories:
            # Update activation based on importance
            self.store._metadata[memory.id].activation = memory.metadata.get("importance", 0)

        # Consolidate to keep only 2 memories
        removed_ids = self.store.consolidate(2)

        # Should have removed 1 memory (the one with lowest activation)
        self.assertEqual(len(removed_ids), 1)

        # Verify we have 2 memories left
        memories = self.store.get_all()
        self.assertEqual(len(memories), 2)

        # The remaining memories should be the ones with higher activation
        importances = [m.metadata.get("importance") for m in memories]
        self.assertIn(0.5, importances)
        self.assertIn(0.9, importances)

    def test_get_nonexistent_memory(self):
        # Try to get a memory that doesn't exist
        with self.assertRaises(KeyError):
            self.store.get("nonexistent_id")

    def test_remove_nonexistent_memory(self):
        # Try to remove a memory that doesn't exist
        with self.assertRaises(KeyError):
            self.store.remove("nonexistent_id")

    def test_id_generation(self):
        # Test that unique IDs are generated
        id1 = self.store.add(self.test_embedding, "Content 1")
        id2 = self.store.add(self.test_embedding, "Content 2")
        id3 = self.store.add(self.test_embedding, "Content 3")

        # IDs should be unique
        self.assertNotEqual(id1, id2)
        self.assertNotEqual(id1, id3)
        self.assertNotEqual(id2, id3)

    def test_content_types(self):
        # Test different content types

        # String content
        str_id = self.store.add(self.test_embedding, "String content")
        str_memory = self.store.get(str_id)
        self.assertEqual(str_memory.content, "String content")

        # Dict content
        dict_content = {"text": "Dict content", "additional": "field"}
        dict_id = self.store.add(self.test_embedding, dict_content)
        dict_memory = self.store.get(dict_id)
        self.assertEqual(dict_memory.content, dict_content)

        # List content
        list_content = ["item1", "item2", "item3"]
        list_id = self.store.add(self.test_embedding, list_content)
        list_memory = self.store.get(list_id)
        self.assertEqual(list_memory.content, list_content)

    def test_metadata_none(self):
        # Test adding memory with no metadata
        memory_id = self.store.add(self.test_embedding, self.test_content)
        memory = self.store.get(memory_id)

        # Metadata should be empty but not None
        self.assertIsNotNone(memory.metadata)
        self.assertEqual(memory.metadata, {})

    def test_update_activation(self):
        # Add a memory
        memory_id = self.store.add(self.test_embedding, self.test_content)

        # Update activation
        self.store.update_activation(memory_id, 0.5)

        # Verify activation was updated
        self.assertEqual(self.store._metadata[memory_id].activation, 0.5)

        # Update again
        self.store.update_activation(memory_id, 0.3)

        # Verify activation is cumulative
        self.assertEqual(self.store._metadata[memory_id].activation, 0.8)

    def test_update_activation_nonexistent(self):
        # Try to update activation for a memory that doesn't exist
        with self.assertRaises(KeyError):
            self.store.update_activation("nonexistent_id", 0.5)

    def test_add_multiple(self):
        # Create memories to add
        memory1 = Memory(
            id="1", embedding=np.array([0.1, 0.2, 0.3]), content="Content 1", metadata={"index": 1}
        )
        memory2 = Memory(
            id="2", embedding=np.array([0.4, 0.5, 0.6]), content="Content 2", metadata={"index": 2}
        )
        memory3 = Memory(
            id=None, embedding=np.array([0.7, 0.8, 0.9]), content="Content 3", metadata={"index": 3}
        )

        # Add multiple memories
        memory_ids = self.store.add_multiple([memory1, memory2, memory3])

        # Should have 3 memory IDs
        self.assertEqual(len(memory_ids), 3)

        # First two IDs should match the memory IDs
        self.assertEqual(memory_ids[0], "1")
        self.assertEqual(memory_ids[1], "2")

        # Get all memories
        memories = self.store.get_all()

        # Should have 3 memories
        self.assertEqual(len(memories), 3)

        # Verify contents
        memory_contents = [m.content for m in memories]
        self.assertIn("Content 1", memory_contents)
        self.assertIn("Content 2", memory_contents)
        self.assertIn("Content 3", memory_contents)

    def test_resolve_id(self):
        # Test resolving different ID types

        # Integer ID
        int_id = 42
        resolved_int_id = self.store._resolve_id(int_id)
        self.assertEqual(resolved_int_id, "42")

        # String ID
        str_id = "memory_id"
        resolved_str_id = self.store._resolve_id(str_id)
        self.assertEqual(resolved_str_id, "memory_id")

        # String of integer ID
        str_int_id = "123"
        resolved_str_int_id = self.store._resolve_id(str_int_id)
        self.assertEqual(resolved_str_int_id, "123")


if __name__ == "__main__":
    unittest.main()
