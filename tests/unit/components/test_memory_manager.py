"""
Unit tests for the MemoryManager component.
"""

import unittest

from pydantic import Field

from memoryweave.components.base import Component
from memoryweave.components.memory_manager import MemoryManager


class MockComponent(Component):
    """Mock component for testing."""

    initialize_called: bool = False
    process_query_called: bool = False
    process_query_result: dict = Field(default_factory=dict)
    config: dict = Field(default_factory=dict)
    initialize_called: bool = False
    last_query: str | None = None
    last_context: dict | None = None

    def initialize(self, config):
        """Initialize the component."""
        self.initialize_called = True
        self.config = config

    def process_query(self, query, context):
        """Process a query."""
        self.process_query_called = True
        self.last_query = query
        self.last_context = context
        return self.process_query_result


class MemoryManagerTest(unittest.TestCase):
    """
    Unit tests for the MemoryManager component.
    """

    def setUp(self):
        """Set up test environment before each test."""
        self.memory_manager = MemoryManager()

        # Create mock components
        self.component1 = MockComponent()
        self.component2 = MockComponent()
        self.component3 = MockComponent()

        # Set up mock return values
        self.component1.process_query_result = {"result1": "value1"}
        self.component2.process_query_result = {"result2": "value2"}
        self.component3.process_query_result = {"result3": "value3"}

        # Register components
        self.memory_manager.register_component("component1", self.component1)
        self.memory_manager.register_component("component2", self.component2)
        self.memory_manager.register_component("component3", self.component3)

    def test_register_component(self):
        """Test registering components."""
        # Check that components were registered
        self.assertIn("component1", self.memory_manager.components)
        self.assertIn("component2", self.memory_manager.components)
        self.assertIn("component3", self.memory_manager.components)

        # Check that the correct components were registered
        self.assertEqual(self.memory_manager.components["component1"], self.component1)
        self.assertEqual(self.memory_manager.components["component2"], self.component2)
        self.assertEqual(self.memory_manager.components["component3"], self.component3)

    def test_build_pipeline(self):
        """Test building a pipeline."""
        # Build a pipeline
        pipeline_config = [
            {"component": "component1", "config": {"param1": "value1"}},
            {"component": "component2"},
            {"component": "component3", "config": {"param3": "value3"}},
        ]
        self.memory_manager.build_pipeline(pipeline_config)

        # Check that the pipeline was built correctly
        self.assertEqual(len(self.memory_manager.pipeline), 3)
        self.assertEqual(self.memory_manager.pipeline[0]["component"], self.component1)
        self.assertEqual(self.memory_manager.pipeline[1]["component"], self.component2)
        self.assertEqual(self.memory_manager.pipeline[2]["component"], self.component3)

    def test_build_pipeline_invalid_component(self):
        """Test building a pipeline with an invalid component."""
        # Build a pipeline with an invalid component
        pipeline_config = [{"component": "invalid_component"}]

        # Check that an exception is raised
        with self.assertRaises(ValueError):
            self.memory_manager.build_pipeline(pipeline_config)

    def test_execute_pipeline(self):
        """Test executing a pipeline."""
        # Build a pipeline
        pipeline_config = [
            {"component": "component1"},
            {"component": "component2"},
            {"component": "component3"},
        ]
        self.memory_manager.build_pipeline(pipeline_config)

        # Execute the pipeline
        query = "Test query"
        context = {"param": "value"}
        result = self.memory_manager.execute_pipeline(query, context)

        # Check that all components were called
        self.assertTrue(self.component1.process_query_called)
        self.assertTrue(self.component2.process_query_called)
        self.assertTrue(self.component3.process_query_called)

        # Check that the query was passed correctly
        self.assertEqual(self.component1.last_query, query)
        self.assertEqual(self.component2.last_query, query)
        self.assertEqual(self.component3.last_query, query)

        # Check that the context was updated correctly
        self.assertIn("result1", result)
        self.assertIn("result2", result)
        self.assertIn("result3", result)
        self.assertEqual(result["result1"], "value1")
        self.assertEqual(result["result2"], "value2")
        self.assertEqual(result["result3"], "value3")


if __name__ == "__main__":
    unittest.main()
