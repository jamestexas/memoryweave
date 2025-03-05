"""
Integration tests for the DynamicContextAdapter.

Tests the DynamicContextAdapter in integration with other components,
particularly the ContextualFabricStrategy.
"""

import numpy as np
import pytest

from memoryweave.components.activation import ActivationManager
from memoryweave.components.associative_linking import AssociativeMemoryLinker
from memoryweave.components.dynamic_context_adapter import DynamicContextAdapter
from memoryweave.components.query_analysis import QueryAnalyzer
from memoryweave.components.retrieval_strategies.contextual_fabric_strategy import (
    ContextualFabricStrategy,
)
from memoryweave.components.temporal_context import TemporalContextBuilder
from memoryweave.storage.refactored.adapter import MemoryAdapter
from memoryweave.storage.refactored.memory_store import StandardMemoryStore


class TestDynamicContextIntegration:
    """Integration tests for DynamicContextAdapter."""

    def setup_method(self):
        """Set up test environment."""
        # Create components
        self.memory_store = StandardMemoryStore()
        self.memory_adapter = MemoryAdapter(self.memory_store)
        self.associative_linker = AssociativeMemoryLinker(self.memory_store)
        self.temporal_context = TemporalContextBuilder(self.memory_store)
        self.activation_manager = ActivationManager(
            memory_store=self.memory_store, associative_linker=self.associative_linker
        )

        self.query_analyzer = QueryAnalyzer()
        self.dynamic_adapter = DynamicContextAdapter()
        self.retrieval_strategy = ContextualFabricStrategy(
            memory_store=self.memory_store,
            associative_linker=self.associative_linker,
            temporal_context=self.temporal_context,
            activation_manager=self.activation_manager,
        )

        # Initialize components
        self.query_analyzer.initialize({})
        self.dynamic_adapter.initialize(
            {
                "adaptation_strength": 1.0,
                "enable_memory_size_adaptation": True,
            }
        )
        self.retrieval_strategy.initialize(
            {
                "confidence_threshold": 0.1,
                "similarity_weight": 0.5,
                "associative_weight": 0.3,
                "temporal_weight": 0.1,
                "activation_weight": 0.1,
            }
        )

        # Add test memories
        self._add_test_memories()

    def _add_test_memories(self):
        """Add test memories to the system."""
        # Add a set of test memories with different topics
        memories = [
            {
                "content": "Python is known for its readability and simple syntax.",
                "embedding": self._make_embedding(["python", "readable", "syntax"]),
                "metadata": {"type": "factual", "topic": "programming", "created_at": 1625000000},
            },
            {
                "content": "Paris is the capital of France.",
                "embedding": self._make_embedding(["paris", "capital", "france"]),
                "metadata": {"type": "factual", "topic": "geography", "created_at": 1625100000},
            },
            {
                "content": "I went to the beach yesterday, it was sunny.",
                "embedding": self._make_embedding(["beach", "yesterday", "sunny"]),
                "metadata": {"type": "personal", "topic": "activities", "created_at": 1625200000},
            },
            {
                "content": "My favorite food is pizza with extra cheese.",
                "embedding": self._make_embedding(["favorite", "food", "pizza", "cheese"]),
                "metadata": {"type": "personal", "topic": "preferences", "created_at": 1625300000},
            },
            {
                "content": "The sky is blue because of Rayleigh scattering.",
                "embedding": self._make_embedding(["sky", "blue", "rayleigh", "scattering"]),
                "metadata": {"type": "factual", "topic": "science", "created_at": 1625400000},
            },
        ]

        # Add to memory store
        for memory in memories:
            self.memory_store.add(
                embedding=memory["embedding"],
                content=memory["content"],
                metadata=memory["metadata"],
            )

    def _make_embedding(self, keywords: list[str]) -> np.ndarray:
        """Create a simple test embedding from keywords."""
        # This is a very simple embedding for testing
        # Real embeddings would be created with a proper embedding model
        embedding = np.zeros(768)

        # Set specific dimensions based on keywords
        for _i, word in enumerate(keywords):
            # Use hash of word to deterministically set some dimensions
            word_hash = hash(word) % 768
            embedding[word_hash] = 1.0

            # Set a few neighboring dimensions too
            for j in range(1, 4):
                idx = (word_hash + j) % 768
                embedding[idx] = 0.9 - (j * 0.2)

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def test_pipeline_integration(self):
        """Test that the adapter integrates properly in the pipeline."""
        # Process a query through the pipeline
        query = "What is my favorite food?"

        # Step 1: Query analysis
        analysis_context = self.query_analyzer.process_query(query, {})

        # Verify query analysis produced expected output
        assert "primary_query_type" in analysis_context

        # Step 2: Dynamic context adaptation
        adaptation_context = self.dynamic_adapter.process_query(
            query,
            {
                **analysis_context,
                "memory_store": self.memory_store,
                "associative_linker": self.associative_linker,
                "temporal_context": self.temporal_context,
                "activation_manager": self.activation_manager,
            },
        )

        # Verify adapter produced parameters
        assert "adapted_retrieval_params" in adaptation_context

        # Step 3: Retrieval with adapted parameters
        combined_context = {
            **analysis_context,
            **adaptation_context,
            "query": query,
            "query_embedding": self._make_embedding(["favorite", "food"]),
            "memory_store": self.memory_store,
        }

        # Directly access the strategy's retrieve method
        results = self.retrieval_strategy.retrieve(
            combined_context["query_embedding"], top_k=5, context=combined_context
        )

        # Verify retrieval returned results
        assert len(results) > 0

        # The first result should be the favorite food memory
        assert "pizza" in results[0].get("content", "").lower()

    def test_large_memory_adaptations(self):
        """Test adaptations for large memory stores."""
        # Create a larger memory store
        large_memory_store = StandardMemoryStore()
        MemoryAdapter(large_memory_store)

        # Add a large number of memories
        for i in range(100):
            embedding = np.random.rand(768)
            embedding = embedding / np.linalg.norm(embedding)
            large_memory_store.add(
                embedding=embedding,
                content=f"Test memory {i}",
                metadata={"type": "test", "created_at": 1625000000 + i * 1000},
            )

        # Set up larger components
        large_strategy = ContextualFabricStrategy(memory_store=large_memory_store)
        large_strategy.initialize(
            {
                "confidence_threshold": 0.1,
                "similarity_weight": 0.5,
                "associative_weight": 0.3,
                "temporal_weight": 0.1,
                "activation_weight": 0.1,
                "debug": True,
            }
        )

        # Process a query
        query = "test query"
        query_embedding = np.random.rand(768)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # First without adaptation
        regular_context = {"query": query, "memory_store": large_memory_store}

        regular_results = large_strategy.retrieve(query_embedding, top_k=5, context=regular_context)

        # Then with dynamic adaptation
        adapted_context = {
            "query": query,
            "memory_store": large_memory_store,
            "adapted_retrieval_params": self.dynamic_adapter.process_query(
                query, {"memory_store": large_memory_store}
            )["adapted_retrieval_params"],
        }

        adapted_results = large_strategy.retrieve(query_embedding, top_k=5, context=adapted_context)

        # Verify both returned results
        assert len(regular_results) > 0
        assert len(adapted_results) > 0

        # Note: We can't easily assert on the quality of results in this test,
        # as we're using random embeddings, but we can verify the system doesn't break

    def test_adaptation_affects_weights(self):
        """Test that adaptation actually changes weights in the strategy."""
        # Save original weights
        original_temporal = self.retrieval_strategy.temporal_weight

        # Process temporal query
        temporal_query = "What did I do yesterday?"

        # Analyze query
        analysis_context = self.query_analyzer.process_query(temporal_query, {})

        # Force temporal type if not detected
        analysis_context["primary_query_type"] = "temporal"
        analysis_context["has_temporal_reference"] = True

        # Dynamic adaptation
        adaptation_context = self.dynamic_adapter.process_query(
            temporal_query, {**analysis_context, "memory_store": self.memory_store}
        )

        # Apply to strategy (this would happen in retrieve())
        adapted_params = adaptation_context["adapted_retrieval_params"]

        # Manually apply parameters to strategy to check they update
        for param_name, param_value in adapted_params.items():
            if hasattr(self.retrieval_strategy, param_name):
                setattr(self.retrieval_strategy, param_name, param_value)

        # Verify weights were updated
        assert self.retrieval_strategy.temporal_weight > original_temporal, (
            "Temporal weight should increase for temporal queries"
        )

        # Run retrieval with adapted strategy
        results = self.retrieval_strategy.retrieve(
            self._make_embedding(["yesterday"]),
            top_k=5,
            context={"query": temporal_query, "memory_store": self.memory_store},
        )

        # Verify beach memory is returned (contains "yesterday")
        found_beach = False
        for result in results:
            if "beach" in result.get("content", "").lower():
                found_beach = True
                break

        assert found_beach, "Should find 'beach' memory with temporal query 'yesterday'"


if __name__ == "__main__":
    pytest.main()
