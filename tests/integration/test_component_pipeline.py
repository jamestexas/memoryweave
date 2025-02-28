"""
Integration tests for the MemoryWeave component pipeline.
"""

import unittest
import numpy as np

from memoryweave.components.memory_manager import MemoryManager
from memoryweave.components.personal_attributes import PersonalAttributeManager
from memoryweave.components.query_analysis import QueryAnalyzer
from memoryweave.components.retrieval_strategies import (
    SimilarityRetrievalStrategy,
    TemporalRetrievalStrategy,
    HybridRetrievalStrategy
)
from memoryweave.components.post_processors import (
    KeywordBoostProcessor,
    SemanticCoherenceProcessor,
    AdaptiveKProcessor
)
from tests.utils.mock_models import MockEmbeddingModel, MockMemory


class ComponentPipelineTest(unittest.TestCase):
    """
    Integration tests for the MemoryWeave component pipeline.
    
    These tests verify that the component pipeline works correctly with
    different configurations.
    """
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create embedding model
        self.embedding_model = MockEmbeddingModel(embedding_dim=768)
        
        # Create memory
        self.memory = MockMemory(embedding_dim=768)
        
        # Populate memory with test data
        self._populate_test_memory()
        
        # Create memory manager
        self.memory_manager = MemoryManager()
        
        # Create components
        self.query_analyzer = QueryAnalyzer()
        self.personal_attributes = PersonalAttributeManager()
        self.similarity_strategy = SimilarityRetrievalStrategy(self.memory)
        self.temporal_strategy = TemporalRetrievalStrategy(self.memory)
        self.hybrid_strategy = HybridRetrievalStrategy(self.memory)
        self.keyword_boost = KeywordBoostProcessor()
        self.coherence_processor = SemanticCoherenceProcessor()
        self.adaptive_k = AdaptiveKProcessor()
        
        # Register components
        self.memory_manager.register_component("query_analyzer", self.query_analyzer)
        self.memory_manager.register_component("personal_attributes", self.personal_attributes)
        self.memory_manager.register_component("similarity_retrieval", self.similarity_strategy)
        self.memory_manager.register_component("temporal_retrieval", self.temporal_strategy)
        self.memory_manager.register_component("hybrid_retrieval", self.hybrid_strategy)
        self.memory_manager.register_component("keyword_boost", self.keyword_boost)
        self.memory_manager.register_component("coherence_check", self.coherence_processor)
        self.memory_manager.register_component("adaptive_k", self.adaptive_k)
        
        # Initialize components
        self.similarity_strategy.initialize({"confidence_threshold": 0.0})
        self.hybrid_strategy.initialize({
            "relevance_weight": 0.7,
            "recency_weight": 0.3,
            "confidence_threshold": 0.0
        })
        self.keyword_boost.initialize({"keyword_boost_weight": 0.5})
        self.coherence_processor.initialize({"coherence_threshold": 0.2})
        self.adaptive_k.initialize({"adaptive_k_factor": 0.3})
    
    def _populate_test_memory(self):
        """Populate memory with test data."""
        # Add personal information
        self._add_personal_memory("My favorite color is blue", 
                                 {"type": "interaction", "speaker": "user"})
        self._add_personal_memory("I live in Seattle", 
                                 {"type": "interaction", "speaker": "user"})
        self._add_personal_memory("I work as a software engineer", 
                                 {"type": "interaction", "speaker": "user"})
        
        # Add factual information
        self._add_factual_memory("Python", 
                                "A high-level programming language known for readability", 
                                ["Programming", "Scripting"])
        self._add_factual_memory("Machine Learning", 
                                "A field of AI that enables systems to learn from data", 
                                ["AI", "Data Science"])
    
    def _add_personal_memory(self, content, metadata):
        """Add a personal memory."""
        embedding = self.embedding_model.encode(content)
        self.memory.add_memory(embedding, content, metadata)
    
    def _add_factual_memory(self, concept, description, related_concepts):
        """Add a factual memory."""
        embedding = self.embedding_model.encode(description)
        metadata = {
            "type": "concept",
            "name": concept,
            "description": description,
            "related": related_concepts
        }
        self.memory.add_memory(embedding, description, metadata)
    
    def test_similarity_pipeline(self):
        """Test pipeline with similarity retrieval strategy."""
        # Build pipeline
        pipeline_config = [
            {"component": "query_analyzer"},
            {"component": "similarity_retrieval"},
            {"component": "keyword_boost"}
        ]
        self.memory_manager.build_pipeline(pipeline_config)
        
        # Create query and context
        query = "Tell me about programming"
        query_embedding = self.embedding_model.encode(query)
        context = {
            "query_embedding": query_embedding,
            "memory": self.memory,
            "top_k": 3
        }
        
        # Execute pipeline
        result_context = self.memory_manager.execute_pipeline(query, context)
        
        # Verify results
        self.assertIn("results", result_context)
        self.assertGreater(len(result_context["results"]), 0)
        
        # Verify query analysis results
        self.assertIn("primary_query_type", result_context)
        self.assertIn("important_keywords", result_context)
    
    def test_hybrid_pipeline_with_processors(self):
        """Test pipeline with hybrid retrieval and all processors."""
        # Build pipeline
        pipeline_config = [
            {"component": "query_analyzer"},
            {"component": "personal_attributes"},
            {"component": "hybrid_retrieval"},
            {"component": "keyword_boost"},
            {"component": "coherence_check"},
            {"component": "adaptive_k"}
        ]
        self.memory_manager.build_pipeline(pipeline_config)
        
        # Create query and context
        query = "What's my favorite color?"
        query_embedding = self.embedding_model.encode(query)
        context = {
            "query_embedding": query_embedding,
            "memory": self.memory,
            "top_k": 3
        }
        
        # Execute pipeline
        result_context = self.memory_manager.execute_pipeline(query, context)
        
        # Verify results
        self.assertIn("results", result_context)
        self.assertGreater(len(result_context["results"]), 0)
        
        # Verify personal attributes
        self.assertIn("personal_attributes", result_context)
        
        # Verify that color information is retrieved
        color_found = False
        for result in result_context["results"]:
            if "blue" in str(result.get("content", "")).lower():
                color_found = True
                break
        
        self.assertTrue(color_found, "Failed to retrieve color information")
    
    def test_temporal_pipeline(self):
        """Test pipeline with temporal retrieval strategy."""
        # Build pipeline
        pipeline_config = [
            {"component": "query_analyzer"},
            {"component": "temporal_retrieval"}
        ]
        self.memory_manager.build_pipeline(pipeline_config)
        
        # Create query and context
        query = "What did we talk about recently?"
        query_embedding = self.embedding_model.encode(query)
        context = {
            "query_embedding": query_embedding,
            "memory": self.memory,
            "top_k": 3
        }
        
        # Execute pipeline
        result_context = self.memory_manager.execute_pipeline(query, context)
        
        # Verify results
        self.assertIn("results", result_context)
        self.assertGreater(len(result_context["results"]), 0)
        
        # Verify that results are ordered by recency
        if len(result_context["results"]) >= 2:
            first_memory_id = result_context["results"][0]["memory_id"]
            second_memory_id = result_context["results"][1]["memory_id"]
            self.assertGreater(first_memory_id, second_memory_id, 
                              "Memories not ordered by recency")


if __name__ == "__main__":
    unittest.main()
