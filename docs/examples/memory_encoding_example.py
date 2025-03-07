"""
Example demonstrating the use of the MemoryEncoder component.

This example shows how to use the MemoryEncoder component to create
memory embeddings for different types of content.
"""

from sentence_transformers import SentenceTransformer

from memoryweave.components.memory_encoding import MemoryEncoder
from memoryweave.factory.memory_factory import create_memory_encoder

# Option 1: Create the encoder manually
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
encoder = MemoryEncoder(embedding_model)
encoder.initialize(
    {
        "context_window_size": 3,
        "use_episodic_markers": True,
    }
)

# Option 2: Use the factory method (recommended)
encoder = create_memory_encoder(
    "sentence-transformers/paraphrase-MiniLM-L6-v2",
    context_window_size=3,
    use_episodic_markers=True,
)

# Example 1: Encode a simple text
text = "MemoryWeave is a memory management system for LLMs with a component-based architecture."
text_embedding = encoder.encode_text(text)
print(f"Text embedding shape: {text_embedding.shape}")

# Example 2: Encode a query-response interaction
query = "What is MemoryWeave?"
response = "MemoryWeave is a memory management system for LLMs."
metadata = {"type": "qa", "importance": 0.8}
interaction_embedding = encoder.encode_interaction(query, response, metadata)
print(f"Interaction embedding shape: {interaction_embedding.shape}")

# Example 3: Encode a concept
concept = "MemoryWeave"
definition = "A memory management system for LLMs with a component-based architecture."
examples = ["Memory retrieval", "Memory storage", "Memory encoding"]
concept_embedding = encoder.encode_concept(concept, definition, examples)
print(f"Concept embedding shape: {concept_embedding.shape}")

# Example 4: Use the process method directly for pipeline integration
data = {
    "type": "interaction",
    "query": "How does MemoryWeave work?",
    "response": "MemoryWeave uses a component-based architecture for memory management.",
    "metadata": {"importance": 0.9, "topics": ["architecture"]},
}
context = {"conversation_history": [], "current_time": 1234567890}
result = encoder.process(data, context)
print(f"Process result: {result.keys()}")
