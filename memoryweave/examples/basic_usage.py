"""
Basic usage example of the MemoryWeave system.
"""

import numpy as np

from memoryweave.core import ContextualMemory, ContextualRetriever, MemoryEncoder
from memoryweave.evaluation import evaluate_conversation


# Simple mock embedding model for demonstration
class MockEmbeddingModel:
    def encode(self, text):
        """Create a simple deterministic embedding based on text content."""
        # This is just for demo purposes - use a real embedding model in practice
        hash_val = hash(text) % 10000
        np.random.seed(hash_val)
        return np.random.rand(768)


def main():
    print("Initializing MemoryWeave system...")

    # Initialize components
    embedding_model = MockEmbeddingModel()
    memory = ContextualMemory(embedding_dim=768, max_memories=100)
    encoder = MemoryEncoder(embedding_model)
    retriever = ContextualRetriever(memory, embedding_model)

    memory_system = {"memory": memory, "encoder": encoder, "retriever": retriever}

    # Add some initial concepts
    print("\nAdding initial knowledge...")
    concept_data = [
        {
            "concept": "Python",
            "description": "A high-level programming language known for its readability and versatility.",
            "related": ["Programming", "Scripting", "Data Science"],
        },
        {
            "concept": "Machine Learning",
            "description": "A field of AI that enables systems to learn from data without explicit programming.",
            "related": ["AI", "Data Science", "Neural Networks"],
        },
        {
            "concept": "Memory Management",
            "description": "Techniques for efficiently allocating, using and freeing computer memory.",
            "related": ["Computing", "Programming", "Performance"],
        },
    ]

    for data in concept_data:
        embedding, metadata = encoder.encode_concept(
            concept=data["concept"],
            description=data["description"],
            related_concepts=data["related"],
        )
        memory.add_memory(embedding, data["description"], metadata)
        print(f"  Added concept: {data['concept']}")

    # Simulate a conversation with memory
    print("\nSimulating conversation with memory augmentation...")
    conversation = []

    # Helper function to simulate LLM response
    def simulate_response(query, retrieved_memories):
        """Simple mock LLM that uses retrieved memories."""
        response = "This is a response about"

        if retrieved_memories:
            topics = []
            for mem in retrieved_memories:
                if mem.get("type") == "concept":
                    topics.append(mem.get("name"))
                elif mem.get("content"):
                    # Extract some keywords (simplified)
                    words = mem.get("content").split()
                    if words:
                        topics.append(words[0])

            if topics:
                response += f" {', '.join(topics)}"

        return response + f" in response to: '{query}'"

    # First turn
    query = "Tell me about Python programming."
    retrieved = retriever.retrieve_for_context(query, conversation)
    response = simulate_response(query, retrieved)

    # Store interaction
    embedding, metadata = encoder.encode_interaction(
        message=query, speaker="user", response=response
    )
    memory.add_memory(embedding, query, metadata)

    # Add to conversation history
    conversation.append({"speaker": "user", "message": query, "response": response})

    print(f"\nUser: {query}")
    print(f"AI: {response}")
    print("  Retrieved memories:")
    for mem in retrieved[:2]:  # Show top 2
        print(f"    - {mem.get('type', 'unknown')}: {mem.get('name', mem.get('content', ''))}")

    # Second turn
    query = "How does it handle memory?"
    retrieved = retriever.retrieve_for_context(query, conversation)
    response = simulate_response(query, retrieved)

    # Store interaction
    embedding, metadata = encoder.encode_interaction(
        message=query, speaker="user", response=response
    )
    memory.add_memory(embedding, query, metadata)

    # Add to conversation history
    conversation.append({"speaker": "user", "message": query, "response": response})

    print(f"\nUser: {query}")
    print(f"AI: {response}")
    print("  Retrieved memories:")
    for mem in retrieved[:2]:  # Show top 2
        print(f"    - {mem.get('type', 'unknown')}: {mem.get('name', mem.get('content', ''))}")

    # Third turn (unrelated topic)
    query = "What do you know about machine learning?"
    retrieved = retriever.retrieve_for_context(query, conversation)
    response = simulate_response(query, retrieved)

    # Store interaction
    embedding, metadata = encoder.encode_interaction(
        message=query, speaker="user", response=response
    )
    memory.add_memory(embedding, query, metadata)

    # Add to conversation history
    conversation.append({"speaker": "user", "message": query, "response": response})

    print(f"\nUser: {query}")
    print(f"AI: {response}")
    print("  Retrieved memories:")
    for mem in retrieved[:2]:  # Show top 2
        print(f"    - {mem.get('type', 'unknown')}: {mem.get('name', mem.get('content', ''))}")

    # Fourth turn (back to previous topic)
    query = "Going back to our discussion about programming, how are variables managed?"
    retrieved = retriever.retrieve_for_context(query, conversation)
    response = simulate_response(query, retrieved)

    # Store interaction
    embedding, metadata = encoder.encode_interaction(
        message=query, speaker="user", response=response
    )
    memory.add_memory(embedding, query, metadata)

    # Add to conversation history
    conversation.append({"speaker": "user", "message": query, "response": response})

    print(f"\nUser: {query}")
    print(f"AI: {response}")
    print("  Retrieved memories:")
    for mem in retrieved[:2]:  # Show top 2
        print(f"    - {mem.get('type', 'unknown')}: {mem.get('name', mem.get('content', ''))}")

    # Evaluate the conversation
    print("\nEvaluating conversation quality...")
    eval_results = evaluate_conversation(conversation, memory_system, embedding_model)

    print(f"  Overall coherence score: {eval_results['overall_coherence']:.2f}")
    print(f"  Average memory relevance: {eval_results['average_relevance']:.2f}")
    print(f"  Average response consistency: {eval_results['average_consistency']:.2f}")


if __name__ == "__main__":
    main()
