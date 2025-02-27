import torch
from rich.console import Console
from transformers import AutoModel, AutoTokenizer

from memoryweave.core import ContextualMemory, ContextualRetriever, MemoryEncoder

c = Console()


# Helper class for sentence embedding
class EmbeddingModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def encode(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Mean pooling
        attention_mask = inputs["attention_mask"]
        embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / counts

        return mean_pooled.numpy()[0]


def main():
    c.log("Testing MemoryWeave core functionality...")

    try:
        # Load a small sentence transformer for embeddings
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        c.log(f"Loading embedding model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        embedding_model = EmbeddingModelWrapper(model, tokenizer)
        c.log("Embedding model loaded successfully!")

        # Initialize memory components
        embedding_dim = model.config.hidden_size  # Usually 384 for MiniLM-L6
        memory = ContextualMemory(embedding_dim=embedding_dim)
        encoder = MemoryEncoder(embedding_model)
        retriever = ContextualRetriever(memory, embedding_model)

        c.log("\nTesting memory encoding and storage...")
        # Add a memory
        test_concept = "Neural Networks"
        test_description = (
            "A computational model inspired by the human brain that can learn patterns from data."
        )
        embedding, metadata = encoder.encode_concept(
            concept=test_concept,
            description=test_description,
            related_concepts=["Deep Learning", "AI", "Machine Learning"],
        )
        memory.add_memory(embedding, test_description, metadata)
        c.log(f"Added concept: {test_concept}")

        # Test retrieval
        c.log("\nTesting memory retrieval...")
        query = "Tell me about artificial neural networks and how they work."
        results = retriever.retrieve_for_context(query, [])

        c.log(f"Query: {query}")
        c.log(f"Retrieved {len(results)} memories:")
        for result in results:
            # ContextualRetriever.retrieve_for_context always returns a list of dictionaries
            c.log(
                f"  Score: {result.get('relevance_score', 0):.4f}, "
                f"Content: {result.get('text', result.get('name', 'unknown'))[:50]}..."
            )

        # Test conversation context
        c.log("\nTesting conversational memory...")
        conversation = []

        # Turn 1
        query = "What are neural networks used for?"
        c.log(f"\nUser: {query}")

        # Simulate response
        response = "Neural networks are used for various tasks like image recognition, language processing, and prediction."

        # Store in memory
        embedding, metadata = encoder.encode_interaction(
            message=query, speaker="user", response=response
        )
        memory.add_memory(embedding, query, metadata)

        # Update conversation history
        conversation.append({"speaker": "user", "message": query, "response": response})

        c.log(f"Assistant: {response}")

        # Turn 2
        query = "Are they difficult to build?"
        c.log(f"\nUser: {query}")

        # Test retrieval with conversation context
        results = retriever.retrieve_for_context(query, conversation)

        c.log(f"Retrieved {len(results)} memories:")
        for result in results:
            c.log(
                f"  Score: {result.get('relevance_score', 0):.4f}, "
                f"Type: {result.get('type', 'unknown')}, "
                f"Content: {result.get('text', result.get('content', 'unknown'))[:50]}..."
            )

        c.log("\nAll tests completed successfully!")

    except Exception as e:
        c.log(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
