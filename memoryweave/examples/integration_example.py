"""
Example demonstrating integration with LLM inference.
"""

import torch
from transformers import AutoModel, AutoTokenizer

from memoryweave.core.contextual_fabric import ContextualMemory
from memoryweave.core.memory_encoding import MemoryEncoder
from memoryweave.core.retrieval import ContextualRetriever


# Simple embedding model class wrapping a transformer
class EmbeddingModel:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use mean pooling for sentence embedding
        attention_mask = inputs["attention_mask"]
        embeddings = outputs.last_hidden_state

        # Mask padding tokens
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask

        # Sum and average
        summed = torch.sum(masked_embeddings, dim=1)
        counts = torch.sum(mask, dim=1)
        mean_pooled = summed / counts

        return mean_pooled.numpy()[0]


# Example LLM inference with MemoryWeave integration
def infer_with_memory(prompt, conversation_history, embedding_model, memory_system):
    """Simulate LLM inference with memory augmentation."""

    # Retrieve relevant memories
    retriever = memory_system["retriever"]
    memory_entries = retriever.retrieve_for_context(prompt, conversation_history, top_k=3)

    # Augment prompt with memory context
    augmented_prompt = prompt
    if memory_entries:
        augmented_prompt = "Context from previous conversation:\n"
        for entry in memory_entries:
            if entry.get("type") == "interaction":
                augmented_prompt += f"- {entry.get('speaker')}: {entry.get('content')}\n"
                if entry.get("response"):
                    augmented_prompt += f"  Response: {entry.get('response')}\n"
            elif entry.get("type") == "concept":
                augmented_prompt += f"- Concept '{entry.get('name')}': {entry.get('description')}\n"

        augmented_prompt += f"\nCurrent input: {prompt}"

    # This would be your actual LLM call in a real implementation
    # For this example, we'll simulate a response
    response = f"This is a simulated response to: {augmented_prompt}"

    # Store the new interaction in memory
    encoder = memory_system["encoder"]
    embedding, metadata = encoder.encode_interaction(
        message=prompt, speaker="user", response=response
    )

    memory = memory_system["memory"]
    memory.add_memory(embedding, prompt, metadata)

    return response


# Example usage
def main():
    # Initialize components
    embedding_model = EmbeddingModel()
    memory = ContextualMemory(embedding_dim=768, max_memories=100)
    encoder = MemoryEncoder(embedding_model)
    retriever = ContextualRetriever(memory, embedding_model)

    memory_system = {"memory": memory, "encoder": encoder, "retriever": retriever}

    # Simulate a conversation
    conversation_history = []

    # First turn
    prompt = "What's the best way to implement a binary search tree?"
    response = infer_with_memory(prompt, conversation_history, embedding_model, memory_system)
    conversation_history.append({"speaker": "user", "message": prompt, "response": response})
    print(f"User: {prompt}")
    print(f"AI: {response}\n")

    # Second turn
    prompt = "How would I balance that tree?"
    response = infer_with_memory(prompt, conversation_history, embedding_model, memory_system)
    conversation_history.append({"speaker": "user", "message": prompt, "response": response})
    print(f"User: {prompt}")
    print(f"AI: {response}\n")

    # Third turn (unrelated)
    prompt = "What's a good recipe for banana bread?"
    response = infer_with_memory(prompt, conversation_history, embedding_model, memory_system)
    conversation_history.append({"speaker": "user", "message": prompt, "response": response})
    print(f"User: {prompt}")
    print(f"AI: {response}\n")

    # Fourth turn (back to trees)
    prompt = "What's the time complexity for searching in that tree we discussed?"
    response = infer_with_memory(prompt, conversation_history, embedding_model, memory_system)
    conversation_history.append({"speaker": "user", "message": prompt, "response": response})
    print(f"User: {prompt}")
    print(f"AI: {response}\n")


if __name__ == "__main__":
    main()
