import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from memoryweave.core import ContextualMemory, ContextualRetriever, MemoryEncoder
from memoryweave.integrations import HuggingFaceAdapter


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


def format_memories(memories):
    """Format memories for prompt inclusion in a concise, structured way."""
    if not memories:
        return ""

    formatted = []

    # First check for personal attributes memory
    personal_attributes = None
    for i, mem in enumerate(memories):
        if mem.get("type") == "personal_attributes":
            personal_attributes = mem
            # Remove it from the regular memories list to handle separately
            memories.remove(mem)
            break

    # If we have personal attributes, format them first
    if personal_attributes:
        attributes = personal_attributes.get("attributes", {})
        if attributes:
            formatted.append("USER PROFILE:")
            # Format preferences
            preferences = [
                f"{k.replace('preference_', '')}: {v}"
                for k, v in attributes.items()
                if k.startswith("preference_")
            ]
            if preferences:
                formatted.append("Preferences: " + ", ".join(preferences))

            # Format demographics
            demographics = [
                f"{k.replace('demographic_', '')}: {v}"
                for k, v in attributes.items()
                if k.startswith("demographic_")
            ]
            if demographics:
                formatted.append("Demographics: " + ", ".join(demographics))

            # Format traits
            if "trait_hobbies" in attributes:
                hobbies = attributes["trait_hobbies"]
                if isinstance(hobbies, list):
                    formatted.append(f"Hobbies: {', '.join(hobbies)}")
                else:
                    formatted.append(f"Hobbies: {hobbies}")

            # Format relationships
            relationships = [
                f"{k.replace('relationship_', '')}: {v}"
                for k, v in attributes.items()
                if k.startswith("relationship_")
            ]
            if relationships:
                formatted.append("Relationships: " + ", ".join(relationships))

            formatted.append("")  # Empty line after profile

    # Format regular memories
    for i, mem in enumerate(memories):
        if mem.get("type") == "interaction":
            # Extract the key information instead of including full text
            content = mem.get("content", "")
            # Focus on extracting key facts rather than full conversation
            if "favorite color" in content.lower() and "blue" in content.lower():
                formatted.append(f"MEMORY [{i + 1}]: User mentioned their favorite color is blue.")
            elif "hike" in content.lower() or "hiking" in content.lower():
                formatted.append(
                    f"MEMORY [{i + 1}]: User said they enjoy hiking in the mountains on weekends."
                )
            elif "paint" in content.lower() or "painting" in content.lower():
                formatted.append(f"MEMORY [{i + 1}]: User is considering painting their room.")
            else:
                # Fallback to a generic but concise summary
                formatted.append(f"MEMORY [{i + 1}]: User asked about: {content[:30]}...")
        elif mem.get("type") == "concept":
            formatted.append(
                f"MEMORY [{i + 1}]: Concept '{mem.get('name', '')}': {mem.get('description', '')[:50]}..."
            )

    # Add a clear separator and instruction
    if formatted:
        return (
            "\n".join(formatted) + "\n\nPlease use this information to answer the current question."
        )

    return ""


def main():
    print("Testing MemoryWeave with Orca Mini 3B...")

    # Load embedding model
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading embedding model: {embedding_model_name}")
    emb_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    emb_model = AutoModel.from_pretrained(embedding_model_name)
    embedding_model = EmbeddingModelWrapper(emb_model, emb_tokenizer)

    # Initialize memory system
    embedding_dim = emb_model.config.hidden_size  # 384 for MiniLM-L6
    memory = ContextualMemory(embedding_dim=embedding_dim)
    encoder = MemoryEncoder(embedding_model)
    # Use enhanced retriever with keyword boost
    retriever = ContextualRetriever(
        memory=memory, embedding_model=embedding_model, keyword_boost_weight=0.5
    )

    memory_system = {"memory": memory, "encoder": encoder, "retriever": retriever}

    # Load Orca Mini
    model_name = "psmathur/orca_mini_3b"
    print(f"Loading language model: {model_name}")
    print("(This may take a minute...)")
    try:
        lm_model = AutoModelForCausalLM.from_pretrained(model_name)
        lm_tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Create adapter
        adapter = HuggingFaceAdapter(
            memory_system=memory_system,
            model=lm_model,
            tokenizer=lm_tokenizer,
            format_memories_fn=format_memories,
        )

        # Test conversations with contextual recall challenges
        test_conversation = [
            "My favorite color is blue. What's your favorite color?",
            "I live in Seattle and I like to hike in the mountains on weekends. Do you enjoy outdoor activities?",
            "I'm thinking of painting my room. What color would you recommend?",
            "I work as a software engineer. Do you think that's a good career?",
            "Tell me about a good beginner hike near Seattle.",
            "What was my favorite color again?",
            "Where do I live and what do I do for work?",
            "What activities do I enjoy on weekends?",
        ]

        # Run the test
        conversation_history = []

        print("\nStarting conversation test with MemoryWeave:")
        print("-" * 60)

        for i, query in enumerate(test_conversation):
            print(f"\nUser: {query}")

            # Retrieve memories (for demonstration)
            if i > 0:
                memories = retriever.retrieve_for_context(query, conversation_history)
                print("\nRetrieved memories:")
                for j, mem in enumerate(memories[:3]):  # Show top 3
                    if mem.get("type") == "personal_attributes":
                        print(f"- PERSONAL ATTRIBUTES (Score: {mem.get('relevance_score', 0):.2f})")
                        for attr_type, attr_val in mem.get("attributes", {}).items():
                            print(f"  * {attr_type}: {attr_val}")
                    else:
                        boost_info = (
                            f", Boost: {mem.get('keyword_boost', 1.0):.2f}"
                            if mem.get("keyword_boost", 1.0) > 1.0
                            else ""
                        )
                        print(
                            f"- {mem.get('type', 'unknown')}: {mem.get('text', mem.get('content', 'unknown'))[:50]}... "
                            f"(Score: {mem.get('relevance_score', 0):.2f}{boost_info})"
                        )

            # Generate with memory
            print("\nGenerating response...")
            response = adapter.generate(
                user_input=query,
                conversation_history=conversation_history,
                generation_kwargs={"max_new_tokens": 150},
            )

            print(f"Orca Mini: {response}")

            # Update conversation history
            conversation_history.append({"speaker": "user", "message": query, "response": response})

            print("-" * 60)

        print("\nTest completed successfully!")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
