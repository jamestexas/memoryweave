import torch
from transformers import AutoModel, AutoTokenizer

from memoryweave.core import ContextualMemory, ContextualRetriever, MemoryEncoder
from memoryweave.evaluation import evaluate_conversation


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
    # Load embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    embedding_model = EmbeddingModelWrapper(model, tokenizer)

    # Initialize memory components
    embedding_dim = model.config.hidden_size
    memory = ContextualMemory(embedding_dim=embedding_dim)
    encoder = MemoryEncoder(embedding_model)
    retriever = ContextualRetriever(memory, embedding_model)

    memory_system = {"memory": memory, "encoder": encoder, "retriever": retriever}

    # Sample conversation with coherence challenges
    conversation = [
        {
            "speaker": "user",
            "message": "My favorite movie is The Matrix. Have you seen it?",
            "response": "Yes, The Matrix is a classic sci-fi film from 1999. It features amazing special effects and a compelling story about reality and perception.",
        },
        {
            "speaker": "user",
            "message": "I also enjoy hiking on weekends. Do you like outdoor activities?",
            "response": "Hiking is a wonderful way to connect with nature! There are many benefits to outdoor activities like hiking, including physical exercise and mental rejuvenation.",
        },
        {
            "speaker": "user",
            "message": "What was that movie I mentioned earlier?",
            "response": "You mentioned that your favorite movie is The Matrix.",
        },
        {
            "speaker": "user",
            "message": "Can you recommend similar movies?",
            "response": "If you enjoyed The Matrix, you might also like these similar sci-fi films: Inception, Blade Runner, The Thirteenth Floor, and Existenz. They all deal with questions of reality and perception.",
        },
    ]

    # Add all conversation turns to memory
    for turn in conversation:
        embedding, metadata = encoder.encode_interaction(
            message=turn["message"], speaker=turn["speaker"], response=turn["response"]
        )
        memory.add_memory(embedding, turn["message"], metadata)

    # Evaluate the conversation
    print("Evaluating conversation quality...")
    eval_results = evaluate_conversation(conversation, memory_system, embedding_model)

    print(f"Overall coherence score: {eval_results['overall_coherence']:.2f}")
    print(f"Average memory relevance: {eval_results['average_relevance']:.2f}")
    print(f"Average response consistency: {eval_results['average_consistency']:.2f}")

    print("\nPer-turn scores:")
    for score in eval_results["turn_scores"]:
        print(
            f"Turn {score['turn']}: Relevance={score['relevance']:.2f}, Consistency={score['consistency']:.2f}"
        )


if __name__ == "__main__":
    main()
