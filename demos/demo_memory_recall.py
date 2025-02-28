"""
MemoryWeave Memory Recall Demo

This script demonstrates how MemoryWeave improves a language model's ability
to recall information from earlier in a conversation, even after topic shifts.

It runs a hardcoded conversation scenario with and without memory augmentation,
then compares the results.
"""

import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memoryweave.core import ContextualMemory, ContextualRetriever, MemoryEncoder

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)


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


class MemoryRecallDemo:
    """
    A simple demo that tests recall capabilities with and without MemoryWeave.
    """

    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.1"):
        """
        Initialize the demo with a specified language model.

        Args:
            model_name: Name of the HuggingFace model to use
        """
        print(f"Loading language model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )

        # Load embedding model (smaller model for embeddings)
        embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"Loading embedding model: {embedding_model_name}")
        from transformers import AutoModel

        emb_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        emb_model = AutoModel.from_pretrained(embedding_model_name)
        self.embedding_model = EmbeddingModelWrapper(emb_model, emb_tokenizer)

        # Initialize memory components
        embedding_dim = emb_model.config.hidden_size
        self.memory = ContextualMemory(
            embedding_dim=embedding_dim,
            use_art_clustering=True,
            vigilance_threshold=0.75,
            default_confidence_threshold=0.3,
            adaptive_retrieval=True,
            semantic_coherence_check=True,
        )
        self.encoder = MemoryEncoder(self.embedding_model)
        self.retriever = ContextualRetriever(
            memory=self.memory, embedding_model=self.embedding_model, confidence_threshold=0.3
        )

        # Results storage
        self.results_with_memory = []
        self.results_without_memory = []

    def _format_prompt(
        self, query, conversation_history=None, include_memories=False, memories=None
    ):
        """Format prompt for Mistral model."""
        if conversation_history is None:
            conversation_history = []

        # Start with system prompt
        prompt = "<s>[INST] "

        # Add memory context if available and requested
        if include_memories and memories:
            prompt += "Here's some relevant information from earlier in our conversation:\n"
            for _, memory in enumerate(memories[:3]):  # Limit to top 3 memories
                if "text" in memory:
                    prompt += f"- {memory['text']}\n"
            prompt += "\n"

        # Add recent conversation history (last 3 turns)
        if conversation_history:
            for turn in conversation_history[-3:]:
                prompt += f"User: {turn['message']}\nAssistant: {turn['response']}\n"

        # Add current query
        prompt += f"User: {query} [/INST]"

        return prompt

    def _check_contains_expected(self, response, expected_items):
        """Check if response contains expected items."""
        if not expected_items:
            return True

        response_lower = response.lower()
        for item in expected_items:
            if item.lower() not in response_lower:
                return False
        return True

    def generate_with_memory(self, query, conversation_history):
        """Generate response with memory augmentation."""
        # Retrieve relevant memories
        memories = self.retriever.retrieve_for_context(query, conversation_history)

        # Format prompt with memories
        prompt = self._format_prompt(
            query, conversation_history, include_memories=True, memories=memories
        )

        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=100,
            temperature=0.7,
        )
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        # Store interaction in memory
        embedding, metadata = self.encoder.encode_interaction(
            message=query, speaker="user", response=response
        )
        self.memory.add_memory(embedding, query, metadata)

        return response, memories

    def generate_without_memory(self, query, conversation_history):
        """Generate response without memory augmentation."""
        # Format prompt without memories
        prompt = self._format_prompt(query, conversation_history)

        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=100,
            temperature=0.7,
        )
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        return response

    def run_test(self, scenario):
        """Run the test scenario with and without memory."""
        # Test with memory
        print("\n" + "=" * 80)
        print("TESTING WITH MEMORY AUGMENTATION")
        print("=" * 80)

        conversation_history = []

        # Introduction phase
        print("\nPHASE 1: INTRODUCING PERSONAL INFORMATION...")
        for turn in scenario["introduction"]:
            query = turn["query"]
            print(f"\nUser: {query}")

            response, _ = self.generate_with_memory(query, conversation_history)
            print(f"Assistant: {response}")

            conversation_history.append({"message": query, "response": response})

        # Distraction phase
        print("\nPHASE 2: CHANGING TOPICS...")
        for turn in scenario["distraction"]:
            query = turn["query"]
            print(f"\nUser: {query}")

            response, _ = self.generate_with_memory(query, conversation_history)
            print(f"Assistant: {response}")

            conversation_history.append({"message": query, "response": response})

        # Recall phase
        print("\nPHASE 3: TESTING RECALL...")
        for turn in scenario["recall"]:
            query = turn["query"]
            expected = turn["expected"]

            print(f"\nUser: {query}")

            response, memories = self.generate_with_memory(query, conversation_history)
            print(f"Assistant: {response}")

            is_correct = self._check_contains_expected(response, expected)
            print(f"Retrieved memories: {len(memories)}")
            print(f"Correct: {'✓' if is_correct else '✗'}")

            self.results_with_memory.append(
                {
                    "query": query,
                    "response": response,
                    "expected": expected,
                    "correct": is_correct,
                    "retrieved_memories": len(memories),
                }
            )

            conversation_history.append({"message": query, "response": response})

        # Test without memory
        print("\n" + "=" * 80)
        print("TESTING WITHOUT MEMORY AUGMENTATION")
        print("=" * 80)

        conversation_history = []

        # Introduction phase
        print("\nPHASE 1: INTRODUCING PERSONAL INFORMATION...")
        for turn in scenario["introduction"]:
            query = turn["query"]
            print(f"\nUser: {query}")

            response = self.generate_without_memory(query, conversation_history)
            print(f"Assistant: {response}")

            conversation_history.append({"message": query, "response": response})

        # Distraction phase
        print("\nPHASE 2: CHANGING TOPICS...")
        for turn in scenario["distraction"]:
            query = turn["query"]
            print(f"\nUser: {query}")

            response = self.generate_without_memory(query, conversation_history)
            print(f"Assistant: {response}")

            conversation_history.append({"message": query, "response": response})

        # Recall phase
        print("\nPHASE 3: TESTING RECALL...")
        for turn in scenario["recall"]:
            query = turn["query"]
            expected = turn["expected"]

            print(f"\nUser: {query}")

            response = self.generate_without_memory(query, conversation_history)
            print(f"Assistant: {response}")

            is_correct = self._check_contains_expected(response, expected)
            print(f"Correct: {'✓' if is_correct else '✗'}")

            self.results_without_memory.append(
                {
                    "query": query,
                    "response": response,
                    "expected": expected,
                    "correct": is_correct,
                    "retrieved_memories": 0,
                }
            )

            conversation_history.append({"message": query, "response": response})

    def generate_report(self):
        """Generate a report comparing results with and without memory."""
        # Calculate accuracy
        with_memory_correct = sum(1 for r in self.results_with_memory if r["correct"])
        without_memory_correct = sum(1 for r in self.results_without_memory if r["correct"])

        with_memory_accuracy = (
            with_memory_correct / len(self.results_with_memory) if self.results_with_memory else 0
        )
        without_memory_accuracy = (
            without_memory_correct / len(self.results_without_memory)
            if self.results_without_memory
            else 0
        )

        # Print report
        print("\n" + "=" * 80)
        print("RESULTS COMPARISON")
        print("=" * 80)

        print("\nWith Memory Augmentation:")
        print(f"  Correct: {with_memory_correct}/{len(self.results_with_memory)}")
        print(f"  Accuracy: {with_memory_accuracy:.2%}")

        print("\nWithout Memory Augmentation:")
        print(f"  Correct: {without_memory_correct}/{len(self.results_without_memory)}")
        print(f"  Accuracy: {without_memory_accuracy:.2%}")

        print(f"\nImprovement: {(with_memory_accuracy - without_memory_accuracy):.2%}")

        # Save detailed results to file
        results = {
            "with_memory": self.results_with_memory,
            "without_memory": self.results_without_memory,
            "summary": {
                "with_memory_accuracy": with_memory_accuracy,
                "without_memory_accuracy": without_memory_accuracy,
                "improvement": with_memory_accuracy - without_memory_accuracy,
            },
        }

        with open("output/memory_recall_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print("\nDetailed results saved to output/memory_recall_results.json")


# Define test scenario
personal_details_scenario = {
    "introduction": [
        {"query": "My name is Alex Thompson.", "expected": None},
        {"query": "I live in Portland, Oregon.", "expected": None},
        {"query": "I have a dog named Rusty who's 5 years old.", "expected": None},
        {"query": "My favorite food is Thai curry.", "expected": None},
        {"query": "I work as a graphic designer at a marketing agency.", "expected": None},
    ],
    "distraction": [
        {
            "query": "Let's talk about something different. What's the capital of France?",
            "expected": None,
        },
        {"query": "Tell me about quantum computing.", "expected": None},
        {"query": "What are the main features of Python programming?", "expected": None},
        {"query": "Explain the concept of climate change.", "expected": None},
        {"query": "What are some popular tourist destinations in Japan?", "expected": None},
        {"query": "How does blockchain technology work?", "expected": None},
        {"query": "What are the benefits of regular exercise?", "expected": None},
        {"query": "Tell me about the history of the internet.", "expected": None},
        {"query": "What are some good books you'd recommend?", "expected": None},
        {"query": "How do electric cars work?", "expected": None},
    ],
    "recall": [
        {"query": "What's my name?", "expected": ["Alex", "Thompson"]},
        {"query": "Where do I live?", "expected": ["Portland", "Oregon"]},
        {"query": "What's my dog's name and how old is he?", "expected": ["Rusty", "5"]},
        {"query": "What kind of food do I like?", "expected": ["Thai", "curry"]},
        {"query": "What's my profession?", "expected": ["graphic designer", "marketing"]},
    ],
}


def main():
    """Run the memory recall demo."""
    print("MemoryWeave Memory Recall Demo")
    print("==============================")

    # Initialize demo
    demo = MemoryRecallDemo()

    # Run test
    demo.run_test(personal_details_scenario)

    # Generate report
    demo.generate_report()


if __name__ == "__main__":
    main()
