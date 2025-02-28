"""
MemoryWeave Enhanced Retrieval Demo

This script demonstrates the enhanced retrieval mechanisms in MemoryWeave,
including two-stage retrieval, query type adaptation, and dynamic threshold adjustment.
"""

import json
import os
import time

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

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


class EnhancedRetrievalDemo:
    """
    A demo that showcases the enhanced retrieval mechanisms in MemoryWeave.
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
        emb_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        emb_model = AutoModel.from_pretrained(embedding_model_name)
        self.embedding_model = EmbeddingModelWrapper(emb_model, emb_tokenizer)

        # Initialize memory components with default settings
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

        # Initialize retriever with basic settings (will be configured later)
        self.retriever = ContextualRetriever(
            memory=self.memory,
            embedding_model=self.embedding_model,
            confidence_threshold=0.3,
        )

        # Results storage
        self.results = {}

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
                elif "description" in memory:
                    prompt += f"- {memory['description']}\n"
                elif "content" in memory:
                    prompt += f"- {memory['content']}\n"
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

    def generate_response(
        self, query, conversation_history, retriever_config, include_memories=True
    ):
        """Generate response with the specified retriever configuration."""
        # Configure retriever
        for param, value in retriever_config.items():
            if hasattr(self.retriever, param):
                setattr(self.retriever, param, value)

        # Retrieve relevant memories
        start_time = time.time()
        memories = self.retriever.retrieve_for_context(query, conversation_history)
        retrieval_time = time.time() - start_time

        # Format prompt
        prompt = self._format_prompt(
            query, conversation_history, include_memories=include_memories, memories=memories
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

        return response, memories, retrieval_time

    def run_demo(self, scenario, retriever_configs):
        """Run the demo with different retriever configurations."""
        results = {}

        for config_name, config in retriever_configs.items():
            print("\n" + "=" * 80)
            print(f"TESTING WITH {config_name}")
            print("=" * 80)

            conversation_history = []
            config_results = []

            # Introduction phase
            print("\nPHASE 1: INTRODUCING PERSONAL INFORMATION...")
            for turn in scenario["introduction"]:
                query = turn["query"]
                print(f"\nUser: {query}")

                response, memories, retrieval_time = self.generate_response(
                    query, conversation_history, config
                )
                print(f"Assistant: {response}")
                print(f"Retrieved {len(memories)} memories in {retrieval_time:.3f}s")

                conversation_history.append({"message": query, "response": response})

            # Distraction phase
            print("\nPHASE 2: CHANGING TOPICS...")
            for turn in scenario["distraction"]:
                query = turn["query"]
                print(f"\nUser: {query}")

                response, memories, retrieval_time = self.generate_response(
                    query, conversation_history, config
                )
                print(f"Assistant: {response}")
                print(f"Retrieved {len(memories)} memories in {retrieval_time:.3f}s")

                conversation_history.append({"message": query, "response": response})

            # Recall phase
            print("\nPHASE 3: TESTING RECALL...")
            for turn in scenario["recall"]:
                query = turn["query"]
                expected = turn["expected"]

                print(f"\nUser: {query}")

                response, memories, retrieval_time = self.generate_response(
                    query, conversation_history, config
                )
                print(f"Assistant: {response}")

                is_correct = self._check_contains_expected(response, expected)
                print(f"Retrieved {len(memories)} memories in {retrieval_time:.3f}s")
                print(f"Correct: {'✓' if is_correct else '✗'}")

                config_results.append(
                    {
                        "query": query,
                        "response": response,
                        "expected": expected,
                        "correct": is_correct,
                        "retrieved_memories": len(memories),
                        "retrieval_time": retrieval_time,
                    }
                )

                conversation_history.append({"message": query, "response": response})

            # Store results for this configuration
            results[config_name] = {
                "config": config,
                "results": config_results,
            }

        return results

    def generate_report(self, results):
        """Generate a report comparing different retriever configurations."""
        # Calculate metrics for each configuration
        summary = {}
        for config_name, config_results in results.items():
            results_list = config_results["results"]
            correct_count = sum(1 for r in results_list if r["correct"])
            accuracy = correct_count / len(results_list) if results_list else 0
            avg_retrieval_time = (
                sum(r["retrieval_time"] for r in results_list) / len(results_list)
                if results_list
                else 0
            )
            avg_memories = (
                sum(r["retrieved_memories"] for r in results_list) / len(results_list)
                if results_list
                else 0
            )

            summary[config_name] = {
                "accuracy": accuracy,
                "correct_count": correct_count,
                "total_queries": len(results_list),
                "avg_retrieval_time": avg_retrieval_time,
                "avg_memories_retrieved": avg_memories,
            }

        # Print report
        print("\n" + "=" * 80)
        print("RESULTS COMPARISON")
        print("=" * 80)

        # Sort by accuracy
        sorted_configs = sorted(summary.items(), key=lambda x: x[1]["accuracy"], reverse=True)

        for config_name, metrics in sorted_configs:
            print(f"\n{config_name}:")
            print(
                f"  Accuracy: {metrics['accuracy']:.2%} ({metrics['correct_count']}/{metrics['total_queries']})"
            )
            print(f"  Avg. Retrieval Time: {metrics['avg_retrieval_time']:.3f}s")
            print(f"  Avg. Memories Retrieved: {metrics['avg_memories_retrieved']:.1f}")

        # Identify best configuration
        best_config = sorted_configs[0][0]
        print(f"\nBest Configuration: {best_config}")

        # Save detailed results to file
        output = {
            "results": results,
            "summary": summary,
            "best_config": best_config,
        }

        with open("output/enhanced_retrieval_results.json", "w") as f:
            json.dump(output, f, indent=2)

        print("\nDetailed results saved to output/enhanced_retrieval_results.json")


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

# Define retriever configurations to test
retriever_configs = {
    "Baseline": {
        "confidence_threshold": 0.3,
        "adaptive_retrieval": False,
        "semantic_coherence_check": False,
        "use_two_stage_retrieval": False,
        "query_type_adaptation": False,
    },
    "Two-Stage Retrieval": {
        "confidence_threshold": 0.3,
        "adaptive_retrieval": False,
        "semantic_coherence_check": False,
        "use_two_stage_retrieval": True,
        "first_stage_k": 20,
        "query_type_adaptation": False,
    },
    "Query Type Adaptation": {
        "confidence_threshold": 0.3,
        "adaptive_retrieval": False,
        "semantic_coherence_check": False,
        "use_two_stage_retrieval": False,
        "query_type_adaptation": True,
    },
    "Adaptive K Selection": {
        "confidence_threshold": 0.3,
        "adaptive_retrieval": True,
        "adaptive_k_factor": 0.15,
        "semantic_coherence_check": False,
        "use_two_stage_retrieval": False,
        "query_type_adaptation": False,
    },
    "Semantic Coherence": {
        "confidence_threshold": 0.3,
        "adaptive_retrieval": False,
        "semantic_coherence_check": True,
        "use_two_stage_retrieval": False,
        "query_type_adaptation": False,
    },
    "Combined Approach": {
        "confidence_threshold": 0.2,
        "adaptive_retrieval": True,
        "adaptive_k_factor": 0.15,
        "semantic_coherence_check": True,
        "use_two_stage_retrieval": True,
        "first_stage_k": 20,
        "query_type_adaptation": True,
    },
}


def main():
    """Run the enhanced retrieval demo."""
    print("MemoryWeave Enhanced Retrieval Demo")
    print("===================================")

    # Initialize demo
    demo = EnhancedRetrievalDemo()

    # Run demo with different configurations
    results = demo.run_demo(personal_details_scenario, retriever_configs)

    # Generate report
    demo.generate_report(results)


if __name__ == "__main__":
    main()
