#!/usr/bin/env python3
"""
comparative_memory_benchmark.py

This script compares MemoryWeave against three common memory approaches:
1. No memory (baseline)
2. Simple vector similarity (traditional RAG)
3. Recency-biased retrieval (ChatGPT-like)
4. MemoryWeave's contextual fabric approach

The benchmark creates a realistic multi-turn conversation, injects facts,
and then tests recall accuracy after conversation drift.
"""

import argparse
import logging
import random
import secrets
import time

import numpy as np
from memoryweave_llm_wrapper import MemoryWeaveLLM, _get_device
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

# Import the API and compatibility wrapper
from memoryweave.api import MemoryWeaveAPI

# Setup console and logging
console = Console()
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(markup=True)])
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"
DEFAULT_TRIALS = 3  # Run multiple trials for statistical significance

# Test facts to inject at different points in conversation
TEST_FACTS = [
    "My favorite food is sushi, especially salmon nigiri.",
    "I have a pet cat named Felix who's 3 years old.",
    "I graduated from Stanford University in 2019.",
    "My sister's name is Elena and she lives in Boston.",
    "I'm allergic to peanuts so I need to be careful with food.",
    "I've been learning to play the piano for about 6 months.",
    "I plan to travel to Japan next spring for cherry blossom season.",
    "My favorite book is 'The Alchemist' by Paulo Coelho.",
]

# Conversation topics to drift between (realistic distractors)
CONVERSATION_TOPICS = [
    (
        "weather",
        [
            "What's the weather like today?",
            "Do you think it will rain later?",
            "I heard there might be a storm coming.",
        ],
    ),
    (
        "movies",
        [
            "Have you seen any good movies recently?",
            "What's your opinion on superhero films?",
            "I've been wanting to watch a good documentary.",
        ],
    ),
    (
        "technology",
        [
            "What do you think about the latest smartphones?",
            "Is AI getting too advanced too quickly?",
            "Have you tried any new apps lately?",
        ],
    ),
    (
        "books",
        [
            "I've been reading more fiction lately.",
            "Do you prefer e-books or physical books?",
            "What genre do you enjoy the most?",
        ],
    ),
    (
        "exercise",
        [
            "I've been trying to exercise more regularly.",
            "What's your favorite way to stay active?",
            "Do you think home workouts are effective?",
        ],
    ),
]


class SimpleVectorMemory:
    """
    A simple vector-based memory system that retrieves based on embedding similarity.
    This represents a traditional RAG approach.
    """

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.memories = []  # List of (text, embedding) tuples

    def add(self, text):
        embedding = self.embedding_model.encode(text, show_progress_bar=False)
        self.memories.append((text, embedding))

    def retrieve(self, query, top_k=3):
        query_embedding = self.embedding_model.encode(query, show_progress_bar=False)
        scores = []

        for text, embedding in self.memories:
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            scores.append((text, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [text for text, _ in scores[:top_k]]


class RecencyMemory:
    """
    A recency-biased memory system that prioritizes recent interactions.
    This represents systems like ChatGPT's memory approach.
    """

    def __init__(self, embedding_model, recency_weight=0.5):
        self.embedding_model = embedding_model
        self.memories = []  # List of (text, embedding, timestamp) tuples
        self.recency_weight = recency_weight

    def add(self, text):
        embedding = self.embedding_model.encode(text, show_progress_bar=False)
        timestamp = time.time()
        self.memories.append((text, embedding, timestamp))

    def retrieve(self, query, top_k=3):
        query_embedding = self.embedding_model.encode(query, show_progress_bar=False)
        scores = []

        # Get current time
        current_time = time.time()
        # Find the oldest and newest timestamps for normalization
        timestamps = [timestamp for _, _, timestamp in self.memories]
        min_time = min(timestamps) if timestamps else current_time
        max_time = current_time
        time_range = max_time - min_time if max_time > min_time else 1.0

        for text, embedding, timestamp in self.memories:
            # Compute similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )

            # Compute recency score (0 to 1, higher for more recent)
            recency = (timestamp - min_time) / time_range

            # Combine scores
            combined_score = (1 - self.recency_weight) * similarity + self.recency_weight * recency
            scores.append((text, combined_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [text for text, _ in scores[:top_k]]


class MemoryBenchmark:
    """
    Benchmark class that compares different memory approaches.
    """

    def __init__(self, model_name=DEFAULT_MODEL):
        self.model_name = model_name
        self.device = _get_device()

        # Create the memory systems to compare
        # 1. MemoryWeave
        self.api = MemoryWeaveAPI(model_name=model_name)

        # Create the compatibility wrapper for older examples
        self.memoryweave = MemoryWeaveLLM(model_name=model_name)

        # 2. Simple vector memory
        self.vector_memory = SimpleVectorMemory(self.api.embedding_model)

        # 3. Recency-biased memory
        self.recency_memory = RecencyMemory(self.api.embedding_model)

        # 4. No memory - we'll just use the bare model without memory features

        self.results = {
            "memoryweave": {"correct": 0, "partial": 0, "incorrect": 0, "total": 0},
            "vector": {"correct": 0, "partial": 0, "incorrect": 0, "total": 0},
            "recency": {"correct": 0, "partial": 0, "incorrect": 0, "total": 0},
            "none": {"correct": 0, "partial": 0, "incorrect": 0, "total": 0},
        }

    def run_trial(self, trial_num):
        """Run a full benchmark trial."""
        console.print(f"\n[bold cyan]Running Trial {trial_num}[/bold cyan]")

        # Select facts for this trial
        facts = random.sample(TEST_FACTS, 4)  # Use 4 facts per trial

        # Clear memories between trials
        self.api = MemoryWeaveAPI(model_name=self.model_name)
        self.memoryweave = MemoryWeaveLLM(model_name=self.model_name)
        self.vector_memory = SimpleVectorMemory(self.api.embedding_model)
        self.recency_memory = RecencyMemory(self.api.embedding_model)

        # Phase 1: Inject facts with conversation drift between them
        console.print("[bold]Phase 1: Injecting facts with conversation drift[/bold]")

        all_conversation = []

        for i, fact in enumerate(facts):
            # Add the fact
            console.print(f"Adding fact {i + 1}: {fact}")
            fact_prompt = f"I need to tell you something important about me: {fact}"

            # Store in all memory systems
            self.memoryweave.chat(fact_prompt)
            self.vector_memory.add(fact_prompt)
            self.recency_memory.add(fact_prompt)

            all_conversation.append(fact_prompt)

            # Add conversation drift (3 turns on a random topic)
            if i < len(facts) - 1:  # No need to drift after the last fact
                topic, messages = secrets.choice(CONVERSATION_TOPICS)
                console.print(f"Adding conversation drift on topic: {topic}")

                for msg in messages:
                    # Process with MemoryWeave (the others just store, don't respond)
                    response = self.memoryweave.chat(msg)
                    console.print(f"  User: {msg}")
                    console.print(f"  Assistant: {response[:80]}...")

                    # Store in all memory systems
                    self.vector_memory.add(msg)
                    self.vector_memory.add(f"Assistant: {response}")
                    self.recency_memory.add(msg)
                    self.recency_memory.add(f"Assistant: {response}")

                    all_conversation.append(msg)
                    all_conversation.append(f"Assistant: {response}")

        # Phase 2: Test recall for each fact
        console.print("\n[bold]Phase 2: Testing recall of facts[/bold]")

        fact_keywords = self._extract_fact_keywords(facts)

        for i, (fact, keywords) in enumerate(zip(facts, fact_keywords)):
            console.print(f"\nTesting recall of fact {i + 1}: {fact}")

            # Create a question for this fact
            question = self._generate_question_for_fact(fact)
            console.print(f"Question: {question}")

            # Test MemoryWeave
            mw_response = self.memoryweave.chat(question)
            mw_score = self._evaluate_recall(mw_response, keywords)
            console.print(f"[green]MemoryWeave response:[/green] {mw_response[:80]}...")
            console.print(f"Score: {mw_score}")
            self._update_results("memoryweave", mw_score)

            # Test Vector Memory
            vector_memories = self.vector_memory.retrieve(question)
            vector_response = self._generate_with_context(question, vector_memories)
            vector_score = self._evaluate_recall(vector_response, keywords)
            console.print(f"[yellow]Vector memory response:[/yellow] {vector_response[:80]}...")
            console.print(f"Score: {vector_score}")
            self._update_results("vector", vector_score)

            # Test Recency Memory
            recency_memories = self.recency_memory.retrieve(question)
            recency_response = self._generate_with_context(question, recency_memories)
            recency_score = self._evaluate_recall(recency_response, keywords)
            console.print(f"[cyan]Recency memory response:[/cyan] {recency_response[:80]}...")
            console.print(f"Score: {recency_score}")
            self._update_results("recency", recency_score)

            # Test No Memory
            no_memory_response = self.api.chat_without_memory(question)
            no_memory_score = self._evaluate_recall(no_memory_response, keywords)
            console.print(f"[red]No memory response:[/red] {no_memory_response[:80]}...")
            console.print(f"Score: {no_memory_score}")
            self._update_results("none", no_memory_score)

    def _extract_fact_keywords(self, facts):
        """Extract key terms from each fact that should appear in correct recall."""
        fact_keywords = []

        for fact in facts:
            if "favorite food" in fact.lower():
                keywords = ["sushi", "salmon", "nigiri"]
            elif "pet" in fact.lower():
                keywords = ["cat", "Felix", "3"]
            elif "graduated" in fact.lower():
                keywords = ["Stanford", "2019"]
            elif "sister" in fact.lower():
                keywords = ["Elena", "Boston"]
            elif "allergic" in fact.lower():
                keywords = ["allergic", "peanuts"]
            elif "piano" in fact.lower():
                keywords = ["piano", "6", "months"]
            elif "travel" in fact.lower():
                keywords = ["Japan", "spring", "cherry blossom"]
            elif "favorite book" in fact.lower():
                keywords = ["Alchemist", "Paulo Coelho"]
            else:
                # For unknown facts, split and take nouns/names
                keywords = [word for word in fact.split() if word[0].isupper() or len(word) > 5]

            fact_keywords.append(keywords)

        return fact_keywords

    def _generate_question_for_fact(self, fact):
        """Generate a question that should prompt recall of the fact."""
        if "favorite food" in fact.lower():
            return "What's my favorite food again?"
        elif "pet" in fact.lower():
            return "Tell me about my pet."
        elif "graduated" in fact.lower():
            return "Where did I graduate from?"
        elif "sister" in fact.lower():
            return "What was my sister's name and where does she live?"
        elif "allergic" in fact.lower():
            return "Do I have any food allergies?"
        elif "piano" in fact.lower():
            return "What instrument have I been learning to play?"
        elif "travel" in fact.lower():
            return "Did I mention any travel plans?"
        elif "favorite book" in fact.lower():
            return "What's my favorite book?"
        else:
            return "Can you remind me of what I told you about myself?"

    def _evaluate_recall(self, response, keywords):
        """
        Evaluate recall quality:
        - "correct": All keywords present
        - "partial": Some keywords present
        - "incorrect": No keywords present
        """
        response_lower = response.lower()
        found_keywords = [keyword.lower() in response_lower for keyword in keywords]

        if all(found_keywords):
            return "correct"
        elif any(found_keywords):
            return "partial"
        else:
            return "incorrect"

    def _update_results(self, system, score):
        """Update the results dictionary."""
        self.results[system][score] += 1
        self.results[system]["total"] += 1

    def _generate_with_context(self, question, context_items):
        """Generate a response using the LLM with provided context."""
        if not context_items:
            return self.api.chat_without_memory(question)

        # Create a simple context-augmented prompt
        context_text = "\n".join(context_items)
        prompt = f"""You are a helpful assistant with memory of previous conversations.

Previous conversation includes:
{context_text}

Based on this context, please answer the user's question:
User: {question}
"""

        # Generate response directly from the API
        response = self.api.chat_without_memory(prompt)

        # Extract just the assistant's response (if needed)
        if "User: " + question in response:
            assistant_response = response.split("User: " + question)[-1].strip()
            return assistant_response
        return response

    def display_results(self):
        """Display the benchmark results."""
        table = Table(title="Memory System Comparison")

        table.add_column("Memory System", style="cyan")
        table.add_column("Correct", style="green")
        table.add_column("Partial", style="yellow")
        table.add_column("Incorrect", style="red")
        table.add_column("Success Rate", style="magenta")

        for system, results in self.results.items():
            if results["total"] == 0:
                success_rate = "0%"
            else:
                # Calculate a weighted success rate: correct=1.0, partial=0.5, incorrect=0.0
                weighted_success = (results["correct"] + 0.5 * results["partial"]) / results[
                    "total"
                ]
                success_rate = f"{weighted_success:.1%}"

            system_name = {
                "memoryweave": "MemoryWeave",
                "vector": "Vector Similarity",
                "recency": "Recency-biased",
                "none": "No Memory",
            }[system]

            table.add_row(
                system_name,
                str(results["correct"]),
                str(results["partial"]),
                str(results["incorrect"]),
                success_rate,
            )

        console.print(table)

        # Display improvement statistics
        if self.results["none"]["total"] > 0 and self.results["memoryweave"]["total"] > 0:
            none_success = (
                self.results["none"]["correct"] + 0.5 * self.results["none"]["partial"]
            ) / self.results["none"]["total"]
            mw_success = (
                self.results["memoryweave"]["correct"]
                + 0.5 * self.results["memoryweave"]["partial"]
            ) / self.results["memoryweave"]["total"]
            improvement = mw_success - none_success
            console.print(
                f"\n[bold]MemoryWeave improvement over No Memory:[/bold] {improvement:.1%}"
            )

            vector_success = (
                self.results["vector"]["correct"] + 0.5 * self.results["vector"]["partial"]
            ) / self.results["vector"]["total"]
            improvement = mw_success - vector_success
            console.print(
                f"[bold]MemoryWeave improvement over Vector Similarity:[/bold] {improvement:.1%}"
            )

            recency_success = (
                self.results["recency"]["correct"] + 0.5 * self.results["recency"]["partial"]
            ) / self.results["recency"]["total"]
            improvement = mw_success - recency_success
            console.print(
                f"[bold]MemoryWeave improvement over Recency-biased:[/bold] {improvement:.1%}"
            )


def run_benchmark(model_name, num_trials):
    """Run the memory benchmark."""
    console.print(f"[bold]Running Memory Benchmark with model: {model_name}[/bold]")
    console.print(f"Number of trials: {num_trials}")

    benchmark = MemoryBenchmark(model_name=model_name)

    for i in range(1, num_trials + 1):
        benchmark.run_trial(i)

    console.print("\n[bold]Benchmark Results[/bold]")
    benchmark.display_results()


def main():
    """Run the benchmark script."""
    parser = argparse.ArgumentParser(description="Compare memory system approaches.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Hugging Face model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help=f"Number of trials to run (default: {DEFAULT_TRIALS})",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        console.print("[yellow]Debug logging enabled[/yellow]")

    run_benchmark(args.model, args.trials)


if __name__ == "__main__":
    main()
