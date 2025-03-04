#!/usr/bin/env python3
"""
temporal_benchmark.py

This script evaluates MemoryWeave's ability to handle temporal references
compared to traditional retrieval methods (vector similarity and recency-based).
It tests scenarios like "what did I say yesterday" or "remind me what we
discussed last week about X" that require understanding temporal context.
"""

import logging
import random
import secrets
import time
from datetime import datetime

import numpy as np
import rich_click as click
from memoryweave_llm_wrapper import MemoryWeaveLLM, _get_device, get_llm, get_tokenizer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

# Setup console and logging
console = Console()
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(markup=True)])
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"

# Define temporal reference types
TEMPORAL_REFERENCES = [
    "yesterday",
    "last week",
    "earlier today",
    "this morning",
    "previously",
    "before",
    "a while ago",
]

# Test scenarios - pairs of (fact, query_template)
TEST_SCENARIOS = [
    ("I visited the Golden Gate Bridge {when}.", "Where did I go {temporal_ref}?"),
    ("I had pasta with mushrooms for dinner {when}.", "What did I eat {temporal_ref}?"),
    ("I watched a documentary about dolphins {when}.", "What show did I watch {temporal_ref}?"),
    (
        "I learned that Jupiter has 79 known moons {when}.",
        "Tell me what I learned about Jupiter {temporal_ref}.",
    ),
    ("My friend Emma called me about her new job {when}.", "Who called me {temporal_ref}?"),
    ("I fixed my bicycle's flat tire {when}.", "What did I repair {temporal_ref}?"),
]


# Simple implementation of alternative memory approaches for comparison
class SimpleVectorMemory:
    """Basic vector similarity retrieval without temporal understanding."""

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.memories = []  # List of (text, embedding, timestamp) tuples

    def add(self, text, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        embedding = self.embedding_model.encode(text, show_progress_bar=False)
        self.memories.append((text, embedding, timestamp))

    def retrieve(self, query, top_k=3):
        query_embedding = self.embedding_model.encode(query, show_progress_bar=False)
        scores = []

        for text, embedding, _ in self.memories:
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            scores.append((text, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [text for text, _ in scores[:top_k]]


class RecencyMemory:
    """Recency-biased retrieval that favors recent memories."""

    def __init__(self, embedding_model, recency_weight=0.7):
        self.embedding_model = embedding_model
        self.memories = []  # List of (text, embedding, timestamp) tuples
        self.recency_weight = recency_weight

    def add(self, text, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        embedding = self.embedding_model.encode(text, show_progress_bar=False)
        self.memories.append((text, embedding, timestamp))

    def retrieve(self, query, top_k=3):
        query_embedding = self.embedding_model.encode(query, show_progress_bar=False)
        scores = []

        # Get current time and normalize timestamps
        current_time = time.time()
        timestamps = [ts for _, _, ts in self.memories]
        min_time = min(timestamps) if timestamps else current_time
        max_time = current_time
        time_range = max_time - min_time if max_time > min_time else 1.0

        for text, embedding, timestamp in self.memories:
            # Compute similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )

            # Compute recency (0 to 1, higher for more recent)
            recency = (timestamp - min_time) / time_range

            # Combine scores
            combined_score = (1 - self.recency_weight) * similarity + self.recency_weight * recency
            scores.append((text, combined_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [text for text, _ in scores[:top_k]]


class TemporalBenchmark:
    """Benchmark for comparing temporal understanding capabilities."""

    def __init__(self, model_name=DEFAULT_MODEL):
        self.model_name = model_name
        self.device = _get_device()
        self.tokenizer = get_tokenizer(model_name)
        self.model = get_llm(model_name, device=self.device)

        # Initialize memory systems
        self.memoryweave = MemoryWeaveLLM(model_name=model_name)
        self.vector_memory = SimpleVectorMemory(self.memoryweave.embedding_model)
        self.recency_memory = RecencyMemory(self.memoryweave.embedding_model)

        # Results tracking
        self.results = {
            "memoryweave": {"correct": 0, "partial": 0, "incorrect": 0, "total": 0},
            "vector": {"correct": 0, "partial": 0, "incorrect": 0, "total": 0},
            "recency": {"correct": 0, "partial": 0, "incorrect": 0, "total": 0},
        }

        # Time periods for simulated memories
        self.now = time.time()
        self.yesterday = self.now - 86400  # 24 hours ago
        self.last_week = self.now - 604800  # 7 days ago
        self.earlier_today = self.now - 21600  # 6 hours ago
        self.this_morning = self.now - 43200  # 12 hours ago

        self.time_periods = {
            "yesterday": self.yesterday,
            "last week": self.last_week,
            "earlier today": self.earlier_today,
            "this morning": self.this_morning,
            "previously": self.last_week,
            "before": self.earlier_today,
            "a while ago": self.last_week,
        }

    def run_benchmark(self, num_scenarios=6):
        """Run the benchmark with a specified number of scenarios."""

        # Select a subset of scenarios to test
        scenarios = random.sample(TEST_SCENARIOS, min(num_scenarios, len(TEST_SCENARIOS)))

        # Add some distractors/conversation drift
        distractors = [
            "I'm thinking about buying a new laptop.",
            "The weather is really nice today.",
            "I've been trying to exercise more regularly.",
            "I wonder if there are any good movies playing this weekend.",
            "Do you have any book recommendations?",
        ]

        console.print("[bold cyan]Temporal Reference Benchmark[/bold cyan]")
        console.print("[bold]Phase 1: Adding memories with timestamps[/bold]")

        # Track fact-temporal_ref-keyword tuples for evaluation
        fact_metadata = []

        # Add memories with timestamps for different time periods
        for i, (fact_template, _query_template) in enumerate(scenarios):
            # Select a random temporal reference
            temporal_ref = secrets.choice(TEMPORAL_REFERENCES)
            timestamp = self.time_periods[temporal_ref]

            # Format the fact with when it happened
            when_str = self._get_natural_time_description(timestamp)
            fact = fact_template.format(when=when_str)

            console.print(f"\nAdding fact {i + 1}: {fact}")
            console.print(f"Associated time: {datetime.fromtimestamp(timestamp)}")

            # Add to all memory systems with the appropriate timestamp
            self.memoryweave.add_memory(
                fact, {"type": "user_fact", "created_at": timestamp, "importance": 0.8}
            )
            self.vector_memory.add(fact, timestamp)
            self.recency_memory.add(fact, timestamp)

            # Extract keywords for evaluation
            keywords = self._extract_keywords(fact)
            fact_metadata.append((fact, temporal_ref, keywords))

            # Add distractor conversations between facts
            if i < len(scenarios) - 1:
                for _ in range(2):
                    distractor = secrets.choice(distractors)
                    console.print(f"Adding distractor: {distractor}")

                    # Add distractor with current timestamp
                    current_ts = time.time()
                    self.memoryweave.add_memory(
                        distractor,
                        {"type": "user_message", "created_at": current_ts, "importance": 0.4},
                    )
                    self.vector_memory.add(distractor, current_ts)
                    self.recency_memory.add(distractor, current_ts)

                    # Small delay to ensure different timestamps
                    time.sleep(0.1)

        console.print("\n[bold]Phase 2: Querying with temporal references[/bold]")

        # Now test recall with temporal references
        for i, (_fact, temporal_ref, keywords) in enumerate(fact_metadata):
            # Get the query template from the corresponding scenario
            _, query_template = scenarios[i]

            # Construct the query
            query = query_template.format(temporal_ref=temporal_ref)
            console.print(f"\n[bold]Testing query {i + 1}:[/bold] {query}")
            console.print(f"Expected keywords: {', '.join(keywords)}")

            # Test MemoryWeave
            console.print("\n[green]Testing MemoryWeave:[/green]")
            mw_response = self.memoryweave.chat(query)
            console.print(f"Response: {mw_response}")
            mw_score = self._evaluate_recall(mw_response, keywords)
            if mw_score == "correct":
                mw_color = "bold green"
            elif mw_score == "partial":
                mw_color = "bold yellow"
            else:
                mw_color = "bold red"
            console.print(f"SCORE: {mw_score}", style=mw_color)
            self._update_results("memoryweave", mw_score)

            # Test Vector Memory
            console.print("\n[yellow]Testing Vector Similarity:[/yellow]")
            vector_memories = self.vector_memory.retrieve(query)
            vector_response = self._generate_with_context(query, vector_memories)
            console.print(f"Response: {vector_response}")
            vector_score = self._evaluate_recall(vector_response, keywords)
            console.print(f"Score: {vector_score}")
            self._update_results("vector", vector_score)

            # Test Recency Memory
            console.print("\n[cyan]Testing Recency-biased:[/cyan]")
            recency_memories = self.recency_memory.retrieve(query)
            recency_response = self._generate_with_context(query, recency_memories)
            console.print(f"Response: {recency_response}")
            recency_score = self._evaluate_recall(recency_response, keywords)
            console.print(f"Score: {recency_score}")
            self._update_results("recency", recency_score)

        # Display results
        self.display_results()

    def _get_natural_time_description(self, timestamp):
        """Convert timestamp to natural language description of when it happened."""
        dt = datetime.fromtimestamp(timestamp)
        now_dt = datetime.fromtimestamp(self.now)

        # Calculate differences
        delta = now_dt - dt

        if delta.days == 1:
            return "yesterday"
        elif delta.days > 1 and delta.days <= 7:
            return f"{delta.days} days ago"
        elif delta.days > 7:
            return "last week"
        elif delta.seconds > 43200:  # 12 hours
            return "this morning"
        elif delta.seconds > 10800:  # 3 hours
            return "earlier today"
        else:
            return "a little while ago"

    def _extract_keywords(self, fact):
        """Extract key terms from the fact that should appear in correct recall."""
        # Simple extraction - get important nouns and entities
        words = fact.split()
        keywords = []

        # Look for capitalized words (likely entities) and longer words
        for word in words:
            word = word.strip(".,!?()\"':;")
            if word and (word[0].isupper() or len(word) > 6):
                keywords.append(word)

        # Add some domain-specific keywords based on content
        if "pasta" in fact.lower():
            keywords.extend(["pasta", "mushrooms", "dinner"])
        elif "bridge" in fact.lower():
            keywords.extend(["Golden Gate", "Bridge", "visited"])
        elif "documentary" in fact.lower():
            keywords.extend(["documentary", "dolphins", "watched"])
        elif "Jupiter" in fact.lower():
            keywords.extend(["Jupiter", "moons", "79"])
        elif "Emma" in fact.lower():
            keywords.extend(["Emma", "called", "job"])
        elif "bicycle" in fact.lower():
            keywords.extend(["bicycle", "tire", "flat", "fixed"])

        # Make the keywords unique and ensure we have at least 2
        keywords = list(set(keywords))
        if len(keywords) < 2:
            keywords.append("important")

        return keywords

    def _evaluate_recall(self, response, keywords):
        """
        Evaluate recall quality:
        - "correct": At least 2 keywords present
        - "partial": At least 1 keyword present
        - "incorrect": No keywords present
        """
        response_lower = response.lower()
        found_keywords = [keyword.lower() in response_lower for keyword in keywords]

        if sum(found_keywords) >= 2:
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
            return "I don't have any information about that."

        # Create a simple context-augmented prompt
        context_text = "\n".join(context_items)
        prompt = f"""You are a helpful assistant with memory of previous conversations.

Previous conversation includes:
{context_text}

Based on this context, please answer the user's question:
User: {question}
Assistant:"""

        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with np.errstate(all="ignore"):  # Ignore numpy warnings
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the assistant's response
        assistant_response = full_response.split("User: " + question)[-1].strip()
        if assistant_response.startswith("Assistant:"):
            assistant_response = assistant_response[len("Assistant:") :].strip()

        return assistant_response

    def display_results(self):
        """Display the benchmark results."""
        table = Table(title="Temporal Reference Benchmark Results")

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
        if all(self.results[s]["total"] > 0 for s in ["memoryweave", "vector", "recency"]):
            mw_success = (
                self.results["memoryweave"]["correct"]
                + 0.5 * self.results["memoryweave"]["partial"]
            ) / self.results["memoryweave"]["total"]

            vector_success = (
                self.results["vector"]["correct"] + 0.5 * self.results["vector"]["partial"]
            ) / self.results["vector"]["total"]

            recency_success = (
                self.results["recency"]["correct"] + 0.5 * self.results["recency"]["partial"]
            ) / self.results["recency"]["total"]

            vector_improvement = mw_success - vector_success
            recency_improvement = mw_success - recency_success

            console.print(
                f"\n[bold]MemoryWeave vs Vector Similarity:[/bold] +{vector_improvement:.1%}"
            )
            console.print(f"[bold]MemoryWeave vs Recency-biased:[/bold] +{recency_improvement:.1%}")


@click.command(
    context_settings=dict(help_option_names=["-h", "--help"]),
    short_help="Run MemoryWeave temporal benchmark compared to other common approaches",
)
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    help=f"Name of the Hugging Face model to load (default: {DEFAULT_MODEL})",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging for more detailed output.",
)
@click.option(
    "--scenarios",
    type=int,
    default=4,
    help="Number of scenarios to test (default: 4)",
)
def main(model, debug, scenarios):
    # Set up debug logging if requested
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        console.print("[yellow]Debug logging enabled[/yellow]")

    # Run benchmark
    benchmark = TemporalBenchmark(model_name=model)
    benchmark.run_benchmark(num_scenarios=scenarios)


if __name__ == "__main__":
    main()
