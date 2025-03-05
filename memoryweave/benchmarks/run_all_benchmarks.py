#!/usr/bin/env python3
"""
Unified Retrieval Benchmark for MemoryWeave

This script benchmarks MemoryWeave using shared resources to compare
three retrieval strategies:
    - hybrid (MemoryWeaveHybridAPI)
    - standard (MemoryWeaveAPI with standard configuration)
    - chunked (ChunkedMemoryWeaveAPI)
    - standard_rag (MemoryWeaveAPI configured as a standard RAG)

Each system uses the built‑in chat method that manages context automatically.
Results (average query time and accuracy) are aggregated across scenarios,
displayed in a rich table, and visualized with matplotlib charts.
"""

import gc
import json
import logging
import os
import secrets
import time
import traceback
from datetime import datetime
from enum import Enum
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rich_click as click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# Import sentence-transformers once at the beginning
from sentence_transformers import util

from memoryweave.api.chunked_memory_weave import ChunkedMemoryWeaveAPI
from memoryweave.api.hybrid_memory_weave import HybridMemoryWeaveAPI
from memoryweave.api.llm_provider import LLMProvider
from memoryweave.api.memory_weave import MemoryWeaveAPI
from memoryweave.components.retriever import _get_embedder

# Configure logging
console = Console(highlight=True)
matplotlib.use("Agg")  # Use non-interactive backend for matplotlib
logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(show_path=False, rich_tracebacks=True)],
)
logger = logging.getLogger("unified_benchmark")

DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = "./benchmark_results"
DEFAULT_SCENARIOS = dict(
    factual=dict(
        name="Factual Recall",
        description="Basic factual recall testing.",
        queries=[
            ("What is my name?", ["Alex", "Thompson"]),
            ("Where do I live?", ["Seattle"]),
        ],
    ),
    temporal=dict(
        name="Temporal Context",
        description="Tests ability to understand time references.",
        queries=[
            ("What did I mention earlier about my home?", ["Seattle"]),
            ("What did I talk about yesterday?", ["Alex", "Thompson", "Seattle"]),
        ],
    ),
    complex=dict(
        name="Complex Context",
        description="Tests handling of longer, more complex information.",
        queries=[
            (
                "I need to remember the details about my new medical regimen. Can you help?",
                ["medication", "doctor", "treatment"],
            ),
        ],
    ),
    preload=[
        "I just got back from the doctor. She said I need to take 25mg of Lisinopril every morning for my blood pressure, 10mg of Zolpidem before bed for sleep, and 500mg of Metformin with food twice daily for my blood sugar. I also need to schedule a follow-up in 3 months to check how the treatment is working."
    ],
)
default_scenarios = {
    "factual": {
        "name": "Factual Recall",
        "description": "Basic factual recall testing.",
        "queries": [
            ("What is my name?", ["Alex", "Thompson"]),
            ("Where do I live?", ["Seattle"]),
        ],
    },
    "temporal": {
        "name": "Temporal Context",
        "description": "Tests ability to understand time references.",
        "queries": [
            ("What did I mention earlier about my home?", ["Seattle"]),
            ("What did I talk about yesterday?", ["Alex", "Thompson", "Seattle"]),
        ],
    },
    "complex": {
        "name": "Complex Context",
        "description": "Tests handling of longer, more complex information.",
        "queries": [
            (
                "I need to remember the details about my new medical regimen. Can you help?",
                ["medication", "doctor", "treatment"],
            ),
        ],
        "preload": [
            "I just got back from the doctor. She said I need to take 25mg of Lisinopril every morning for my blood pressure, 10mg of Zolpidem before bed for sleep, and 500mg of Metformin with food twice daily for my blood sugar. I also need to schedule a follow-up in 3 months to check how the treatment is working."
        ],
    },
}


def compute_accuracy_cosine(response: str, expected_answers: list[str], embedder) -> float:
    """
    Compute cosine similarity between the LLM response and each expected answer.
    Returns the best similarity score as a float between 0 and 1.
    """
    if not expected_answers:
        return 0.0

    response_emb = embedder.encode(response, convert_to_tensor=True)
    expected_embs = embedder.encode(expected_answers, convert_to_tensor=True)

    cos_scores = util.cos_sim(response_emb, expected_embs)[0]  # shape: [n_expected]
    best_score = cos_scores.max().item()
    return best_score


def compute_keyword_accuracy(response: str, expected_answers: list[str]) -> float:
    """
    Compute accuracy based on presence of expected keywords in the response.
    Returns a score between 0 and 1.
    """
    if not expected_answers:
        return 0.0

    response_lower = response.lower()
    found_keywords = 0

    for keyword in expected_answers:
        if keyword.lower() in response_lower:
            found_keywords += 1

    return found_keywords / len(expected_answers)


def get_random_seed_message() -> str:
    """Returns a randomly selected seed message."""
    seed_messages = [
        "My name is Zephyr Quirk, and I live in the floating city of Aethelgard, which is said to exist above the clouds.",
        "I am Dr. Aris Thorne, and I specialize in chroniton particle research at the Temporal Dynamics Lab. My primary project is the stabilization of a K-flux anomaly.",
        "I contemplate the nature of subjective reality, and I believe that consciousness is a fundamental force, like gravity.",
        "My name is River Song, I enjoy hiking in the Appalachian mountains, and I have a rare condition that causes me to perceive time non-linearly.",
        "The old clock tower chimed 13 times, and I, known only as the 'Whisperer', knew that the crimson moon was about to rise, and that the ancient prophecy of the shadow beasts would begin. I live in a small cottage near the edge of the forgotten forest.",
        "I am known as the 'Chromatic Weaver', and I reside within the kaleidoscopic caverns of Xylos, a dimension where colors possess sentience.",
        "My name is Indigo Nightshade, and I am the last custodian of the 'Ethereal Codex', a book that contains the secrets of interdimensional travel.",
        "I am a 'Temporal Gardener', and my work involves cultivating the timelines of alternate realities to ensure their harmonious coexistence.",
        "My name is Solara Lux, and I am a cartographer of celestial anomalies, mapping the uncharted regions of the cosmic void.",
        "I am a 'Dream Architect', and I construct elaborate dreamscapes for those seeking to explore the depths of their subconscious.",
    ]
    return secrets.choice(seed_messages)


class SystemType(str, Enum):
    MEMORYWEAVE_HYBRID = "memoryweave_hybrid"
    MEMORYWEAVE_DEFAULT = "memoryweave_standard"
    MEMORYWEAVE_CHUNKED = "memoryweave_chunked"
    STANDARD_RAG = "standard_rag"


class SharedResources:
    """Container for shared LLM and embedding model instances."""

    def __init__(self):
        self.llm_provider = None
        self.embedding_model = None

    def clear(self):
        self.llm_provider = None
        self.embedding_model = None
        gc.collect()


# TODO: This is fixing the ability for hybrid memory to properly call the adapter cache building, which was not working correctly
# This is a temporary fix until the root cause is identified and fixed
def enhanced_build_cache(adapter):
    """Enhanced cache building that correctly finds memories in the hybrid store"""
    console.print("  - Building enhanced hybrid adapter cache")
    try:
        # Get all memories directly from the hybrid store
        memories = adapter.memory_store.get_all()

        if not memories:
            adapter._embeddings_matrix = np.zeros((0, 768))
            adapter._metadata_dict = []
            adapter._ids_list = []
            adapter._index_to_id_map = {}
            console.print("  - No memories found in hybrid store")
            return

        # Process each memory
        embeddings = []
        metadata_list = []
        ids_list = []
        adapter._index_to_id_map = {}

        for idx, memory in enumerate(memories):
            embeddings.append(memory.embedding)

            # Create metadata entry
            mdata = {}
            if hasattr(memory, "metadata") and memory.metadata:
                mdata = memory.metadata.copy()

            mdata["memory_id"] = idx
            mdata["original_id"] = memory.id
            metadata_list.append(mdata)

            ids_list.append(memory.id)
            adapter._index_to_id_map[idx] = memory.id

        # Store processed data
        adapter._embeddings_matrix = np.stack(embeddings) if embeddings else np.zeros((0, 768))
        adapter._metadata_dict = metadata_list
        adapter._ids_list = ids_list

        console.print(f"  - Enhanced hybrid cache built with {len(embeddings)} memories")

    except Exception as e:
        console.print(f"  - Error in enhanced hybrid cache: {e}")
        import traceback

        console.print(traceback.format_exc())
        adapter._embeddings_matrix = np.zeros((0, 768))
        adapter._metadata_dict = []
        adapter._ids_list = []
        adapter._index_to_id_map = {}


class UnifiedRetrievalBenchmark:
    """
    Benchmarks MemoryWeave retrieval strategies using shared resources.
    It creates a new system instance (using shared LLM and embedding model)
    for each strategy, runs a set of queries from each scenario, and aggregates results.

    Args:
        model_name (str): Name of the LLM model to use.
        systems_to_test (list[SystemType]): List of system types to test.
        output_dir (str): Directory to save benchmark results.
        scenarios (dict): Dictionary of scenarios to run.
        debug (bool): Enable debug logging for more detailed output.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        systems_to_test: list[SystemType] = None,
        output_dir: str = OUTPUT_DIR,
        scenarios: dict[str, Any] = DEFAULT_SCENARIOS,
        debug: bool = False,
    ):
        self.model_name = model_name
        if not os.path.exists(output_dir):
            logger.debug(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)

        self.output_dir = output_dir
        self.debug = debug
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # If no systems provided, test these strategies:
        self.systems_to_test = systems_to_test or [
            SystemType.MEMORYWEAVE_HYBRID,
            SystemType.MEMORYWEAVE_DEFAULT,
            SystemType.MEMORYWEAVE_CHUNKED,
            SystemType.STANDARD_RAG,
        ]

        # Enhanced scenarios with better testing for different memory capabilities

        # Load scenarios or use defaults
        self.scenarios = scenarios

        self.shared_resources = SharedResources()
        self.results = {
            "timestamp": self.timestamp,
            "model": self.model_name,
            "systems_tested": [s.value for s in self.systems_to_test],
            "scenarios_run": list(self.scenarios.keys()),
            "scenario_results": {},
            "system_metrics": {},
        }

    def initialize_shared_resources(self) -> bool:
        """Initialize shared LLM provider and embedding model."""
        console.print("[bold cyan]Initializing Shared Resources[/bold cyan]")
        try:
            # Import components on-demand

            embedding_model = DEFAULT_EMBEDDING_MODEL
            console.print(f"Loading embedding model: {embedding_model}")
            self.shared_resources.embedding_model = _get_embedder(
                model_name=embedding_model,
                device="mps",
            )

            console.print(f"Loading LLM: {self.model_name}")
            self.shared_resources.llm_provider = LLMProvider(
                model_name=self.model_name,
                device="mps",
            )

            initial_memory = self._get_memory_usage()
            console.print(f"Initial memory usage: {initial_memory:.2f} MB")
            console.print("[green]✓[/green] Shared resources initialized")
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to initialize shared resources: {e}")
            if self.debug:
                console.print(traceback.format_exc())
            return False
        return True

    def _get_memory_usage(self) -> float:
        """Get current process memory usage in MB"""
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def create_system(self, system_type: SystemType):
        """
        Instantiate a system of a given type and properly initialize with shared resources.
        This improved version ensures components are correctly initialized.
        """
        try:
            if system_type == SystemType.MEMORYWEAVE_HYBRID:
                return self._create_hybrid_system()
            elif system_type == SystemType.MEMORYWEAVE_CHUNKED:
                return self._create_chunked_system()
            elif system_type == SystemType.MEMORYWEAVE_DEFAULT:
                return self._create_default_system()
            elif system_type == SystemType.STANDARD_RAG:
                return self._create_standard_rag_system()
            else:
                console.print(f"[red]✗[/red] Unknown system type: {system_type}")
                return None

        except Exception as e:
            console.print(f"[red]✗[/red] Error creating system: {e}")
            if self.debug:
                console.print(traceback.format_exc())
            return None

    def _create_hybrid_system(self):
        """Creates and initializes a HybridMemoryWeaveAPI system."""
        console.print("Creating HybridMemoryWeaveAPI system...")

        system = HybridMemoryWeaveAPI(
            model_name=self.model_name,
            embedding_model_name="unused",  # will be replaced
            llm_provider=self.shared_resources.llm_provider,
            debug=self.debug,
        )
        _original_method = system.hybrid_memory_adapter._build_cache
        system.hybrid_memory_adapter._build_cache = lambda: enhanced_build_cache(
            system.hybrid_memory_adapter
        )
        system.embedding_model = self.shared_resources.embedding_model
        system.hybrid_memory_adapter.invalidate_cache()
        console.print("  [green]✓[/green] Enhanced hybrid adapter cache building applied")

        if hasattr(system, "configure_chunking"):
            system.configure_chunking(
                adaptive_chunk_threshold=800,
                max_chunks_per_memory=3,
                enable_auto_chunking=True,
            )

        if hasattr(system, "strategy") and hasattr(system, "hybrid_memory_adapter"):
            system.strategy.memory_store = system.hybrid_memory_adapter
            console.print("  [green]✓[/green] Connected strategy to hybrid adapter")

        if hasattr(system, "hybrid_memory_adapter"):
            system.hybrid_memory_adapter.invalidate_cache()

        if self.debug:
            self._debug_system_state(system, "HybridMemoryWeaveAPI")

        console.print("  [green]✓[/green] HybridMemoryWeave initialized")
        return system

    def _create_chunked_system(self):
        """Creates and initializes a ChunkedMemoryWeaveAPI system."""
        console.print("Creating ChunkedMemoryWeaveAPI system...")

        system = ChunkedMemoryWeaveAPI(
            model_name=self.model_name,
            embedding_model_name="unused",  # will be replaced
            llm_provider=self.shared_resources.llm_provider,
            debug=self.debug,
        )

        system.embedding_model = self.shared_resources.embedding_model

        if hasattr(system, "configure_chunking"):
            system.configure_chunking(
                auto_chunk_threshold=500,
                chunk_size=200,
                chunk_overlap=50,
                max_chunk_count=5,
            )

        if hasattr(system, "strategy") and hasattr(system, "chunked_memory_adapter"):
            system.strategy.memory_store = system.chunked_memory_adapter
            console.print("  [green]✓[/green] Connected strategy to chunked adapter")

        if hasattr(system, "chunked_memory_adapter"):
            system.chunked_memory_adapter.invalidate_cache()

        if self.debug:
            self._debug_system_state(system, "ChunkedMemoryWeaveAPI")

        console.print("  [green]✓[/green] ChunkedMemoryWeave initialized")
        return system

    def _create_default_system(self):
        """Creates and initializes a MemoryWeaveAPI system (standard)."""
        console.print("Creating MemoryWeaveAPI (Standard) system...")

        system = MemoryWeaveAPI(
            model_name=self.model_name,
            embedding_model_name="unused",
            llm_provider=self.shared_resources.llm_provider,
            debug=self.debug,
        )

        system.embedding_model = self.shared_resources.embedding_model

        if hasattr(system, "memory_store_adapter"):
            system.memory_store_adapter.invalidate_cache()

        if self.debug:
            self._debug_system_state(system, "MemoryWeaveAPI (Standard)")

        console.print("  [green]✓[/green] MemoryWeave Standard initialized")
        return system

    def _create_standard_rag_system(self):
        """Creates and initializes a MemoryWeaveAPI system (standard RAG)."""
        console.print("Creating MemoryWeaveAPI (RAG) system...")

        system = MemoryWeaveAPI(
            model_name=self.model_name,
            embedding_model_name="unused",
            llm_provider=self.shared_resources.llm_provider,
            enable_category_management=False,
            enable_personal_attributes=False,
            enable_semantic_coherence=False,
            enable_dynamic_thresholds=False,
            debug=self.debug,
        )

        system.embedding_model = self.shared_resources.embedding_model

        system.strategy.initialize(
            {
                "confidence_threshold": 0.1,
                "similarity_weight": 1.0,
                "associative_weight": 0.0,
                "temporal_weight": 0.0,
                "activation_weight": 0.0,
            }
        )

        if hasattr(system, "memory_store_adapter"):
            system.memory_store_adapter.invalidate_cache()

        if self.debug:
            self._debug_system_state(system, "MemoryWeaveAPI (RAG)")

        console.print("  [green]✓[/green] Standard RAG initialized")
        return system

    def _debug_system_state(self, system, system_name):
        """Print detailed debug information about system internal state."""
        console.print(f"  [blue]Debug information for {system_name}:[/blue]")

        # Check memory stores
        if hasattr(system, "memory_store"):
            console.print(f"  - memory_store: {type(system.memory_store).__name__}")
        if hasattr(system, "chunked_memory_store"):
            console.print(f"  - chunked_memory_store: {type(system.chunked_memory_store).__name__}")
        if hasattr(system, "hybrid_memory_store"):
            console.print(f"  - hybrid_memory_store: {type(system.hybrid_memory_store).__name__}")

        # Check adapters
        if hasattr(system, "memory_store_adapter"):
            console.print(f"  - memory_store_adapter: {type(system.memory_store_adapter).__name__}")
        if hasattr(system, "chunked_memory_adapter"):
            console.print(
                f"  - chunked_memory_adapter: {type(system.chunked_memory_adapter).__name__}"
            )
            # Check chunk embeddings shape
            if hasattr(system.chunked_memory_adapter, "chunk_embeddings"):
                shape = getattr(system.chunked_memory_adapter.chunk_embeddings, "shape", None)
                console.print(f"  - chunk_embeddings shape: {shape}")
        if hasattr(system, "hybrid_memory_adapter"):
            console.print(
                f"  - hybrid_memory_adapter: {type(system.hybrid_memory_adapter).__name__}"
            )

        # Check settings
        if hasattr(system, "auto_chunk_threshold"):
            console.print(f"  - auto_chunk_threshold: {system.auto_chunk_threshold}")
        if hasattr(system, "adaptive_chunk_threshold"):
            console.print(f"  - adaptive_chunk_threshold: {system.adaptive_chunk_threshold}")

        # Check strategy
        if hasattr(system, "strategy"):
            console.print(f"  - strategy: {type(system.strategy).__name__}")
            console.print(f"  - similarity_weight: {system.strategy.similarity_weight}")
            console.print(f"  - associative_weight: {system.strategy.associative_weight}")
            console.print(f"  - temporal_weight: {system.strategy.temporal_weight}")
            console.print(f"  - activation_weight: {system.strategy.activation_weight}")

    def verify_system_functionality(self, system, system_type):
        """Test system with a standard memory entry to verify it's working properly."""
        console.print("  Verifying system functionality...")

        # Test memory with standard content
        test_content = "This is a test memory containing information about Seattle and a person named Alex Thompson who lives there."

        try:
            # Add the test memory - CRITICAL FIX: Use the right method
            memory_id = None
            if system_type == SystemType.MEMORYWEAVE_HYBRID and hasattr(
                system, "hybrid_memory_store"
            ):
                # Add directly to the hybrid memory store
                embedding = system.embedding_model.encode(test_content)
                memory_id = system.hybrid_memory_store.add(embedding, test_content)
                # Force cache rebuild
                if hasattr(system, "hybrid_memory_adapter"):
                    system.hybrid_memory_adapter.invalidate_cache()
                console.print("  - Added test memory directly to hybrid store")
            else:
                # Use standard add_memory
                memory_id = system.add_memory(test_content)

            # Check system-specific behavior
            if system_type == SystemType.MEMORYWEAVE_CHUNKED:
                if hasattr(system, "chunked_memory_store") and hasattr(
                    system.chunked_memory_store, "get_chunks"
                ):
                    chunks = system.chunked_memory_store.get_chunks(memory_id)
                    console.print(f"  - Test memory chunked into {len(chunks)} parts")
                else:
                    console.print(
                        "  [yellow]- Chunking not verified (method not available)[/yellow]"
                    )

            elif system_type == SystemType.MEMORYWEAVE_HYBRID:
                if hasattr(system, "hybrid_memory_store") and hasattr(
                    system.hybrid_memory_store, "is_hybrid"
                ):
                    is_hybrid = system.hybrid_memory_store.is_hybrid(memory_id)
                    console.print(f"  - Test memory stored as hybrid: {is_hybrid}")
                else:
                    console.print(
                        "  [yellow]- Hybrid storage not verified (method not available)[/yellow]"
                    )

            # ADDITIONAL DIAGNOSTIC: Check memory counts in all stores
            if hasattr(system, "memory_store"):
                console.print(
                    f"  - Memory count in standard store: {len(system.memory_store.get_all())}"
                )
            if hasattr(system, "hybrid_memory_store"):
                console.print(
                    f"  - Memory count in hybrid store: {len(system.hybrid_memory_store.get_all())}"
                )

            # Test retrieval - CRITICAL FIX: Use hybrid_search if available
            if system_type == SystemType.MEMORYWEAVE_HYBRID and hasattr(
                system.hybrid_memory_adapter, "search_hybrid"
            ):
                query_embedding = system.embedding_model.encode("Where does Alex live?")
                result = system.hybrid_memory_adapter.search_hybrid(
                    query_embedding, limit=10, keywords=["alex", "live", "seattle"]
                )
                console.print(f"  - Direct hybrid search returned {len(result)} results")

            # Standard retrieval test
            result = system.retrieve("Where does Alex live?", top_k=1)
            if result and len(result) > 0:
                console.print(f"  - Test retrieval successful: {len(result)} results")
                return True
            else:
                console.print("  [yellow]- Test retrieval returned no results[/yellow]")

                # Last-ditch effort: try getting ALL memories and check manually
                if hasattr(system.strategy, "memory_store") and hasattr(
                    system.strategy.memory_store, "get_all"
                ):
                    memories = system.strategy.memory_store.get_all()
                    console.print(f"  - Strategy's memory store has {len(memories)} memories")

                return False

        except Exception as e:
            console.print(f"  [red]- System verification failed: {e}[/red]")
            if self.debug:
                import traceback

                console.print(traceback.format_exc())
            return False

    def run_benchmarks(self):
        """Run the benchmark over each scenario and each system type."""
        if not self.initialize_shared_resources():
            console.print("[red]Failed to initialize shared resources, aborting benchmark[/red]")
            return

        overall_metrics = {
            s.value: {"total_time": 0, "total_queries": 0, "accuracy": []}
            for s in self.systems_to_test
        }

        for scenario_key, scenario in self.scenarios.items():
            console.print(
                f"\n[bold cyan]Running scenario: {scenario_key}[/bold cyan] - {scenario.get('description', '')}"
            )
            scenario_result = {}

            # Get queries for this scenario
            queries = scenario.get("queries", [])

            # Check for preload data
            preload_data = scenario.get("preload", [])

            for system_type in self.systems_to_test:
                console.print(f"\n[bold]Testing system: {system_type.value}[/bold]")

                # Create and initialize system
                system = self.create_system(system_type)
                if system is None:
                    continue

                # Verify system functionality
                if not self.verify_system_functionality(system, system_type):
                    console.print(
                        "[yellow]System verification failed, results may not be reliable[/yellow]"
                    )

                # Preload scenario-specific data if any
                if preload_data:
                    console.print("  Preloading scenario data...")
                    for data in preload_data:
                        system.add_memory(data)

                # --- Seed conversation with a random message ---
                seed_message = get_random_seed_message()
                seed_response = system.chat(seed_message, max_new_tokens=150)
                console.print(f"[bold cyan]Seed turn message:[/bold cyan] {seed_message}")
                console.print(f"[bold cyan]Seed turn response:[/bold cyan] {seed_response}\n")
                # -----------------------------------------------------

                query_results = []
                total_time = 0
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(),
                    TimeElapsedColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task(
                        f"Processing {len(queries)} queries...", total=len(queries)
                    )
                    for query, expected in queries:
                        start = time.time()
                        response = system.chat(query, max_new_tokens=150)
                        console.print(f"[bold magenta]Query:[/bold magenta] {query}")
                        console.print(f"[bold green]Response:[/bold green] {response}\n")
                        elapsed = time.time() - start
                        total_time += elapsed

                        # Compute accuracy using both methods
                        cosine_accuracy = compute_accuracy_cosine(
                            response,
                            expected,
                            embedder=self.shared_resources.embedding_model,
                        )

                        keyword_accuracy = compute_keyword_accuracy(response, expected)

                        # Use combined accuracy score
                        combined_accuracy = (cosine_accuracy + keyword_accuracy) / 2

                        query_results.append(
                            {
                                "query": query,
                                "expected": expected,
                                "response": response,
                                "time": elapsed,
                                "cosine_accuracy": cosine_accuracy,
                                "keyword_accuracy": keyword_accuracy,
                                "combined_accuracy": combined_accuracy,
                            }
                        )
                        progress.advance(task)
                scenario_result[system_type.value] = {
                    "queries": query_results,
                    "stats": {
                        "total_time": total_time,
                        "total_queries": len(queries),
                        "accuracy": sum(r["combined_accuracy"] for r in query_results)
                        / len(query_results)
                        if query_results
                        else 0,
                    },
                }
                overall_metrics[system_type.value]["total_time"] += total_time
                overall_metrics[system_type.value]["total_queries"] += len(queries)
                overall_metrics[system_type.value]["accuracy"].append(
                    scenario_result[system_type.value]["stats"]["accuracy"]
                )
                # Clean up system instance
                del system
                gc.collect()
            self.results["scenario_results"][scenario_key] = scenario_result

        # Compute summary metrics across scenarios
        summary = {}
        for sys, metrics in overall_metrics.items():
            total_time = metrics["total_time"]
            total_queries = metrics["total_queries"]
            avg_time = total_time / total_queries if total_queries > 0 else 0
            avg_accuracy = (
                sum(metrics["accuracy"]) / len(metrics["accuracy"]) if metrics["accuracy"] else 0
            )
            summary[sys] = {"avg_time": avg_time, "avg_accuracy": avg_accuracy}
        self.results["system_metrics"] = summary

        self.display_results()
        self.save_results()
        self._create_comparative_charts()

    def display_results(self):
        """Display aggregated benchmark results in a rich table."""
        summary = self.results.get("system_metrics", {})
        table = Table(title="Unified Retrieval Benchmark Results")
        table.add_column("System", style="cyan")
        table.add_column("Avg Time (s)", style="yellow")
        table.add_column("Avg Accuracy", style="green")
        for sys, metrics in summary.items():
            table.add_row(sys, f"{metrics['avg_time']:.3f}", f"{metrics['avg_accuracy']:.2f}")
        console.print("\n[bold cyan]Overall Benchmark Results[/bold cyan]")
        console.print(table)

        # Display per-scenario comparison
        for scenario_key, scenario_data in self.results.get("scenario_results", {}).items():
            scenario_table = Table(title=f"Results for {scenario_key} scenario")
            scenario_table.add_column("System", style="cyan")
            scenario_table.add_column("Avg Time (s)", style="yellow")
            scenario_table.add_column("Accuracy", style="green")

            for system, data in scenario_data.items():
                stats = data.get("stats", {})
                avg_time = stats.get("total_time", 0) / stats.get("total_queries", 1)
                accuracy = stats.get("accuracy", 0)
                scenario_table.add_row(system, f"{avg_time:.3f}", f"{accuracy:.2f}")

            console.print(scenario_table)

    def save_results(self):
        """Save benchmark results to a JSON file."""
        filename = os.path.join(
            self.output_dir, f"unified_retrieval_benchmark_{self.timestamp}.json"
        )
        try:
            with open(filename, "w") as f:
                json.dump(self.results, f, indent=2)
            console.print(f"\n[bold green]Results saved to:[/bold green] {filename}")
        except Exception as e:
            console.print(f"[yellow]Failed to save results: {e}[/yellow]")

    def _create_comparative_charts(self):
        """Generate comparative charts for average query time and accuracy."""
        try:
            summary = self.results.get("system_metrics", {})
            systems = list(summary.keys())
            times = [summary[sys]["avg_time"] for sys in systems]
            accuracies = [summary[sys]["avg_accuracy"] for sys in systems]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Colors for the bars
            colors = ["skyblue", "lightgreen", "salmon", "khaki"]

            # Adjusting for better readability
            system_labels = [s.replace("memoryweave_", "") for s in systems]

            # Time chart
            ax1.bar(system_labels, times, color=colors[: len(systems)])
            ax1.set_title("Average Query Time")
            ax1.set_ylabel("Time (s)")
            ax1.tick_params(axis="x", rotation=45)

            # Add values on top of bars
            for i, v in enumerate(times):
                ax1.text(i, v + 0.1, f"{v:.2f}s", ha="center")

            # Accuracy chart
            ax2.bar(system_labels, accuracies, color=colors[: len(systems)])
            ax2.set_title("Average Accuracy")
            ax2.set_ylabel("Accuracy Score")
            ax2.set_ylim(0, 1.0)  # Set y-axis from 0 to 1
            ax2.tick_params(axis="x", rotation=45)

            # Add values on top of bars
            for i, v in enumerate(accuracies):
                ax2.text(i, v + 0.02, f"{v:.2f}", ha="center")

            plt.tight_layout()
            chart_path = os.path.join(
                self.output_dir, f"unified_retrieval_benchmark_chart_{self.timestamp}.png"
            )
            plt.savefig(chart_path)
            console.print(f"\n[bold green]Performance chart saved to:[/bold green] {chart_path}")

            # Try to display the chart (will work in environments that support it)
            # In your _create_comparative_charts method, remove or comment out plt.show():
            try:
                plt.savefig(chart_path)
                console.print(
                    f"\n[bold green]Performance chart saved to:[/bold green] {chart_path}"
                )
                # In headless mode, skip showing the chart
                # plt.show()  # Remove or comment out this line
            except Exception as e:
                console.print(f"[yellow]Failed to create charts: {e}[/yellow]")

        except Exception as e:
            console.print(f"[yellow]Failed to create charts: {e}[/yellow]")
            if self.debug:
                import traceback

                console.print(traceback.format_exc())


@click.command()
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    show_default=True,
    help=f"Name of the model to load (default: {DEFAULT_MODEL})",
)
@click.option(
    "--scenario",
    multiple=True,
    help="Specific scenarios to run (if not provided, all scenarios run)",
)
@click.option(
    "--system",
    "system_types",
    multiple=True,
    type=click.Choice([s.value for s in SystemType], case_sensitive=False),
    help="Specific systems to test (if not provided, all systems tested)",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging for more detailed output",
)
def main(model: str, scenario: list[str], system_types: list[str], debug: bool):
    """
    Unified Retrieval Benchmark: Compare MemoryWeave retrieval strategies under identical conditions.
    """
    if debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("memoryweave").setLevel(logging.DEBUG)
        console.print("[yellow]Debug logging enabled[/yellow]")

    # Filter scenarios if specified
    scenarios_to_run = {}
    if scenario:
        # Custom scenarios
        # You can define your scenarios here or keep them in a separate file
        all_scenarios = {
            "factual": {
                "name": "Factual Recall",
                "description": "Recall basic facts.",
                "queries": [
                    ("What is my name?", ["Alex", "Thompson"]),
                    ("Where do I live?", ["Seattle"]),
                ],
            },
            "temporal": {
                "name": "Temporal Context",
                "description": "Tests ability to understand time references.",
                "queries": [
                    ("What did I mention earlier about my home?", ["Seattle"]),
                    ("What did I say yesterday about my name?", ["Alex", "Thompson"]),
                ],
                "preload": [
                    "Yesterday I told you my name is Alex Thompson.",
                    "Earlier today I mentioned that I live in Seattle.",
                ],
            },
            "complex": {
                "name": "Complex Context",
                "description": "Tests handling of longer, more complex information.",
                "queries": [
                    (
                        "I need to remember the details about my medical regimen. Can you help?",
                        ["Lisinopril", "25mg", "Zolpidem", "10mg", "Metformin", "500mg"],
                    ),
                    (
                        "When do I need to schedule my follow-up appointment?",
                        ["3 months", "follow-up"],
                    ),
                ],
                "preload": [
                    "I just got back from the doctor. She said I need to take 25mg of Lisinopril every morning for my blood pressure, 10mg of Zolpidem before bed for sleep, and 500mg of Metformin with food twice daily for my blood sugar. I also need to schedule a follow-up in 3 months to check how the treatment is working."
                ],
            },
        }

        # Filter to only selected scenarios
        for key in scenario:
            if key in all_scenarios:
                scenarios_to_run[key] = all_scenarios[key]
    else:
        # No specific scenarios selected, use all
        scenarios_to_run = None

    # Filter systems if specified
    systems_to_test = None
    if system_types:
        systems_to_test = [SystemType(s) for s in system_types]

    benchmark = UnifiedRetrievalBenchmark(
        model_name=model, scenarios=scenarios_to_run, systems_to_test=systems_to_test, debug=debug
    )
    benchmark.run_benchmarks()
    """
    Unified Retrieval Benchmark: Compare MemoryWeave retrieval strategies under identical conditions.
    """
    if debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("memoryweave").setLevel(logging.DEBUG)
        console.print("[yellow]Debug logging enabled[/yellow]")

    # Filter scenarios if specified
    scenarios_to_run = {}
    if scenario:
        # Custom scenarios
        # You can define your scenarios here or keep them in a separate file
        all_scenarios = {
            "factual": {
                "name": "Factual Recall",
                "description": "Recall basic facts.",
                "queries": [
                    ("What is my name?", ["Alex", "Thompson"]),
                    ("Where do I live?", ["Seattle"]),
                ],
            },
            "temporal": {
                "name": "Temporal Context",
                "description": "Tests ability to understand time references.",
                "queries": [
                    ("What did I mention earlier about my home?", ["Seattle"]),
                    ("What did I say yesterday about my name?", ["Alex", "Thompson"]),
                ],
                "preload": [
                    "Yesterday I told you my name is Alex Thompson.",
                    "Earlier today I mentioned that I live in Seattle.",
                ],
            },
            "complex": {
                "name": "Complex Context",
                "description": "Tests handling of longer, more complex information.",
                "queries": [
                    (
                        "I need to remember the details about my medical regimen. Can you help?",
                        ["Lisinopril", "25mg", "Zolpidem", "10mg", "Metformin", "500mg"],
                    ),
                    (
                        "When do I need to schedule my follow-up appointment?",
                        ["3 months", "follow-up"],
                    ),
                ],
                "preload": [
                    "I just got back from the doctor. She said I need to take 25mg of Lisinopril every morning for my blood pressure, 10mg of Zolpidem before bed for sleep, and 500mg of Metformin with food twice daily for my blood sugar. I also need to schedule a follow-up in 3 months to check how the treatment is working."
                ],
            },
        }

        # Filter to only selected scenarios
        for key in scenario:
            if key in all_scenarios:
                scenarios_to_run[key] = all_scenarios[key]
    else:
        # No specific scenarios selected, use all
        scenarios_to_run = None

    # Filter systems if specified
    systems_to_test = None
    if system_types:
        systems_to_test = [SystemType(s) for s in system_types]

    benchmark = UnifiedRetrievalBenchmark(
        model_name=model, scenarios=scenarios_to_run, systems_to_test=systems_to_test, debug=debug
    )
    benchmark.run_benchmarks()


if __name__ == "__main__":
    main()
