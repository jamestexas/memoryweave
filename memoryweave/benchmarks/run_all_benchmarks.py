#!/usr/bin/env python3
"""
Unified Retrieval Benchmark for MemoryWeave

This script benchmarks MemoryWeave using shared resources to compare
retrieval strategies across different implementations:
    - hybrid (MemoryWeaveHybridAPI)
    - standard (MemoryWeaveAPI with standard configuration)
    - chunked (ChunkedMemoryWeaveAPI)
    - standard_rag (MemoryWeaveAPI configured as a standard RAG)

Each system uses the built‑in chat method that manages context automatically.
Results are aggregated across scenarios, displayed in a rich table, and
visualized with matplotlib charts.
"""

import gc
import json
import logging
import os
import platform
import secrets
import traceback
from datetime import datetime
from enum import Enum
from importlib.util import find_spec
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rich_click as click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.traceback import install

# Import sentence-transformers once at the beginning
from memoryweave.api.chunked_memory_weave import ChunkedMemoryWeaveAPI
from memoryweave.api.hybrid_memory_weave import HybridMemoryWeaveAPI
from memoryweave.api.llm_provider import LLMProvider
from memoryweave.api.memory_weave import MemoryWeaveAPI
from memoryweave.benchmarks.utils.perf_timer import PerformanceTimer, compute_accuracy
from memoryweave.components.retriever import _get_embedder

# Install rich traceback handler for better error reporting
install(show_locals=True)

# Disable warning messages from dependencies (huggingface transformers and the like)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

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

# Enhanced benchmark scenarios for better testing
ENHANCED_SCENARIOS = {
    "factual": {
        "name": "Factual Recall",
        "description": "Recall basic facts with clear patterns.",
        "queries": [
            ("What is my name?", ["Alex", "Thompson"]),
            ("Where do I live?", ["Seattle"]),
            ("What is my profession?", ["software", "engineer", "developer"]),
            ("What are my hobbies?", ["hiking", "photography", "reading"]),
        ],
        "preload": [
            "My name is Alex Thompson and I live in Seattle.",
            "I work as a software engineer at a tech company in downtown Seattle.",
            "My favorite hobbies are hiking in the mountains, photography, and reading science fiction books.",
        ],
    },
    "temporal": {
        "name": "Temporal Context",
        "description": "Tests ability to understand time references.",
        "queries": [
            ("What did I mention earlier about my home?", ["Seattle", "downtown"]),
            ("What did I tell you yesterday about my career?", ["software", "engineer", "tech"]),
            (
                "What activities did I say I enjoyed last week?",
                ["hiking", "mountains", "photography", "reading"],
            ),
            (
                "What was the medical information I shared with you recently?",
                ["blood pressure", "Lisinopril", "medication"],
            ),
        ],
        "preload": [
            "Yesterday I told you that I'm a software engineer with 8 years of experience in the tech industry.",
            "Last week I mentioned that I enjoy hiking in the Cascade mountains, landscape photography, and reading science fiction novels.",
            "Earlier today I mentioned that I recently moved to a new apartment in downtown Seattle with a view of Puget Sound.",
            "I just got back from the doctor. She said I need to take 25mg of Lisinopril every morning for my blood pressure.",
        ],
    },
    "complex": {
        "name": "Complex Context",
        "description": "Tests handling of longer, more complex information.",
        "queries": [
            (
                "Can you summarize my medical regimen?",
                [
                    "Lisinopril",
                    "25mg",
                    "morning",
                    "blood pressure",
                    "Zolpidem",
                    "10mg",
                    "sleep",
                    "Metformin",
                    "500mg",
                    "blood sugar",
                ],
            ),
            ("When do I need to schedule my follow-up appointment?", ["3 months", "follow-up"]),
            (
                "What side effects should I watch for with my medications?",
                ["dizziness", "drowsiness", "stomach", "cough"],
            ),
            ("How should I take the Metformin?", ["twice", "daily", "food"]),
        ],
        "preload": [
            "I just got back from the doctor. She said I need to take 25mg of Lisinopril every morning for my blood pressure, 10mg of Zolpidem before bed for sleep, and 500mg of Metformin with food twice daily for my blood sugar. I also need to schedule a follow-up in 3 months to check how the treatment is working.",
            "The doctor warned me to watch for side effects including dizziness from the Lisinopril, drowsiness from the Zolpidem, and stomach upset from the Metformin. She also mentioned that Lisinopril might cause a dry cough and that Metformin works best when I maintain a consistent meal schedule.",
        ],
    },
    "associative": {
        "name": "Associative Memory",
        "description": "Tests ability to make connections between related information.",
        "queries": [
            (
                "Tell me about my vacation plans.",
                ["Italy", "Rome", "Florence", "Venice", "September"],
            ),
            (
                "What dietary restrictions should I keep in mind for my trip?",
                ["gluten", "celiac", "pasta", "bread", "alternatives"],
            ),
            (
                "What languages will be useful for my upcoming travels?",
                ["Italian", "basic phrases", "translation app"],
            ),
        ],
        "preload": [
            "I'm planning a trip to Italy in September, visiting Rome, Florence, and Venice over two weeks.",
            "I have celiac disease, so I need to avoid gluten. I'm a bit worried about finding gluten-free alternatives to pasta and bread in Italy.",
            "I've been learning some basic Italian phrases, but I'll probably rely on a translation app for most conversations during my trip.",
        ],
    },
    "chunked_content": {
        "name": "Large Document Processing",
        "description": "Tests ability to handle and retrieve from large documents.",
        "queries": [
            (
                "What are the core components of the MemoryWeave architecture?",
                ["contextual fabric", "activation", "associative", "temporal"],
            ),
            (
                "How does MemoryWeave handle temporal context?",
                ["time references", "recency", "timestamps"],
            ),
            (
                "What retrieval strategies does MemoryWeave support?",
                ["similarity", "hybrid", "BM25", "vector", "contextual fabric"],
            ),
        ],
        "preload": [
            # Include the README.md content
            """MemoryWeave is an experimental approach to memory management for language models that uses a "contextual fabric" approach inspired by biological memory systems. Rather than traditional knowledge graph approaches with discrete nodes and edges, MemoryWeave focuses on capturing rich contextual signatures of information for improved long-context coherence in LLM conversations.

MemoryWeave implements several biologically-inspired memory management principles:
- Contextual Fabric: Memory traces capture rich contextual signatures rather than isolated facts
- Activation-Based Retrieval: Memory retrieval uses dynamic activation patterns similar to biological systems
- Episodic Structure: Memories maintain temporal relationships and episodic anchoring
- Non-Structured Memory: Works with raw LLM outputs without requiring structured formats

MemoryWeave uses a component-based architecture with several modular pieces that can be configured for different use cases:

Core Components:
1. Memory Management
   - MemoryManager: Coordinates memory operations and component interactions
   - MemoryStore: Stores embeddings, content, and metadata
   - VectorStore: Handles vector similarity search and indexing
   - ActivationManager: Manages memory activation levels

2. Retrieval Strategies
   - SimilarityRetrievalStrategy: Pure vector similarity-based retrieval
   - TemporalRetrievalStrategy: Time and recency-based retrieval
   - HybridRetrievalStrategy: Combines similarity and temporal retrieval
   - HybridBM25VectorStrategy: Combines lexical and semantic matching
   - ContextualFabricStrategy: Advanced multi-dimensional contextual retrieval

3. Query Processing
   - QueryAnalyzer: Analyzes and classifies query types
   - QueryAdapter: Adapts retrieval parameters based on query
   - PersonalAttributeManager: Extracts and manages personal attributes"""
        ],
    },
}


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


class UnifiedRetrievalBenchmark:
    """
    Benchmarks MemoryWeave retrieval strategies using shared resources.
    It creates a new system instance (using shared LLM and embedding model)
    for each strategy, runs a set of queries from each scenario, and aggregates results.

    Args:
        model_name (str): Name of the LLM model to use.
        systems_to_test (list[SystemType]): list of system types to test.
        output_dir (str): Directory to save benchmark results.
        scenarios (dict): dictionary of scenarios to run.
        debug (bool): Enable debug logging for more detailed output.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        systems_to_test: list[SystemType] = None,
        output_dir: str = OUTPUT_DIR,
        scenarios: dict[str, Any] = ENHANCED_SCENARIOS,
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

        # Load scenarios
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
        console.print(Panel("[bold cyan]Initializing Shared Resources[/bold cyan]", expand=False))
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
        if find_spec("psutil") is None:
            return 0.0

        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    def create_system(self, system_type: SystemType):
        """
        Instantiate a system of a given type and properly initialize with shared resources.
        This improved version ensures components are correctly initialized.
        """
        try:
            # First create the system instance
            system = self._create_system_instance(system_type)

            if system is None:
                return None

            # Then ensure proper initialization of memory storage
            self._initialize_memory_storage(system, system_type)

            # Force cache invalidation to ensure clean state
            self._invalidate_all_caches(system)

            # Verify the memory connections are properly established
            self._verify_memory_connections(system)

            return system

        except Exception as e:
            console.print(f"[red]✗[/red] Error creating system: {e}")
            if self.debug:
                console.print(traceback.format_exc())
            return None

    def _create_system_instance(self, system_type: SystemType):
        """Create the appropriate system instance based on type."""
        console.print(f"Creating {system_type.value} system...")

        if system_type == SystemType.MEMORYWEAVE_HYBRID:
            system = HybridMemoryWeaveAPI(
                model_name=self.model_name,
                embedding_model_name="unused",  # will be replaced
                llm_provider=self.shared_resources.llm_provider,
                debug=self.debug,
            )

            # Check if we need to add _build_cache method
            # if hasattr(system, "hybrid_memory_adapter"):
            #     added = add_build_cache_to_hybrid_adapter(system.hybrid_memory_adapter)
            #     if added:
            #         console.print("  [green]✓[/green] Added _build_cache method to hybrid adapter")

            console.print("  [green]✓[/green] Created HybridMemoryWeaveAPI")
            return system

        elif system_type == SystemType.MEMORYWEAVE_CHUNKED:
            system = ChunkedMemoryWeaveAPI(
                model_name=self.model_name,
                embedding_model_name="unused",  # will be replaced
                llm_provider=self.shared_resources.llm_provider,
                debug=self.debug,
            )
            console.print("  [green]✓[/green] Created ChunkedMemoryWeaveAPI")
            return system

        elif system_type == SystemType.MEMORYWEAVE_DEFAULT:
            system = MemoryWeaveAPI(
                model_name=self.model_name,
                embedding_model_name="unused",  # will be replaced
                llm_provider=self.shared_resources.llm_provider,
                debug=self.debug,
            )
            console.print("  [green]✓[/green] Created MemoryWeaveAPI (Standard)")
            return system

        elif system_type == SystemType.STANDARD_RAG:
            system = MemoryWeaveAPI(
                model_name=self.model_name,
                embedding_model_name="unused",  # will be replaced
                llm_provider=self.shared_resources.llm_provider,
                enable_category_management=False,
                enable_personal_attributes=False,
                enable_semantic_coherence=False,
                enable_dynamic_thresholds=False,
                debug=self.debug,
            )

            # Configure strategy for pure RAG
            if hasattr(system, "strategy"):
                system.strategy.initialize(
                    {
                        "confidence_threshold": 0.1,
                        "similarity_weight": 1.0,
                        "associative_weight": 0.0,
                        "temporal_weight": 0.0,
                        "activation_weight": 0.0,
                    }
                )
                console.print("  [green]✓[/green] Configured strategy for pure RAG")

            console.print("  [green]✓[/green] Created MemoryWeaveAPI (RAG)")
            return system

        else:
            console.print(f"[red]✗[/red] Unknown system type: {system_type}")
            return None

    def _initialize_memory_storage(self, system, system_type):
        """Initialize memory storage consistently across all systems."""
        # Replace the embedding model with the shared one
        if hasattr(system, "embedding_model"):
            system.embedding_model = self.shared_resources.embedding_model
            console.print("  [green]✓[/green] set shared embedding model")

        # For hybrid system, set up proper connections
        if system_type == SystemType.MEMORYWEAVE_HYBRID:
            if hasattr(system, "hybrid_memory_adapter") and hasattr(system, "hybrid_memory_store"):
                # Ensure adapter has reference to store
                system.hybrid_memory_adapter.memory_store = system.hybrid_memory_store

                # Ensure strategy connects to the right adapter
                if hasattr(system, "strategy"):
                    system.strategy.memory_store = system.hybrid_memory_adapter
                    console.print("  [green]✓[/green] Connected strategy to hybrid adapter")

                # Handle special case for hybrid store
                if hasattr(system.hybrid_memory_adapter, "hybrid_store"):
                    console.print("  [green]✓[/green] Verified hybrid_store reference exists")

        # For chunked system, set up proper connections
        elif system_type == SystemType.MEMORYWEAVE_CHUNKED:
            if hasattr(system, "chunked_memory_adapter") and hasattr(
                system, "chunked_memory_store"
            ):
                # Ensure adapter has reference to store
                system.chunked_memory_adapter.memory_store = system.chunked_memory_store

                # Ensure strategy connects to the right adapter
                if hasattr(system, "strategy"):
                    system.strategy.memory_store = system.chunked_memory_adapter
                    console.print("  [green]✓[/green] Connected strategy to chunked adapter")

        # Configure chunking for appropriate systems
        if system_type == SystemType.MEMORYWEAVE_CHUNKED and hasattr(system, "configure_chunking"):
            system.configure_chunking(
                auto_chunk_threshold=500,
                chunk_size=200,
                chunk_overlap=50,
                max_chunk_count=5,
            )
            console.print("  [green]✓[/green] Configured chunking parameters")

        if system_type == SystemType.MEMORYWEAVE_HYBRID and hasattr(system, "configure_chunking"):
            system.configure_chunking(
                adaptive_chunk_threshold=800,
                max_chunks_per_memory=3,
                enable_auto_chunking=True,
            )
            console.print("  [green]✓[/green] Configured hybrid chunking parameters")

    def _invalidate_all_caches(self, system):
        """Force cache invalidation on all adapters."""
        cache_invalidated = False

        # Try different adapters
        for adapter_name in [
            "memory_adapter",
            "memory_store_adapter",
            "hybrid_memory_adapter",
            "chunked_memory_adapter",
        ]:
            if hasattr(system, adapter_name):
                adapter = getattr(system, adapter_name)
                if hasattr(adapter, "invalidate_cache"):
                    adapter.invalidate_cache()
                    cache_invalidated = True
                    console.print(f"  [green]✓[/green] Invalidated cache on {adapter_name}")

        if not cache_invalidated:
            console.print(
                "  [yellow]- No caches invalidated (no invalidate_cache method found)[/yellow]"
            )

    def _verify_memory_connections(self, system):
        """Verify that all memory components are properly connected."""
        # Check strategy connection
        if hasattr(system, "strategy") and hasattr(system.strategy, "memory_store"):
            if system.strategy.memory_store is not None:
                console.print("  [green]✓[/green] Strategy has memory_store reference")
            else:
                console.print("  [yellow]- Strategy missing memory_store reference[/yellow]")

        # Check retrieval orchestrator
        if hasattr(system, "retrieval_orchestrator") and hasattr(
            system.retrieval_orchestrator, "strategy"
        ):
            if system.retrieval_orchestrator.strategy is not None:
                console.print("  [green]✓[/green] Retrieval orchestrator has strategy reference")
            else:
                console.print(
                    "  [yellow]- Retrieval orchestrator missing strategy reference[/yellow]"
                )

    def verify_system_functionality(self, system, system_type):
        """Test system with a standard memory entry to verify it's working properly."""
        console.print("  Verifying system functionality...")

        # Test memory with standard content
        test_content = "This is a test memory containing information about Seattle and a person named Alex Thompson who lives there."

        try:
            # Add the test memory with error handling
            memory_id = None
            try:
                memory_id = system.add_memory(test_content)
                console.print("  [green]✓[/green] Added test memory via standard method")
            except Exception as e:
                console.print(f"  [yellow]Error adding memory via standard method: {e}[/yellow]")
                # Try alternative method for hybrid systems
                if system_type == SystemType.MEMORYWEAVE_HYBRID:
                    try:
                        embedding = system.embedding_model.encode(test_content)
                        if hasattr(system, "hybrid_memory_store"):
                            memory_id = system.hybrid_memory_store.add(embedding, test_content)
                            if hasattr(system, "hybrid_memory_adapter"):
                                system.hybrid_memory_adapter.invalidate_cache()
                            console.print(
                                "  [green]✓[/green] Added test memory using direct hybrid store method"
                            )
                    except Exception as inner_e:
                        console.print(
                            f"  [red]Error with direct hybrid store method: {inner_e}[/red]"
                        )

            if not memory_id:
                console.print("  [red]Failed to add test memory[/red]")
                return False

            console.print(f"  [green]✓[/green] Added test memory with ID: {memory_id}")

            # Verify functionality through multiple methods
            success = self._verify_all_memory_access(system, memory_id, system_type)
            return success

        except Exception as e:
            console.print(f"  [red]System verification failed: {e}[/red]")
            if self.debug:
                console.print(traceback.format_exc())
            return False

    def _verify_all_memory_access(self, system, memory_id, system_type):
        """Run all verification methods and ensure at least one succeeds."""
        # Define verification methods
        verification_methods = [
            self._verify_standard_retrieval,
            self._verify_direct_memory_access,
            self._verify_keyword_retrieval,
        ]

        # Try each verification method
        success_count = 0
        for method in verification_methods:
            if method(system, memory_id, system_type):
                success_count += 1

        # Return true if at least one verification method succeeded
        console.print(
            f"  - {success_count} out of {len(verification_methods)} verification methods succeeded"
        )
        return success_count >= 1

    def _verify_standard_retrieval(self, system, memory_id, system_type):
        """Verify memory retrieval works using standard retrieve method."""
        try:
            console.print("  - Testing standard retrieval...")
            result = system.retrieve("Where does Alex live?", top_k=1)

            if result and len(result) > 0:
                console.print(
                    f"  [green]✓[/green] Standard retrieval returned {len(result)} results"
                )
                return True
            else:
                console.print("  [yellow]- Standard retrieval returned no results[/yellow]")
                return False
        except Exception as e:
            console.print(f"  [yellow]- Standard retrieval failed: {e}[/yellow]")
            return False

    def _verify_direct_memory_access(self, system, memory_id, system_type):
        """Verify direct memory access works."""
        try:
            console.print("  - Testing direct memory access...")

            # Try different memory stores depending on system type
            memory = None

            if system_type == SystemType.MEMORYWEAVE_HYBRID and hasattr(
                system, "hybrid_memory_store"
            ):
                memory = system.hybrid_memory_store.get(memory_id)
            elif system_type == SystemType.MEMORYWEAVE_CHUNKED and hasattr(
                system, "chunked_memory_store"
            ):
                memory = system.chunked_memory_store.get(memory_id)
            elif hasattr(system, "memory_store"):
                memory = system.memory_store.get(memory_id)

            if memory:
                console.print("  [green]✓[/green] Direct memory access succeeded")
                return True
            else:
                console.print("  [yellow]- Direct memory access returned no result[/yellow]")
                return False
        except Exception as e:
            console.print(f"  [yellow]- Direct memory access failed: {e}[/yellow]")
            return False

    def _verify_keyword_retrieval(self, system, memory_id, system_type):
        """Verify keyword-based retrieval works."""
        try:
            console.print("  - Testing keyword retrieval...")

            # Use search_by_keyword if available
            if hasattr(system, "search_by_keyword"):
                result = system.search_by_keyword("Seattle", limit=5)

                if result and len(result) > 0:
                    console.print(
                        f"  [green]✓[/green] Keyword retrieval returned {len(result)} results"
                    )
                    return True
                else:
                    console.print("  [yellow]- Keyword retrieval returned no results[/yellow]")

            # Try direct chat as fallback
            response = system.chat("Tell me what you know about Seattle", max_new_tokens=100)
            if "Seattle" in response and len(response) > 10:
                console.print(
                    "  [green]✓[/green] Chat retrieval succeeded with Seattle information"
                )
                return True

            return False
        except Exception as e:
            console.print(f"  [yellow]- Keyword retrieval failed: {e}[/yellow]")
            return False

    def run_benchmarks(self):
        """Run the benchmark over each scenario and each system type with improved timing and metrics."""
        console.rule("[bold blue]Starting Unified Retrieval Benchmark")

        if not self.initialize_shared_resources():
            console.print("[red]Failed to initialize shared resources, aborting benchmark[/red]")
            return

        # Create detailed result structure for timing analysis
        timing_data = {}
        benchmark_data = {}

        for scenario_key, scenario in self.scenarios.items():
            console.print(
                Panel(
                    f"[bold cyan]Scenario: {scenario_key}[/bold cyan]\n{scenario.get('description', '')}",
                    expand=False,
                )
            )
            scenario_result = {}
            benchmark_data[scenario_key] = {}
            timing_data[scenario_key] = {}

            # Get queries for this scenario
            queries = scenario.get("queries", [])

            # Check for preload data
            preload_data = scenario.get("preload", [])

            for system_type in self.systems_to_test:
                console.print(f"\n[bold]Testing system: {system_type.value}[/bold]")
                benchmark_data[scenario_key][system_type.value] = {
                    "query_results": [],
                    "system_verification": False,
                    "errors": [],
                }

                # Initialize performance timer for this system
                timer = PerformanceTimer()
                timing_data[scenario_key][system_type.value] = timer

                try:
                    # Create and initialize system
                    system = self.create_system(system_type)
                    if system is None:
                        benchmark_data[scenario_key][system_type.value]["errors"].append(
                            "Failed to create system"
                        )
                        continue

                    # Verify system functionality
                    verification_success = self.verify_system_functionality(system, system_type)
                    benchmark_data[scenario_key][system_type.value]["system_verification"] = (
                        verification_success
                    )

                    if not verification_success:
                        console.print(
                            "[yellow]System verification failed, results may not be reliable[/yellow]"
                        )
                        benchmark_data[scenario_key][system_type.value]["errors"].append(
                            "System verification failed"
                        )

                    # Preload scenario-specific data
                    self._preload_scenario_data(
                        system, preload_data, benchmark_data[scenario_key][system_type.value]
                    )

                    # Run seed conversation
                    self._run_seed_conversation(system)

                    # Run the actual queries with detailed timing
                    query_results = []
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
                            # Run query with detailed timing
                            query_data = self.run_detailed_query(
                                system=system, query=query, expected_answers=expected, timer=timer
                            )

                            query_results.append(query_data)
                            progress.advance(task)

                    # Calculate scenario results
                    successful_queries = [r for r in query_results if r["status"] == "success"]

                    avg_accuracy = (
                        sum(r["accuracy"]["combined"] for r in successful_queries)
                        / len(successful_queries)
                        if successful_queries
                        else 0
                    )

                    # Calculate average times for each phase
                    timing_summary = timer.get_summary()

                    scenario_result[system_type.value] = {
                        "queries": query_results,
                        "stats": {
                            "total_time": sum(r["time"] for r in successful_queries),
                            "total_queries": len(queries),
                            "successful_queries": len(successful_queries),
                            "accuracy": avg_accuracy,
                            "keyword_accuracy": (
                                sum(r["accuracy"]["keyword"] for r in successful_queries)
                                / len(successful_queries)
                                if successful_queries
                                else 0
                            ),
                            "cosine_accuracy": (
                                sum(r["accuracy"]["cosine"] for r in successful_queries)
                                / len(successful_queries)
                                if successful_queries
                                else 0
                            ),
                            "timing": timing_summary,
                        },
                    }

                    # Store detailed results for analysis
                    benchmark_data[scenario_key][system_type.value]["query_results"] = query_results

                    # Clean up system instance
                    del system
                    gc.collect()

                except Exception as e:
                    console.print(f"[red]Error testing system {system_type.value}: {e}[/red]")
                    benchmark_data[scenario_key][system_type.value]["errors"].append(
                        f"System testing error: {str(e)}"
                    )
                    if self.debug:
                        console.print(traceback.format_exc())

                self.results["scenario_results"][scenario_key] = scenario_result

        # Store the complete timing data
        self.results["timing_data"] = timing_data

        # Store the detailed benchmark data
        self.results["benchmark_data"] = benchmark_data

        # Generate summary metrics
        self._generate_summary_metrics()

        # Display the results
        self.display_results()
        self.save_results()
        self._create_comparative_charts()

    def _preload_scenario_data(self, system, preload_data, error_tracking):
        """Preload scenario data with proper error handling."""
        if not preload_data:
            return

        console.print("  Preloading scenario data...")
        for data in preload_data:
            try:
                system.add_memory(data)
            except Exception as e:
                console.print(f"  [yellow]Error preloading data: {e}[/yellow]")
                error_tracking["errors"].append(f"Error preloading data: {str(e)}")

                # Try direct method as fallback
                try:
                    embedding = system.embedding_model.encode(data)
                    if hasattr(system, "hybrid_memory_store"):
                        system.hybrid_memory_store.add(embedding, data)
                    elif hasattr(system, "chunked_memory_store"):
                        system.chunked_memory_store.add(embedding, data)
                    elif hasattr(system, "memory_store"):
                        system.memory_store.add(embedding, data)
                    console.print("  [green]✓[/green] Used fallback method for preloading")
                except Exception as inner_e:
                    console.print(f"  [red]Fallback preloading also failed: {inner_e}[/red]")

    def _run_seed_conversation(self, system):
        """Run a seed conversation to initialize the system."""
        seed_message = get_random_seed_message()
        try:
            seed_response = system.chat(seed_message, max_new_tokens=150)
            # Print in chronological order for easier reading
            console.print(f"[bold cyan]Seed message:[/bold cyan] {seed_message}")
            console.print(f"[bold cyan]Seed response:[/bold cyan] {seed_response}\n")
        except Exception as e:
            console.print(f"[yellow]Error during seed conversation: {e}[/yellow]")
            # Continue anyway - seed conversation is not critical

    def _generate_summary_metrics(self):
        """Generate summary metrics across scenarios."""
        summary = {}

        # Process each scenario
        for _scenario_key, scenario_data in self.results.get("scenario_results", {}).items():
            for system, data in scenario_data.items():
                if system not in summary:
                    summary[system] = {
                        "total_time": 0,
                        "total_queries": 0,
                        "retrieval_time": 0,
                        "inference_time": 0,
                        "accuracy": [],
                    }

                # Add stats
                stats = data.get("stats", {})
                summary[system]["total_time"] += stats.get("total_time", 0)
                summary[system]["total_queries"] += stats.get("total_queries", 0)

                # Add timing data if available
                timing = stats.get("timing", {})
                if "retrieval" in timing:
                    summary[system]["retrieval_time"] += timing["retrieval"].get("total", 0)
                if "inference" in timing:
                    summary[system]["inference_time"] += timing["inference"].get("total", 0)

                # Add accuracy
                summary[system]["accuracy"].append(stats.get("accuracy", 0))

        # Calculate averages
        for system, metrics in summary.items():
            total_queries = metrics["total_queries"]
            if total_queries > 0:
                metrics["avg_time"] = metrics["total_time"] / total_queries
                metrics["avg_retrieval_time"] = metrics["retrieval_time"] / total_queries
                metrics["avg_inference_time"] = metrics["inference_time"] / total_queries

            metrics["avg_accuracy"] = (
                sum(metrics["accuracy"]) / len(metrics["accuracy"]) if metrics["accuracy"] else 0
            )

        self.results["system_metrics"] = summary

    def display_results(self):
        """Display aggregated benchmark results in a rich table with enhanced timing details."""
        summary = self.results.get("system_metrics", {})

        # Overall results table
        table = Table(title="Unified Retrieval Benchmark Results")
        table.add_column("System")
        table.add_column("Avg Time (s)", header_style="yellow")
        table.add_column("Retrieval (s)", header_style="yellow")
        table.add_column("Inference (s)", header_style="yellow")
        table.add_column("Avg Accuracy", header_style="green")
        table.add_column("Total Queries", header_style="blue")

        for sys, metrics in summary.items():
            table.add_row(
                sys,
                f"{metrics.get('avg_time', 0):.3f}",
                f"{metrics.get('avg_retrieval_time', 0):.3f}",
                f"{metrics.get('avg_inference_time', 0):.3f}",
                f"{metrics.get('avg_accuracy', 0):.2f}",
                str(metrics.get("total_queries", 0)),
            )

        console.print("\n[bold cyan]Overall Benchmark Results[/bold cyan]")
        console.print(table)

        # Display per-scenario comparison
        for scenario_key, scenario_data in self.results.get("scenario_results", {}).items():
            scenario_table = Table(title=f"Results for {scenario_key} scenario")
            scenario_table.add_column("System", header_style="cyan")
            scenario_table.add_column("Total (s)", header_style="yellow")
            scenario_table.add_column("Retrieval (s)", header_style="yellow")
            scenario_table.add_column("Inference (s)", header_style="yellow")
            scenario_table.add_column("Accuracy", header_style="green")
            scenario_table.add_column("Keyword Acc", header_style="green")
            scenario_table.add_column("Success Rate", header_style="blue")

            for system, data in scenario_data.items():
                stats = data.get("stats", {})
                timing = stats.get("timing", {})

                # Calculate average times
                avg_time = stats.get("total_time", 0) / stats.get("total_queries", 1)

                retrieval_time = 0
                if "retrieval" in timing:
                    retrieval_time = timing["retrieval"].get("avg", 0)

                inference_time = 0
                if "inference" in timing:
                    inference_time = timing["inference"].get("avg", 0)

                accuracy = stats.get("accuracy", 0)
                keyword_acc = stats.get("keyword_accuracy", 0)
                success_rate = (
                    f"{stats.get('successful_queries', 0)}/{stats.get('total_queries', 0)}"
                )

                scenario_table.add_row(
                    system,
                    f"{avg_time:.3f}",
                    f"{retrieval_time:.3f}",
                    f"{inference_time:.3f}",
                    f"{accuracy:.2f}",
                    f"{keyword_acc:.2f}",
                    success_rate,
                )

            console.print(scenario_table)

        # Display any errors encountered
        errors_found = False
        for scenario_key, scenario_data in self.results.get("benchmark_data", {}).items():
            for system, data in scenario_data.items():
                if data.get("errors"):
                    if not errors_found:
                        console.print("\n[bold red]Errors Encountered[/bold red]")
                        errors_found = True
                    console.print(f"[yellow]{scenario_key} - {system}:[/yellow]")
                    for error in data["errors"]:
                        console.print(f"  - {error}")

    def save_results(self):
        """Save benchmark results to a JSON file with enhanced metadata."""
        # Add version information and execution details
        self.results["metadata"] = {
            "version": "2.0",
            "execution_time": datetime.now().isoformat(),
            "memory_usage_mb": self._get_memory_usage(),
            "platform": self._get_platform_info(),
        }

        filename = os.path.join(
            self.output_dir, f"unified_retrieval_benchmark_{self.timestamp}.json"
        )
        try:
            with open(filename, "w") as f:
                json.dump(self.results, f, indent=2)
            console.print(f"\n[bold green]Results saved to:[/bold green] {filename}")
        except Exception as e:
            console.print(f"[yellow]Failed to save results: {e}[/yellow]")

        # Also save a summary for easy review
        summary_filename = os.path.join(self.output_dir, f"benchmark_summary_{self.timestamp}.txt")
        try:
            with open(summary_filename, "w") as f:
                f.write("MemoryWeave Benchmark Summary\n")
                f.write(f"Date: {datetime.now().isoformat()}\n")
                f.write(f"Model: {self.model_name}\n\n")

                f.write("Overall Results:\n")
                for sys, metrics in self.results.get("system_metrics", {}).items():
                    f.write(f"  {sys}:\n")
                    f.write(f"    Average Time: {metrics['avg_time']:.3f}s\n")
                    f.write(f"    Average Accuracy: {metrics['avg_accuracy']:.2f}\n")
                    f.write(f"    Total Queries: {metrics['total_queries']}\n\n")

            console.print(f"\n[bold green]Summary saved to:[/bold green] {summary_filename}")
        except Exception as e:
            console.print(f"[yellow]Failed to save summary: {e}[/yellow]")

    def _create_comparative_charts(self):
        """Generate comprehensive comparative charts with detailed timing information."""
        try:
            summary = self.results.get("system_metrics", {})
            systems = list(summary.keys())

            if not systems:
                console.print("[yellow]No data available for charts[/yellow]")
                return

            # Extract timing data
            times = [summary[sys].get("avg_time", 0) for sys in systems]
            retrieval_times = [summary[sys].get("avg_retrieval_time", 0) for sys in systems]
            inference_times = [summary[sys].get("avg_inference_time", 0) for sys in systems]
            accuracies = [summary[sys].get("avg_accuracy", 0) for sys in systems]

            # Create a comprehensive figure for multiple metrics
            fig = plt.figure(figsize=(16, 10))

            # Add title
            plt.suptitle("MemoryWeave Unified Retrieval Benchmark", fontsize=16, fontweight="bold")
            plt.subplots_adjust(top=0.88, wspace=0.3, hspace=0.4)

            # Accuracy subplot
            ax1 = plt.subplot(2, 2, 1)
            bars1 = ax1.bar(systems, accuracies, color="#5DA5DA")
            ax1.set_title("Average Accuracy", fontsize=12)
            ax1.set_ylabel("Accuracy Score")
            ax1.set_ylim(0, 1.0)
            ax1.tick_params(axis="x", rotation=45, labelsize=10)

            # Add values on top of bars
            for i, v in enumerate(accuracies):
                ax1.text(i, v + 0.03, f"{v:.2f}", ha="center")

            # Time comparison (total, retrieval, inference)
            ax2 = plt.subplot(2, 2, 2)

            # Set bar width
            bar_width = 0.25
            r1 = np.arange(len(systems))
            r2 = [x + bar_width for x in r1]
            r3 = [x + bar_width for x in r2]

            # Create grouped bars
            bars2 = ax2.bar(r1, times, width=bar_width, label="Total", color="#5DA5DA")
            bars3 = ax2.bar(
                r2, retrieval_times, width=bar_width, label="Retrieval", color="#FAA43A"
            )
            bars4 = ax2.bar(
                r3, inference_times, width=bar_width, label="Inference", color="#60BD68"
            )

            # Add labels and legend
            ax2.set_title("Performance Breakdown", fontsize=12)
            ax2.set_ylabel("Time (s)")
            ax2.set_xticks([r + bar_width for r in range(len(systems))])
            ax2.set_xticklabels(systems, rotation=45, ha="right", fontsize=10)
            ax2.legend()

            # Stacked percentage chart
            ax3 = plt.subplot(2, 2, 3)

            # Calculate percentages
            percentages = []
            for i in range(len(systems)):
                total = max(times[i], 0.001)  # Avoid division by zero
                retrieval_pct = min(retrieval_times[i] / total, 1.0) * 100
                inference_pct = min(inference_times[i] / total, 1.0) * 100
                other_pct = max(0, 100 - retrieval_pct - inference_pct)
                percentages.append([retrieval_pct, inference_pct, other_pct])

            percentages = np.array(percentages)

            # Create stacked bars
            ax3.bar(systems, percentages[:, 0], label="Retrieval", color="#FAA43A")
            ax3.bar(
                systems,
                percentages[:, 1],
                bottom=percentages[:, 0],
                label="Inference",
                color="#60BD68",
            )
            ax3.bar(
                systems,
                percentages[:, 2],
                bottom=percentages[:, 0] + percentages[:, 1],
                label="Other",
                color="#F17CB0",
            )

            # Add labels and legend
            ax3.set_title("Time Percentage Breakdown", fontsize=12)
            ax3.set_ylabel("Percentage (%)")
            ax3.set_ylim(0, 100)
            ax3.set_yticks(range(0, 101, 20))
            ax3.tick_params(axis="x", rotation=45, labelsize=10)
            ax3.legend()

            # Per-scenario performance
            ax4 = plt.subplot(2, 2, 4)

            # Process data for grouped bar chart
            scenario_keys = list(self.results.get("scenario_results", {}).keys())
            width = 0.8 / len(systems)  # width of the bars
            x = np.arange(len(scenario_keys))

            # Plot bars for each system
            for i, system in enumerate(systems):
                system_scores = []
                for scenario in scenario_keys:
                    if scenario in self.results.get("scenario_results", {}):
                        if system in self.results["scenario_results"][scenario]:
                            score = (
                                self.results["scenario_results"][scenario][system]
                                .get("stats", {})
                                .get("accuracy", 0)
                            )
                            system_scores.append(score)
                        else:
                            system_scores.append(0)
                    else:
                        system_scores.append(0)

                # Calculate bar positions
                positions = x + (i - len(systems) / 2 + 0.5) * width

                ax4.bar(
                    positions, system_scores, width * 0.9, label=system.replace("memoryweave_", "")
                )

            # Add labels and legend
            ax4.set_title("Accuracy by Scenario", fontsize=12)
            ax4.set_xticks(x)
            ax4.set_xticklabels(scenario_keys, rotation=45, ha="right", fontsize=10)
            ax4.set_ylabel("Accuracy Score")
            ax4.set_ylim(0, 1.0)
            ax4.legend(loc="upper right", fontsize=8)

            plt.tight_layout()
            chart_path = os.path.join(
                self.output_dir, f"unified_retrieval_benchmark_chart_{self.timestamp}.png"
            )
            plt.savefig(chart_path, dpi=150)
            console.print(f"\n[bold green]Performance chart saved to:[/bold green] {chart_path}")

        except Exception as e:
            console.print(f"[yellow]Failed to create charts: {e}[/yellow]")
            if self.debug:
                console.print(traceback.format_exc())

        def _get_platform_info(self):
            """Get information about the platform for more detailed reporting."""
            import platform

            info = {
                "python_version": platform.python_version(),
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            }

            # Try to get more detailed hardware info
            try:
                if find_spec("psutil") is not None:
                    import psutil

                    mem = psutil.virtual_memory()
                    info["memory_total"] = mem.total / (1024 * 1024 * 1024)  # GB
                    info["memory_available"] = mem.available / (1024 * 1024 * 1024)  # GB
                    info["cpu_count"] = psutil.cpu_count()
            except Exception:  # noqa: S110
                pass

            return info

    # Add this method to the UnifiedRetrievalBenchmark class
    def run_detailed_query(
        self, system, query: str, expected_answers: list[str], timer: PerformanceTimer
    ) -> dict[str, Any]:
        """
        Run a query with detailed performance timing.

        Args:
            system: The system to test
            query: The query to run
            expected_answers: Expected answers for accuracy calculation
            timer: Performance timer instance

        Returns:
            dictionary with query results and performance metrics
        """
        query_data = {
            "query": query,
            "expected": expected_answers,
            "status": "success",
            "timings": {},
        }

        try:
            # Start timing total operation
            timer.start("total")

            # Preparation phase (embedding computation, etc.)
            timer.start("preparation")
            # This varies by system, but generally involves:
            query_info = None
            query_embedding = None

            if hasattr(system, "_analyze_query") and hasattr(system, "_compute_embedding"):
                # Start with query analysis if available
                query_info = system._analyze_query(query)
                query_embedding = system._compute_embedding(query)
            timer.stop("preparation")

            # Retrieval phase
            timer.start("retrieval")
            if hasattr(system, "retrieve") and query_embedding is not None:
                # Direct retrieval if we have embedding and method
                retrieved_memories = system.retrieve(
                    query=query, query_embedding=query_embedding, top_k=10
                )
            else:
                # Skip direct measurement if we can't separate it
                retrieved_memories = []
            timer.stop("retrieval")

            # Memory operations (update activations, etc.)
            timer.start("memory_ops")
            # This would normally happen inside the chat method
            # We're just measuring the time separately
            timer.stop("memory_ops")

            # Inference phase (LLM generation)
            timer.start("inference")
            response = system.chat(query, max_new_tokens=150)
            timer.stop("inference")

            # Stop timing total operation
            total_time = timer.stop("total")

            # Print the query and response if debugging
            console.print(f"[bold magenta]Query:[/bold magenta] {query}")
            console.print(f"[bold green]Response:[/bold green] {response}\n")

            # Calculate accuracy
            accuracy_results = compute_accuracy(
                response,
                expected_answers,
                embedder=self.shared_resources.embedding_model,
            )

            # Store results
            query_data.update(
                {
                    "response": response,
                    "time": total_time,
                    "timings": {
                        "preparation": timer.get_average("preparation"),
                        "retrieval": timer.get_average("retrieval"),
                        "inference": timer.get_average("inference"),
                        "memory_ops": timer.get_average("memory_ops"),
                        "total": total_time,
                    },
                    "accuracy": accuracy_results,
                    "retrieved_count": len(retrieved_memories) if retrieved_memories else 0,
                }
            )

        except Exception as e:
            console.print(f"[red]Error processing query: {e}[/red]")
            query_data.update(
                {
                    "status": "error",
                    "error": str(e),
                    "time": timer.stop("total") if "total" in timer._start_times else 0,
                }
            )
            if self.debug:
                console.print(traceback.format_exc())

        return query_data

    def _get_platform_info(self):
        """Get information about the platform for more detailed reporting."""

        info = {
            "python_version": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        }

        # Try to get more detailed hardware info
        try:
            if find_spec("psutil") is not None:
                import psutil

                mem = psutil.virtual_memory()
                info["memory_total"] = mem.total / (1024 * 1024 * 1024)  # GB
                info["memory_available"] = mem.available / (1024 * 1024 * 1024)  # GB
                info["cpu_count"] = psutil.cpu_count()
        except Exception:
            pass

        return info


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
@click.option(
    "--output-dir",
    default=OUTPUT_DIR,
    help=f"Directory to save benchmark results (default: {OUTPUT_DIR})",
)
@click.option(
    "--seed",
    is_flag=True,
    help="Use consistent random seed for reproducibility",
)
def main(
    model: str,
    scenario: list[str],
    system_types: list[str],
    debug: bool,
    output_dir: str,
    seed: bool,
):
    """
    Unified Retrieval Benchmark: Compare MemoryWeave retrieval strategies with enhanced metrics.

    This benchmark runs multiple retrieval strategies on the same test scenarios and compares
    their performance in terms of accuracy, speed, and reliability.
    """
    if seed:
        # set consistent seeds for reproducibility
        import random

        random.seed(42)
        np.random.seed(42)
        try:
            import torch

            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
            console.print("[yellow]Random seeds fixed for reproducibility[/yellow]")
        except ImportError:
            pass

    if debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("memoryweave").setLevel(logging.DEBUG)
        console.print("[yellow]Debug logging enabled[/yellow]")

    # Filter scenarios if specified
    scenarios_to_run = {}
    if scenario:
        # Filter to only selected scenarios
        for key in scenario:
            if key in ENHANCED_SCENARIOS:
                scenarios_to_run[key] = ENHANCED_SCENARIOS[key]
            else:
                console.print(f"[yellow]Warning: Scenario '{key}' not found, skipping[/yellow]")
    else:
        # No specific scenarios selected, use all
        scenarios_to_run = ENHANCED_SCENARIOS

    # Filter systems if specified
    systems_to_test = None
    if system_types:
        systems_to_test = [SystemType(s) for s in system_types]

    # Print banner
    console.print(
        Panel(
            "[bold cyan]MemoryWeave Unified Retrieval Benchmark[/bold cyan]\n"
            f"Model: {model}\n"
            f"Scenarios: {', '.join(scenarios_to_run.keys())}\n"
            f"Systems: {', '.join([s.value for s in (systems_to_test or list(SystemType))])}\n"
            f"Output directory: {output_dir}",
            expand=False,
        )
    )

    # Run the benchmark
    benchmark = UnifiedRetrievalBenchmark(
        model_name=model,
        scenarios=scenarios_to_run,
        systems_to_test=systems_to_test,
        debug=debug,
        output_dir=output_dir,
    )

    benchmark.run_benchmarks()


if __name__ == "__main__":
    main()
