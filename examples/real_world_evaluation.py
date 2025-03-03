#!/usr/bin/env python
"""
MemoryWeave Real-World Evaluation Script

This script measures the performance of MemoryWeave by comparing different configurations
against real-world use cases. It offers a standardized way to benchmark and visualize
how well the architecture performs against baseline approaches.

Features:
- Uses real sentence embeddings for semantic similarity
- Implements comprehensive evaluation metrics
- Tests against diverse query types and contexts
- Provides detailed visualization of results

Usage:
    python -m examples.real_world_evaluation --dataset datasets/enhanced_evaluation.json --save-path results.json
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import rich_click as click
from rich.console import Console, Group
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

# set up rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger("memoryweave")
console = Console()

# Attempt to import MemoryWeave components
try:
    from memoryweave.components.memory_manager import MemoryManager
    from memoryweave.components.retrieval_strategies import (
        HybridRetrievalStrategy,
        SimilarityRetrievalStrategy,
        TemporalRetrievalStrategy,
        TwoStageRetrievalStrategy,
    )
    from memoryweave.components.retriever import Retriever
    from memoryweave.core.contextual_memory import ContextualMemory
except ImportError as e:
    logger.error(f"[bold red]Error importing MemoryWeave components: {e}[/bold red]")
    raise

# Try to import sentence_transformers; use a mock if not available
try:
    import logging as python_logging

    from sentence_transformers import SentenceTransformer

    # Suppress sentence-transformers progress bars by setting higher log level
    for logger_name in ["sentence_transformers", "transformers"]:
        python_logging.getLogger(logger_name).setLevel(python_logging.ERROR)

    # Use a model with 384 dimensions to match the expected size
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="mps")
    EMBEDDING_DIM = 384  # This model outputs 384-dimensional embeddings

    logger.info("[green]Using SentenceTransformer for embeddings (dim=%d)[/green]" % EMBEDDING_DIM)
    USE_REAL_EMBEDDINGS = True
except ImportError:
    logger.warning("[yellow]SentenceTransformer not available, using mock embedding model[/yellow]")
    logger.warning(
        "[bold yellow]WARNING: Mock embeddings will not represent semantic similarity accurately[/bold yellow]"
    )
    logger.warning(
        "[yellow]Consider installing sentence-transformers package for accurate evaluation[/yellow]"
    )
    USE_REAL_EMBEDDINGS = False
    EMBEDDING_DIM = 384  # Use this dimension for consistency

    class MockEmbeddingModel:
        def __init__(self, embedding_dim=EMBEDDING_DIM):
            self.embedding_dim = embedding_dim
            self.call_count = 0
            self.cached_embeddings = {}  # Cache embeddings for consistency

        def encode(self, text, batch_size=32):
            """Create a deterministic but unique embedding for any text."""
            self.call_count += 1
            if isinstance(text, list):
                return np.array([self._encode_single(t) for t in text])
            return self._encode_single(text)

        def _encode_single(self, text):
            """Create a single embedding with basic semantic properties."""
            # Check cache first
            if text in self.cached_embeddings:
                return self.cached_embeddings[text]

            # Create a deterministic embedding based on text content
            if not text:
                embedding = np.zeros(self.embedding_dim)
            else:
                # Try to create embeddings that have some semantic properties
                # Words that share tokens will have somewhat more similar embeddings
                words = text.lower().split()
                embedding = np.zeros(self.embedding_dim)

                for i, word in enumerate(words):
                    # Use word hash to seed a section of the embedding
                    word_hash = hash(word) % 1000000
                    np.random.seed(word_hash)

                    # Create a component for this word
                    word_vec = np.random.randn(self.embedding_dim)

                    # Add to overall embedding with diminishing importance for later words
                    embedding += word_vec * (1.0 / (i + 1))

            # Normalize
            if np.linalg.norm(embedding) > 0:
                embedding = embedding / np.linalg.norm(embedding)

            # Cache for future use
            self.cached_embeddings[text] = embedding
            return embedding

    embedding_model = MockEmbeddingModel()


@dataclass
class MemoryEntry:
    """Represents a single memory entry."""

    text: str
    metadata: dict[str, Any]
    embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.embedding is None:
            self.embedding = embedding_model.encode(self.text)


@dataclass
class EvaluationConfig:
    """Configuration for an evaluation run."""

    name: str
    description: str
    retriever_type: str  # "baseline", "contextual", "component"

    # Threshold settings
    confidence_threshold: float = 0.3

    # Feature flags
    use_art_clustering: bool = False
    semantic_coherence_check: bool = False
    adaptive_retrieval: bool = False
    use_two_stage_retrieval: bool = False
    query_type_adaptation: bool = False
    dynamic_threshold_adjustment: bool = False
    memory_decay_enabled: bool = False

    # Two-stage retrieval parameters
    first_stage_k: int = 20
    first_stage_threshold_factor: float = 0.7

    # System parameters
    embedding_dim: int = 384  # Match the SentenceTransformer output dimension
    max_memories: int = 1000
    result_count: int = 10  # Number of results to retrieve

    def to_dict(self) -> dict[str, Any]:
        """Convert config to a dictionary."""
        return asdict(self)


@dataclass
class EvaluationResults:
    """Results from an evaluation run."""

    config: EvaluationConfig

    # Performance metrics
    setup_time: float
    query_times: list[float] = field(default_factory=list)
    memory_usage: float = 0.0

    # Retrieval metrics
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    semantic_similarity: float = 0.0
    retrieval_counts: list[int] = field(default_factory=list)

    # Additional metrics
    by_query_type: dict[str, dict[str, float]] = field(default_factory=dict)
    error_rate: float = 0.0

    # Derived metrics
    @property
    def avg_query_time(self) -> float:
        """Calculate average query time."""
        return np.mean(self.query_times) if self.query_times else 0.0

    @property
    def avg_retrieval_count(self) -> float:
        """Calculate average number of retrieved items."""
        return np.mean(self.retrieval_counts) if self.retrieval_counts else 0.0

    @property
    def overall_score(self) -> float:
        """Calculate an overall quality score (F1 + semantic similarity)."""
        # Combined score that balances precision/recall with semantic quality
        if self.semantic_similarity > 0:
            return 0.6 * self.f1_score + 0.4 * self.semantic_similarity
        return self.f1_score

    def to_dict(self) -> dict[str, Any]:
        """Convert results to a dictionary."""
        return {
            "config": self.config.to_dict(),
            "setup_time": self.setup_time,
            "avg_query_time": self.avg_query_time,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "semantic_similarity": self.semantic_similarity,
            "overall_score": self.overall_score,
            "memory_usage": self.memory_usage,
            "avg_retrieval_count": self.avg_retrieval_count,
            "by_query_type": self.by_query_type,
            "error_rate": self.error_rate,
        }


class RealWorldEvaluator:
    """Evaluates MemoryWeave on real-world use cases."""

    def __init__(self, configs: list[EvaluationConfig]):
        """Initialize with configurations to test."""
        self.configs = configs
        self.memories: list[MemoryEntry] = []
        self.test_queries: list[dict[str, Any]] = []
        self.results: list[EvaluationResults] = []

    def load_dataset(self, dataset_path: str) -> None:
        """Load evaluation dataset from file."""
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            logger.error(f"[bold red]Dataset file not found: {dataset_path}[/bold red]")
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        try:
            with open(dataset_path) as f:
                data = json.load(f)

            # Check if this is new format or old format
            if isinstance(data, dict) and "queries" in data:
                self.test_queries = data["queries"]
            elif isinstance(data, list):
                self.test_queries = data
            else:
                raise ValueError("Dataset format not recognized")

            logger.info(f"[green]Loaded dataset with {len(self.test_queries)} queries[/green]")

            # Generate memories from expected answers
            self._generate_memories_from_queries()

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"[bold red]Error parsing dataset: {e}[/bold red]")
            raise

    def _generate_memories_from_queries(self) -> None:
        """Generate memory entries from test queries."""
        # Clear existing memories
        self.memories = []

        # Create memory entries from expected answers in queries
        for i, query in enumerate(self.test_queries):
            # Check for new format first (expected_answers)
            if "expected_answers" in query and query["expected_answers"]:
                for j, answer_text in enumerate(query["expected_answers"]):
                    # Create metadata that's useful for retrieval
                    metadata = {
                        "memory_id": i * 100 + j,  # Create a unique ID per answer
                        "category": query.get("category", "unknown"),
                        "context": query.get("context", ""),
                        "difficulty": query.get("difficulty", "basic"),
                        "content": answer_text,
                        "timestamp": time.time() - (i * 600),  # Space out timestamps
                        "importance": np.random.uniform(0.5, 1.0),
                    }

                    # Add memory
                    memory = MemoryEntry(text=answer_text, metadata=metadata)
                    self.memories.append(memory)

            # Backward compatibility with old format
            elif "expected_answer" in query and query["expected_answer"]:
                memory_text = query["expected_answer"]

                # Create metadata that's useful for retrieval
                metadata = {
                    "memory_id": i,
                    "category": query.get("category", "unknown"),
                    "content": memory_text,
                    "timestamp": time.time() - (i * 600),  # Space out timestamps
                    "importance": np.random.uniform(0.5, 1.0),
                }

                # Add memory
                memory = MemoryEntry(text=memory_text, metadata=metadata)
                self.memories.append(memory)

        # Group memories by category for metrics
        categories = {}
        for memory in self.memories:
            category = memory.metadata.get("category", "unknown")
            if category not in categories:
                categories[category] = 0
            categories[category] += 1

        # Log statistics
        logger.info(f"[green]Generated {len(self.memories)} memories from queries[/green]")
        for category, count in categories.items():
            logger.info(f"[green]  - {category}: {count} memories[/green]")

    def prepare_memory_system(self, config: EvaluationConfig) -> tuple[Any, Any]:
        """Prepare memory system based on configuration."""
        logger.info(f"Preparing memory system for: [bold cyan]{config.name}[/bold cyan]")

        # Create memory with consistent embedding dimensions
        memory = ContextualMemory(
            embedding_dim=EMBEDDING_DIM,  # Use consistent embedding dimension
            max_memories=config.max_memories,
            use_art_clustering=config.use_art_clustering,
        )

        # Load memories into the system
        for mem in self.memories:
            memory.add_memory(embedding=mem.embedding, text=mem.text, metadata=mem.metadata)

        # Create retriever based on configuration type
        if config.retriever_type == "baseline":
            # Simple baseline retriever with minimal features
            retriever = Retriever(memory=memory, embedding_model=embedding_model)
            retriever.minimum_relevance = config.confidence_threshold
            retriever.retrieval_strategy = SimilarityRetrievalStrategy(memory)

            # Try to bypass the component pipeline for baseline
            try:
                # set a flag to bypass the component pipeline if the attribute exists
                if hasattr(retriever, "use_component_pipeline"):
                    retriever.use_component_pipeline = False
                    logger.debug("[green]Disabled component pipeline for baseline config[/green]")

                # If there's a memory manager, disable all components
                if hasattr(retriever, "memory_manager"):
                    if hasattr(retriever.memory_manager, "disable_all_pipeline_components"):
                        retriever.memory_manager.disable_all_pipeline_components()
                        logger.debug(
                            "[green]Disabled all pipeline components for baseline config[/green]"
                        )
                    elif hasattr(retriever.memory_manager, "pipeline_components"):
                        retriever.memory_manager.pipeline_components = {}
                        logger.debug(
                            "[green]Cleared pipeline components for baseline config[/green]"
                        )
            except Exception as e:
                logger.warning(f"[yellow]Couldn't disable pipeline components: {e}[/yellow]")

            # Force initialization
            if hasattr(retriever, "initialize_components"):
                retriever.initialize_components()

        elif config.retriever_type == "contextual":
            # Retriever with contextual features
            retriever = Retriever(memory=memory, embedding_model=embedding_model)
            retriever.minimum_relevance = config.confidence_threshold

            # Disable problematic components if possible
            if hasattr(retriever, "memory_manager"):
                try:
                    # Try to safely disable memory decay component which might be causing issues
                    pipeline_components = getattr(
                        retriever.memory_manager, "pipeline_components", {}
                    )
                    if "memory_decay" in pipeline_components:
                        logger.debug(
                            "[green]Disabling memory decay component for contextual config[/green]"
                        )
                        pipeline_components.pop("memory_decay")
                except Exception as e:
                    logger.warning(f"[yellow]Couldn't modify pipeline components: {e}[/yellow]")

            # Use a hybrid retrieval strategy
            if config.use_two_stage_retrieval:
                retriever.retrieval_strategy = TwoStageRetrievalStrategy(memory)
            else:
                retriever.retrieval_strategy = HybridRetrievalStrategy(memory)

            # Configure features
            if hasattr(retriever, "configure_semantic_coherence"):
                retriever.configure_semantic_coherence(enable=config.semantic_coherence_check)
            if hasattr(retriever, "configure_query_type_adaptation"):
                retriever.configure_query_type_adaptation(enable=config.query_type_adaptation)

            # Force initialization
            if hasattr(retriever, "initialize_components"):
                retriever.initialize_components()

        elif config.retriever_type == "component":
            # Full component architecture with all features
            retriever = Retriever(memory=memory, embedding_model=embedding_model)
            retriever.minimum_relevance = config.confidence_threshold

            # Handle memory decay issue
            if hasattr(retriever, "memory_manager"):
                try:
                    # Try to safely disable memory decay component or replace its process_query method
                    pipeline_components = getattr(
                        retriever.memory_manager, "pipeline_components", {}
                    )
                    if "memory_decay" in pipeline_components:
                        if not hasattr(pipeline_components["memory_decay"], "process_query"):
                            logger.debug(
                                "[green]Working around memory decay component issue[/green]"
                            )
                            # Either remove it or fix it
                            if not config.memory_decay_enabled:
                                pipeline_components.pop("memory_decay")
                            else:
                                # Try to add the missing method if we know it needs it
                                from types import MethodType

                                def temp_process_query(self, query, context):
                                    return self.process({"query": query}, context)

                                pipeline_components["memory_decay"].process_query = MethodType(
                                    temp_process_query, pipeline_components["memory_decay"]
                                )
                except Exception as e:
                    logger.warning(f"[yellow]Couldn't fix memory decay component: {e}[/yellow]")

            # Configure all advanced features
            if config.use_two_stage_retrieval:
                retriever.configure_two_stage_retrieval(
                    enable=True,
                    first_stage_k=config.first_stage_k,
                    first_stage_threshold_factor=config.first_stage_threshold_factor,
                )

            if config.query_type_adaptation:
                retriever.configure_query_type_adaptation(
                    enable=True,
                    adaptation_strength=1.0,
                )

            if config.semantic_coherence_check:
                retriever.configure_semantic_coherence(enable=True)

            if config.dynamic_threshold_adjustment:
                retriever.enable_dynamic_threshold_adjustment(
                    enable=True,
                    window_size=5,
                )

            # Be extra cautious with memory decay
            if config.memory_decay_enabled and hasattr(retriever, "configure_memory_decay"):
                try:
                    retriever.configure_memory_decay(
                        enable=True,
                        decay_rate=0.05,
                    )
                except Exception as e:
                    logger.warning(f"[yellow]Couldn't configure memory decay: {e}[/yellow]")

            # Force initialization of components
            retriever.initialize_components()
        else:
            logger.warning(
                f"[yellow]Unknown retriever type '{config.retriever_type}', using default[/yellow]"
            )
            retriever = Retriever(memory=memory, embedding_model=embedding_model)
            retriever.initialize_components()

        return memory, retriever

    def run_evaluation(
        self,
        save_path: Optional[str] = None,
        quiet: bool = False,
        debug: bool = False,
        visualize: bool = True,
    ) -> list[EvaluationResults]:
        """Run evaluation for all configurations."""
        if not self.test_queries:
            logger.error("[bold red]No test queries loaded. Load a dataset first.[/bold red]")
            return []

        # set log level based on options
        if debug:
            logger.setLevel(logging.DEBUG)
            # Enable more verbose logging for components
            logging.getLogger("memoryweave.components").setLevel(logging.DEBUG)
            logging.getLogger("memoryweave.pipeline").setLevel(logging.DEBUG)
        elif quiet:
            # Quiet mode - reduce log output
            logger.setLevel(logging.WARNING)
            logging.getLogger("memoryweave").setLevel(logging.WARNING)
            logging.getLogger("rich").setLevel(logging.WARNING)

        self.results = []

        # Create progress display
        progress_columns = [
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[cyan]{task.completed}/{task.total}"),
            TimeElapsedColumn(),
        ]

        with Progress(*progress_columns) as progress:
            # Run each configuration
            for config in self.configs:
                config_task = progress.add_task(
                    f"Evaluating {config.name}",
                    total=len(self.test_queries) + 1,  # +1 for setup
                )

                try:
                    # Time setup
                    setup_start = time.time()
                    memory, retriever = self.prepare_memory_system(config)
                    setup_time = time.time() - setup_start

                    # Update progress
                    progress.update(config_task, advance=1)

                    # Prepare for results
                    query_times = []
                    retrieval_counts = []
                    expected_indices = []
                    retrieved_indices = []

                    # Run each query
                    for query_data in self.test_queries:
                        query = query_data["query"]
                        expected_answer = query_data.get("expected_answer", "")

                        # Find memory IDs that contain the expected answer
                        if "expected_answers" in query_data and query_data["expected_answers"]:
                            # New format - multiple expected answers
                            expected_ids = set()
                            for expected_answer in query_data["expected_answers"]:
                                answer_ids = self._find_memories_with_text(expected_answer, query)
                                expected_ids.update(answer_ids)
                        else:
                            # Old format - single expected answer
                            expected_ids = self._find_memories_with_text(expected_answer, query)

                        # Debug finding the right expected IDs
                        if (
                            debug
                            and not expected_ids
                            and not query_data.get("category") == "out-of-domain"
                        ):
                            logger.debug(
                                f"[yellow]No expected IDs found for query: {query}[/yellow]"
                            )

                        expected_indices.append(expected_ids)

                        try:
                            # Different approach based on retriever type
                            if config.retriever_type == "baseline":
                                # For baseline, try direct approach to bypass component pipeline
                                start_time = time.time()
                                query_embedding = embedding_model.encode(
                                    query,
                                    show_progress_bar=False,
                                )

                                # Try direct memory retrieval if available
                                if hasattr(memory, "retrieve_memories"):
                                    # Use direct retrieval with configuration-specific result count
                                    memory_results = memory.retrieve_memories(
                                        query_embedding,
                                        top_k=config.result_count,
                                        confidence_threshold=config.confidence_threshold,
                                    )

                                    # Convert to standard format
                                    results = []
                                    for idx, score, metadata in memory_results:
                                        results.append({
                                            "memory_id": idx,
                                            "relevance_score": score,
                                            "content": metadata.get("content", ""),
                                        })
                                else:
                                    # Fall back to regular retrieval
                                    results = retriever.retrieve(query, top_k=config.result_count)

                                query_time = time.time() - start_time
                            else:
                                # Normal approach for other retriever types
                                start_time = time.time()
                                results = retriever.retrieve(query, top_k=config.result_count)
                                query_time = time.time() - start_time

                            query_times.append(query_time)

                            # Extract retrieved indices
                            result_ids = set()
                            for r in results:
                                if "memory_id" in r:
                                    result_ids.add(r["memory_id"])
                            retrieved_indices.append(result_ids)
                            retrieval_counts.append(len(result_ids))

                            # Debug info
                            if debug:
                                logger.debug(f"Query: {query}")
                                logger.debug(f"Expected IDs: {expected_ids}")
                                logger.debug(f"Retrieved IDs: {result_ids}")
                                logger.debug(f"Retrieved count: {len(result_ids)}")
                                logger.debug(f"Query time: {query_time:.4f}s")
                        except Exception as e:
                            logger.warning(
                                f"[yellow]Error processing query '{query}': {e}[/yellow]"
                            )
                            # Add empty result for this query to maintain consistent counts
                            retrieved_indices.append(set())
                            retrieval_counts.append(0)
                            query_times.append(0)

                        # Update progress
                        progress.update(config_task, advance=1)

                except Exception as e:
                    logger.error(
                        f"[bold red]Error evaluating config '{config.name}': {e}[/bold red]"
                    )
                    if debug:
                        import traceback

                        logger.error(traceback.format_exc())
                    # Skip to next config
                    progress.update(config_task, completed=True)
                    continue

                # Calculate metrics if we have results
                if expected_indices and retrieved_indices:
                    precisions, recalls, f1_scores = [], [], []
                    semantic_scores = []
                    error_count = 0
                    query_type_metrics = {}

                    # Process each query's results
                    for idx, (query_data, expected, retrieved) in enumerate(
                        zip(self.test_queries, expected_indices, retrieved_indices)
                    ):
                        query = query_data["query"]
                        category = query_data.get("category", "unknown")

                        # Initialize category metrics if needed
                        if category not in query_type_metrics:
                            query_type_metrics[category] = {
                                "precisions": [],
                                "recalls": [],
                                "f1_scores": [],
                                "counts": 0,
                            }

                        # Count this query
                        query_type_metrics[category]["counts"] += 1

                        # Skip evaluation for out-of-domain queries
                        if category == "out-of-domain":
                            # For out-of-domain, precision is perfect if nothing retrieved
                            if not retrieved:
                                precision = 1.0
                            else:
                                precision = 0.0

                            # For out-of-domain, recall is always 1.0 since nothing expected
                            recall = 1.0
                        else:
                            # Calculate precision/recall for in-domain queries
                            if retrieved:
                                precision = len(expected.intersection(retrieved)) / len(retrieved)
                            else:
                                precision = 0.0

                            if expected:
                                recall = len(expected.intersection(retrieved)) / len(expected)
                            else:
                                # If nothing expected but it's not explicitly out-of-domain, it's an error
                                recall = 0.0
                                error_count += 1

                        # Calculate F1 score
                        if precision + recall > 0:
                            f1 = 2 * (precision * recall) / (precision + recall)
                        else:
                            f1 = 0.0

                        # Add to overall metrics
                        precisions.append(precision)
                        recalls.append(recall)
                        f1_scores.append(f1)

                        # Add to category metrics
                        query_type_metrics[category]["precisions"].append(precision)
                        query_type_metrics[category]["recalls"].append(recall)
                        query_type_metrics[category]["f1_scores"].append(f1)

                        # Calculate semantic similarity if we have real embeddings
                        if USE_REAL_EMBEDDINGS and retrieved and query:
                            # Get query embedding
                            query_embedding = embedding_model.encode(query, show_progress_bar=False)

                            # Get embeddings of retrieved documents and their scores
                            retrieved_embeddings = []
                            retrieved_scores = []

                            for (
                                r
                            ) in results:  # Corrected: iterate through results, not retrieved IDs
                                memory_id = r.get("memory_id")
                                if memory_id is not None:  # Check if memory_id is valid
                                    for memory in self.memories:
                                        if (
                                            memory.metadata.get("memory_id") == memory_id
                                            and memory.embedding is not None
                                        ):
                                            retrieved_embeddings.append(memory.embedding)
                                            retrieved_scores.append(
                                                r.get("relevance_score", 0.0)
                                            )  # Fallback to 0 if score missing

                            if retrieved_embeddings:
                                # Calculate average semantic similarity (using relevance scores for now as a proxy)
                                avg_semantic_score = np.mean(retrieved_scores)
                                semantic_scores.append(avg_semantic_score)
                            else:
                                semantic_scores.append(0.0)  # No retrieved docs, no similarity.
                        else:
                            semantic_scores.append(
                                0.0
                            )  # No real embeddings or nothing retrieved, no semantic score

                    # Finalize metrics for config
                    avg_precision = np.mean(precisions) if precisions else 0.0
                    avg_recall = np.mean(recalls) if recalls else 0.0
                    avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
                    avg_semantic_similarity = np.mean(semantic_scores) if semantic_scores else 0.0
                    error_rate = error_count / len(self.test_queries) if self.test_queries else 0.0

                    # Calculate category-specific metrics
                    for category, metrics in query_type_metrics.items():
                        category_precisions = metrics["precisions"]
                        category_recalls = metrics["recalls"]
                        category_f1s = metrics["f1_scores"]

                        query_type_metrics[category] = {
                            "precision": np.mean(category_precisions)
                            if category_precisions
                            else 0.0,
                            "recall": np.mean(category_recalls) if category_recalls else 0.0,
                            "f1_score": np.mean(category_f1s) if category_f1s else 0.0,
                            "count": metrics["counts"],
                        }

                    # Record results
                    results = EvaluationResults(
                        config=config,
                        setup_time=setup_time,
                        query_times=query_times,
                        precision=avg_precision,
                        recall=avg_recall,
                        f1_score=avg_f1,
                        semantic_similarity=avg_semantic_similarity,
                        retrieval_counts=retrieval_counts,
                        by_query_type=query_type_metrics,
                        error_rate=error_rate,
                        memory_usage=memory.get_memory_usage()
                        if hasattr(memory, "get_memory_usage")
                        else 0.0,  # Optional memory usage
                    )
                    self.results.append(results)

                    # Output results to console
                    self._output_results_console(results)

                    # Save results if path provided
                    if save_path:
                        self.save_results(save_path)

                progress.update(config_task, completed=True)

        if visualize:
            self.visualize_results()

        return self.results

    def _find_memories_with_text(self, text: str, query: Optional[str] = None) -> set[int]:
        """Find memory IDs that contain specific text."""
        found_ids = set()
        for memory in self.memories:
            if text.lower() in memory.text.lower():  # Case-insensitive matching
                found_ids.add(memory.metadata.get("memory_id"))
        return found_ids

    def _output_results_console(self, results: EvaluationResults) -> None:
        """Output evaluation results to the console in a rich table."""
        console.rule(f"[bold magenta]Evaluation Results: {results.config.name}[/bold magenta]")

        # Summary Panel
        config_table = Table(show_header=False, box=None)
        config_table.add_row("[bold]Configuration Name:[/bold]", results.config.name)
        config_table.add_row("[bold]Description:[/bold]", results.config.description)
        config_table.add_row("[bold]Retriever Type:[/bold]", results.config.retriever_type)

        # Feature Flags Table
        features_table = Table(title="Feature Flags", show_header=True, header_style="bold cyan")
        features_table.add_column("Feature", style="dim", no_wrap=True)
        features_table.add_column("Enabled", style="magenta")
        config_dict = results.config.to_dict()  # Get config as dict to iterate over flags
        for key, value in config_dict.items():
            if (
                key.startswith("use_")
                or key.endswith("_enabled")
                or key
                in [
                    "semantic_coherence_check",
                    "adaptive_retrieval",
                    "query_type_adaptation",
                    "dynamic_threshold_adjustment",
                    "use_two_stage_retrieval",
                ]
            ):
                features_table.add_row(key, "[green]Yes[/green]" if value else "[red]No[/red]")

        # Metrics Table
        metrics_table = Table(
            title="Performance Metrics", show_header=True, header_style="bold cyan"
        )
        metrics_table.add_column("Metric", style="dim", no_wrap=True)
        metrics_table.add_column("Value", justify="right", style="green")
        metrics_table.add_row("setup Time", f"{results.setup_time:.4f} sec")
        metrics_table.add_row("Avg. Query Time", f"{results.avg_query_time:.4f} sec")
        metrics_table.add_row(
            "Memory Usage", f"{results.memory_usage:.2f} MB" if results.memory_usage > 0 else "N/A"
        )
        metrics_table.add_row("Precision", f"{results.precision:.4f}")
        metrics_table.add_row("Recall", f"{results.recall:.4f}")
        metrics_table.add_row("F1 Score", f"{results.f1_score:.4f}")
        metrics_table.add_row("Semantic Similarity", f"{results.semantic_similarity:.4f}")
        metrics_table.add_row("Overall Score", f"{results.overall_score:.4f}")
        metrics_table.add_row("Avg. Retrieval Count", f"{results.avg_retrieval_count:.2f}")
        metrics_table.add_row("Error Rate", f"{results.error_rate:.4f}")

        # Query Type Metrics Table (if available)
        if results.by_query_type:
            query_type_table = Table(
                title="Metrics by Query Type", show_header=True, header_style="bold cyan"
            )
            query_type_table.add_column("Query Type", style="dim", no_wrap=True)
            query_type_table.add_column("Precision", justify="right", style="green")
            query_type_table.add_column("Recall", justify="right", style="green")
            query_type_table.add_column("F1 Score", justify="right", style="green")
            query_type_table.add_column("Count", justify="right", style="dim")

            for q_type, q_metrics in results.by_query_type.items():
                query_type_table.add_row(
                    q_type,
                    f"{q_metrics['precision']:.4f}",
                    f"{q_metrics['recall']:.4f}",
                    f"{q_metrics['f1_score']:.4f}",
                    str(q_metrics["count"]),
                )
        else:
            query_type_table = None

        # Combine tables into a panel
        panel_elements = [config_table, features_table, metrics_table]
        if query_type_table:
            panel_elements.append(query_type_table)

        results_panel = Panel(
            Group(*panel_elements),
            title="[bold magenta]Evaluation Summary[/bold magenta]",
            border_style="magenta",
            padding=(1, 2),
        )
        console.print(results_panel)

    def save_results(self, save_path: str) -> None:
        """Save evaluation results to a JSON file."""
        all_results_json = [r.to_dict() for r in self.results]
        try:
            with open(save_path, "w") as f:
                json.dump(all_results_json, f, indent=4)
            logger.info(f"[green]Results saved to [bold]{save_path}[/bold][/green]")
        except Exception as e:
            logger.error(f"[bold red]Error saving results to {save_path}: {e}[/bold red]")

    def visualize_results(self) -> None:
        """Visualize evaluation results using matplotlib."""
        if not self.results:
            logger.warning("[yellow]No results available to visualize.[/yellow]")
            return

        config_names = [res.config.name for res in self.results]
        f1_scores = [res.f1_score for res in self.results]
        semantic_similarity_scores = [res.semantic_similarity for res in self.results]
        overall_scores = [res.overall_score for res in self.results]
        query_times = [res.avg_query_time for res in self.results]

        # set up matplotlib figure and axes
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # Adjusted for 4 subplots
        fig.suptitle("MemoryWeave Evaluation Metrics Comparison", fontsize=16)

        # F1 Score Bar Chart
        axes[0, 0].bar(config_names, f1_scores, color="skyblue")
        axes[0, 0].set_ylabel("F1 Score")
        axes[0, 0].set_title("F1 Score Comparison")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Semantic Similarity Bar Chart
        axes[0, 1].bar(config_names, semantic_similarity_scores, color="lightcoral")
        axes[0, 1].set_ylabel("Semantic Similarity")
        axes[0, 1].set_title("Semantic Similarity Comparison")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Overall Score Bar Chart
        axes[1, 0].bar(config_names, overall_scores, color="lightgreen")
        axes[1, 0].set_ylabel("Overall Score")
        axes[1, 0].set_title("Overall Score Comparison")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Query Time Bar Chart
        axes[1, 1].bar(config_names, query_times, color="gold")
        axes[1, 1].set_ylabel("Avg. Query Time (seconds)")
        axes[1, 1].set_title("Average Query Time Comparison")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle
        plt.show()


@click.command()
@click.option(
    "--dataset",
    "-d",
    type=click.Path(exists=True),
    default="datasets/enhanced_evaluation.json",
    help="Path to the evaluation dataset JSON file.",
)
@click.option(
    "--save-path",
    "-s",
    type=click.Path(),
    default="results.json",
    help="Path to save the evaluation results as JSON.",
)
@click.option(
    "--quiet", "-q", is_flag=True, default=False, help="Enable quiet mode (less logging)."
)
@click.option(
    "--debug", "-D", is_flag=True, default=False, help="Enable debug mode (more verbose logging)."
)
@click.option("--no-visualize", is_flag=True, default=False, help="Disable result visualization.")
def main(dataset: str, save_path: str, quiet: bool, debug: bool, no_visualize: bool):
    """
    Run real-world evaluation of MemoryWeave with different configurations.
    """
    visualize = not no_visualize  # Invert flag logic

    # Define configurations to evaluate
    configs = [
        EvaluationConfig(
            name="Baseline",
            description="Simple baseline retriever with minimal features.",
            retriever_type="baseline",
        ),
        EvaluationConfig(
            name="Contextual",
            description="Retriever with contextual features and hybrid retrieval.",
            retriever_type="contextual",
            semantic_coherence_check=True,
            query_type_adaptation=True,
        ),
        EvaluationConfig(
            name="Component",
            description="Full component architecture with advanced features.",
            retriever_type="component",
            semantic_coherence_check=True,
            query_type_adaptation=True,
            dynamic_threshold_adjustment=True,
            memory_decay_enabled=True,
            use_two_stage_retrieval=True,
        ),
        EvaluationConfig(
            name="Component (No Decay)",
            description="Component architecture without memory decay.",
            retriever_type="component",
            semantic_coherence_check=True,
            query_type_adaptation=True,
            dynamic_threshold_adjustment=True,
            memory_decay_enabled=False,  # Memory decay disabled for this config
            use_two_stage_retrieval=True,
        ),
    ]

    # Initialize and run evaluator
    evaluator = RealWorldEvaluator(configs=configs)

    try:
        evaluator.load_dataset(dataset)
        evaluator.run_evaluation(save_path, quiet, debug, visualize)

    except FileNotFoundError:
        logger.error("[bold red]Dataset file not found. Please check the path.[/bold red]")
    except Exception as e:
        logger.error(f"[bold red]Evaluation failed: {e}[/bold red]")
        if debug:
            import traceback

            logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
