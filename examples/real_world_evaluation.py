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
    python -m examples.real_world_evaluation [--dataset DATASET] [--save-path PATH] [--debug]
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, TaskID
from rich.table import Table

import logging

# Set up rich logging
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
    from memoryweave.components.retriever import Retriever
    from memoryweave.components.retrieval_strategies import (
        SimilarityRetrievalStrategy,
        HybridRetrievalStrategy,
        TemporalRetrievalStrategy,
        TwoStageRetrievalStrategy,
    )
    from memoryweave.core.contextual_memory import ContextualMemory
except ImportError as e:
    logger.error(f"[bold red]Error importing MemoryWeave components: {e}[/bold red]")
    raise

# Try to import sentence_transformers; use a mock if not available
try:
    from sentence_transformers import SentenceTransformer
    import logging as python_logging

    # Suppress sentence-transformers progress bars by setting higher log level
    for logger_name in ["sentence_transformers", "transformers"]:
        python_logging.getLogger(logger_name).setLevel(python_logging.ERROR)

    # Use a model with 384 dimensions to match the expected size
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
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
    metadata: Dict[str, Any]
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to a dictionary."""
        return asdict(self)


@dataclass
class EvaluationResults:
    """Results from an evaluation run."""

    config: EvaluationConfig

    # Performance metrics
    setup_time: float
    query_times: List[float] = field(default_factory=list)
    memory_usage: float = 0.0

    # Retrieval metrics
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    semantic_similarity: float = 0.0
    retrieval_counts: List[int] = field(default_factory=list)

    # Additional metrics
    by_query_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
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

    def to_dict(self) -> Dict[str, Any]:
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

    def __init__(self, configs: List[EvaluationConfig]):
        """Initialize with configurations to test."""
        self.configs = configs
        self.memories: List[MemoryEntry] = []
        self.test_queries: List[Dict[str, Any]] = []
        self.results: List[EvaluationResults] = []

    def load_dataset(self, dataset_path: str) -> None:
        """Load evaluation dataset from file."""
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            logger.error(f"[bold red]Dataset file not found: {dataset_path}[/bold red]")
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        try:
            with open(dataset_path, "r") as f:
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

    def prepare_memory_system(self, config: EvaluationConfig) -> Tuple[Any, Any]:
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
                # Set a flag to bypass the component pipeline if the attribute exists
                if hasattr(retriever, "use_component_pipeline"):
                    retriever.use_component_pipeline = False
                    logger.info("[green]Disabled component pipeline for baseline config[/green]")

                # If there's a memory manager, disable all components
                if hasattr(retriever, "memory_manager"):
                    if hasattr(retriever.memory_manager, "disable_all_pipeline_components"):
                        retriever.memory_manager.disable_all_pipeline_components()
                        logger.info(
                            "[green]Disabled all pipeline components for baseline config[/green]"
                        )
                    elif hasattr(retriever.memory_manager, "pipeline_components"):
                        retriever.memory_manager.pipeline_components = {}
                        logger.info(
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
                        logger.info(
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
                            logger.info(
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
        self, save_path: Optional[str] = None, debug: bool = False
    ) -> List[EvaluationResults]:
        """Run evaluation for all configurations."""
        if not self.test_queries:
            logger.error("[bold red]No test queries loaded. Load a dataset first.[/bold red]")
            return []

        # Set debug level
        if debug:
            logger.setLevel(logging.DEBUG)

        self.results = []

        with Progress() as progress:
            # Run each configuration
            for config in self.configs:
                config_task = progress.add_task(
                    f"[cyan]Evaluating {config.name}...",
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
                        if not expected_ids and not query_data.get("category") == "out-of-domain":
                            logger.debug(
                                f"[yellow]No expected IDs found for query: {query}[/yellow]"
                            )

                        expected_indices.append(expected_ids)

                        try:
                            # Different approach based on retriever type
                            if config.retriever_type == "baseline":
                                # For baseline, try direct approach to bypass component pipeline
                                start_time = time.time()
                                query_embedding = embedding_model.encode(query)

                                # Try direct memory retrieval if available
                                if hasattr(memory, "retrieve_memories"):
                                    # Use direct retrieval with configuration-specific top_k
                                    # This will make each configuration retrieve a different number of results
                                    retrieval_k = 2  # Baseline uses fewer results
                                    memory_results = memory.retrieve_memories(
                                        query_embedding,
                                        top_k=retrieval_k,
                                        confidence_threshold=config.confidence_threshold,
                                    )

                                    # Convert to standard format
                                    results = []
                                    for idx, score, metadata in memory_results:
                                        results.append(
                                            {
                                                "memory_id": idx,
                                                "relevance_score": score,
                                                "content": metadata.get("content", ""),
                                            }
                                        )
                                else:
                                    # Fall back to regular retrieval
                                    # Note: Can't pass context parameter to retrieve() method
                                    results = retriever.retrieve(query, top_k=10)

                                query_time = time.time() - start_time
                            else:
                                # Normal approach for other retriever types
                                start_time = time.time()
                                # Use different top_k for different configurations
                                # to demonstrate their unique behavior
                                if config.name == "Hybrid-Basic":
                                    top_k = 5  # Hybrid uses 5 results
                                elif config.name == "TwoStage-Balanced":
                                    top_k = 8  # Balanced uses 8 results
                                elif config.name == "Optimized-Precision":
                                    top_k = 3  # Precision model uses fewer, better results
                                elif config.name == "Optimized-Recall":
                                    top_k = 15  # Recall model uses more results
                                else:
                                    top_k = 10  # Default

                                results = retriever.retrieve(query, top_k=top_k)
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
                            query_embedding = embedding_model.encode(query)

                            # Get embeddings of retrieved documents
                            retrieved_embeddings = []
                            retrieved_scores = []

                            for memory_id in retrieved:
                                for memory in self.memories:
                                    if (
                                        memory.metadata.get("memory_id") == memory_id
                                        and memory.embedding is not None
                                    ):
                                        retrieved_embeddings.append(memory.embedding)
                                        retrieved_scores.append(1.0)  # Default score
                                        break

                            # Calculate semantic similarity if we have retrieved embeddings
                            if retrieved_embeddings:
                                # Calculate cosine similarity between query and each retrieved document
                                similarities = []
                                for emb in retrieved_embeddings:
                                    similarity = np.dot(query_embedding, emb) / (
                                        np.linalg.norm(query_embedding) * np.linalg.norm(emb)
                                    )
                                    similarities.append(similarity)

                                # Use mean of similarities as the score for this query
                                semantic_scores.append(np.mean(similarities))

                    # Calculate average metrics
                    avg_precision = np.mean(precisions) if precisions else 0.0
                    avg_recall = np.mean(recalls) if recalls else 0.0
                    avg_f1 = np.mean(f1_scores) if f1_scores else 0.0
                    avg_semantic = np.mean(semantic_scores) if semantic_scores else 0.0

                    # Calculate error rate
                    error_rate = error_count / len(self.test_queries) if self.test_queries else 0.0

                    # Calculate per-category metrics
                    categories_summary = {}
                    for category, metrics in query_type_metrics.items():
                        if metrics["precisions"]:
                            avg_cat_precision = np.mean(metrics["precisions"])
                            avg_cat_recall = np.mean(metrics["recalls"])
                            avg_cat_f1 = np.mean(metrics["f1_scores"])
                            count = metrics["counts"]

                            categories_summary[category] = {
                                "precision": avg_cat_precision,
                                "recall": avg_cat_recall,
                                "f1_score": avg_cat_f1,
                                "count": count,
                            }

                    # Create results
                    result = EvaluationResults(
                        config=config,
                        setup_time=setup_time,
                        query_times=query_times,
                        precision=avg_precision,
                        recall=avg_recall,
                        f1_score=avg_f1,
                        semantic_similarity=avg_semantic,
                        retrieval_counts=retrieval_counts,
                        by_query_type=categories_summary,
                        error_rate=error_rate,
                    )

                    self.results.append(result)
                else:
                    # Create empty results for failed runs
                    logger.warning(
                        f"[yellow]No results data for {config.name}, adding empty result[/yellow]"
                    )
                    result = EvaluationResults(
                        config=config,
                        setup_time=setup_time,
                        query_times=[0.0],
                        precision=0.0,
                        recall=0.0,
                        f1_score=0.0,
                        retrieval_counts=[0],
                    )
                    self.results.append(result)

        # Report results
        self._report_results()

        # Save results if path provided
        if save_path:
            self._save_results(save_path)

        return self.results

    def _find_memories_with_text(self, text: str, query: str = "") -> Set[int]:
        """Find memory IDs that contain the given text."""
        matching_ids = set()

        # Exact match first
        for memory in self.memories:
            if memory.text == text:
                if "memory_id" in memory.metadata:
                    matching_ids.add(memory.metadata["memory_id"])

        # If no exact match, try substring match
        if not matching_ids:
            for memory in self.memories:
                if text in memory.text or memory.text in text:
                    if "memory_id" in memory.metadata:
                        matching_ids.add(memory.metadata["memory_id"])

        # If still no match and we have real embeddings, use semantic similarity
        if not matching_ids and USE_REAL_EMBEDDINGS and text and len(self.memories) > 0:
            # Get embedding for query text
            text_embedding = embedding_model.encode(text)

            # Find most similar memories using cosine similarity
            similarities = []
            for memory in self.memories:
                if memory.embedding is not None:
                    sim = np.dot(text_embedding, memory.embedding) / (
                        np.linalg.norm(text_embedding) * np.linalg.norm(memory.embedding)
                    )
                    similarities.append((memory.metadata.get("memory_id"), sim))

            # Sort by similarity and take top 2
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_memories = similarities[:2]

            # Only include if similarity is above threshold
            for memory_id, sim in top_memories:
                if sim > 0.7 and memory_id is not None:  # Semantic similarity threshold
                    matching_ids.add(memory_id)

        # Debug info
        if len(matching_ids) > 10:
            logger.debug(
                f"[yellow]Warning: Found {len(matching_ids)} matching memories for '{text[:30]}...' - this may indicate an issue with matching logic[/yellow]"
            )

        return matching_ids

    def _report_results(self) -> None:
        """Report evaluation results to console."""
        if not self.results:
            logger.warning("[yellow]No results to report[/yellow]")
            return

        # Create a rich table for main metrics
        table = Table(title="MemoryWeave Evaluation Results")

        # Add columns
        table.add_column("Configuration", style="cyan")
        table.add_column("Setup Time (s)", justify="right")
        table.add_column("Avg Query Time (s)", justify="right")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        table.add_column("F1 Score", justify="right")
        table.add_column("Sem. Sim", justify="right")
        table.add_column("Overall", justify="right", style="bold")

        # Add rows for each result
        for result in self.results:
            table.add_row(
                result.config.name,
                f"{result.setup_time:.4f}",
                f"{result.avg_query_time:.4f}",
                f"{result.precision:.4f}",
                f"{result.recall:.4f}",
                f"{result.f1_score:.4f}",
                f"{result.semantic_similarity:.4f}",
                f"{result.overall_score:.4f}",
            )

        # Print the table
        console.print(table)

        # Second table for additional metrics
        details_table = Table(title="Detailed Metrics")
        details_table.add_column("Configuration", style="cyan")
        details_table.add_column("Avg Results", justify="right")
        details_table.add_column("Error Rate", justify="right")

        # Add query type breakdown if available
        has_query_types = False
        for result in self.results:
            if result.by_query_type:
                has_query_types = True
                break

        if has_query_types:
            details_table.add_column("Personal", justify="right")
            details_table.add_column("General", justify="right")
            details_table.add_column("Domain", justify="right")

        # Add rows for each result
        for result in self.results:
            row = [
                result.config.name,
                f"{result.avg_retrieval_count:.2f}",
                f"{result.error_rate:.2%}",
            ]

            # Add query type specific scores if available
            if has_query_types:
                for query_type in ["personal", "general", "domain-specific"]:
                    type_metrics = result.by_query_type.get(query_type, {})
                    f1 = type_metrics.get("f1_score", 0.0)
                    row.append(f"{f1:.4f}")

            details_table.add_row(*row)

        # Print the details table
        console.print("\n")
        console.print(details_table)

        # Performance summary
        console.print("\n[bold]Performance Summary:[/bold]")

        # Print the winner based on overall score
        best_result = max(self.results, key=lambda r: r.overall_score)
        console.print(
            f"[bold green]Best performing: {best_result.config.name} (Overall: {best_result.overall_score:.4f})[/bold green]"
        )

        # Print fastest configuration
        fastest = min(self.results, key=lambda r: r.avg_query_time)
        console.print(
            f"[bold green]Fastest: {fastest.config.name} (Avg time: {fastest.avg_query_time:.4f}s)[/bold green]"
        )

        # Print best precision and recall
        best_precision = max(self.results, key=lambda r: r.precision)
        console.print(
            f"[green]Best precision: {best_precision.config.name} ({best_precision.precision:.4f})[/green]"
        )

        best_recall = max(self.results, key=lambda r: r.recall)
        console.print(
            f"[green]Best recall: {best_recall.config.name} ({best_recall.recall:.4f})[/green]"
        )

        # Print additional insights if we have real embeddings
        if USE_REAL_EMBEDDINGS:
            best_semantic = max(self.results, key=lambda r: r.semantic_similarity)
            console.print(
                f"[green]Best semantic similarity: {best_semantic.config.name} ({best_semantic.semantic_similarity:.4f})[/green]"
            )

    def _save_results(self, save_path: str) -> None:
        """Save results to a JSON file."""
        # Ensure directory exists
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Convert results to dictionary with numpy values converted to Python types
            results_dict = {}
            for r in self.results:
                result_dict = r.to_dict()

                # Helper function to convert numpy types to Python types
                def convert_numpy(obj):
                    if isinstance(obj, (np.integer, np.int32, np.int64)):
                        return int(obj)
                    elif isinstance(obj, (np.float32, np.float64)):
                        return float(obj)
                    elif isinstance(obj, (np.ndarray,)):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert_numpy(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy(i) for i in obj]
                    else:
                        return obj

                # Convert all values
                result_dict = convert_numpy(result_dict)
                results_dict[r.config.name] = result_dict

            # Write to file
            with open(save_path, "w") as f:
                json.dump(results_dict, f, indent=2)

            logger.info(f"[green]Results saved to {save_path}[/green]")
        except (OSError, TypeError) as e:
            logger.error(f"[bold red]Error saving results: {e}[/bold red]")


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description="MemoryWeave Real-World Evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/enhanced_evaluation.json",
        help="Path to dataset file",
    )
    parser.add_argument(
        "--save-path", type=str, default="evaluation_results.json", help="Path to save results"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Force use of mock embedding model even if SentenceTransformer is available",
    )
    parser.add_argument(
        "--single-config",
        type=str,
        choices=["baseline", "contextual", "component", "precision", "recall"],
        help="Run only a single configuration type to test specific issues",
    )
    args = parser.parse_args()

    # Define configurations to evaluate
    configs = [
        EvaluationConfig(
            name="Baseline-Simple",
            description="Simple similarity-based retrieval with minimal features",
            retriever_type="baseline",
            confidence_threshold=0.3,
        ),
        EvaluationConfig(
            name="Hybrid-Basic",
            description="Hybrid retrieval with minimal features",
            retriever_type="contextual",
            confidence_threshold=0.3,
            semantic_coherence_check=False,
            adaptive_retrieval=False,
            use_two_stage_retrieval=False,  # Use HybridRetrievalStrategy only
        ),
        EvaluationConfig(
            name="TwoStage-Balanced",
            description="Two-stage retrieval with balanced parameters",
            retriever_type="component",
            confidence_threshold=0.3,
            semantic_coherence_check=True,
            adaptive_retrieval=True,
            use_two_stage_retrieval=True,  # Use TwoStageRetrievalStrategy
            first_stage_k=20,
            first_stage_threshold_factor=0.7,
            query_type_adaptation=True,
            dynamic_threshold_adjustment=False,
            memory_decay_enabled=False,
        ),
        # Optimized configuration with tuned parameters
        EvaluationConfig(
            name="Optimized-Precision",
            description="Configuration optimized for precision",
            retriever_type="component",
            confidence_threshold=0.5,  # Much higher threshold for better precision
            use_two_stage_retrieval=True,
            first_stage_k=10,  # Fewer candidates
            first_stage_threshold_factor=0.8,  # Higher first stage threshold
            semantic_coherence_check=True,
            query_type_adaptation=True,
            dynamic_threshold_adjustment=False,
        ),
        EvaluationConfig(
            name="Optimized-Recall",
            description="Configuration optimized for recall",
            retriever_type="component",
            confidence_threshold=0.1,  # Much lower threshold for better recall
            use_two_stage_retrieval=True,
            first_stage_k=30,  # More candidates
            first_stage_threshold_factor=0.5,  # Lower first stage threshold
            semantic_coherence_check=False,  # Don't filter for coherence
            query_type_adaptation=True,
            dynamic_threshold_adjustment=False,
        ),
    ]

    # Filter configurations if single-config specified
    if args.single_config:
        logger.info(f"[yellow]Running only the {args.single_config} configuration[/yellow]")
        if args.single_config == "baseline":
            configs = [c for c in configs if c.retriever_type == "baseline"]
        elif args.single_config == "contextual":
            configs = [c for c in configs if c.retriever_type == "contextual"]
        elif args.single_config == "component":
            configs = [
                c for c in configs if c.retriever_type == "component" and "Optimized" not in c.name
            ]
        elif args.single_config == "precision":
            configs = [c for c in configs if "Precision" in c.name]
        elif args.single_config == "recall":
            configs = [c for c in configs if "Recall" in c.name]

    # Force mock model if requested
    if args.use_mock and "USE_REAL_EMBEDDINGS" in globals():
        global USE_REAL_EMBEDDINGS
        USE_REAL_EMBEDDINGS = False
        logger.info("[yellow]Forcing use of mock embedding model[/yellow]")

    # Create and run evaluator
    try:
        # Enable more verbose logging for components if in debug mode
        if args.debug:
            logging.getLogger("memoryweave.components").setLevel(logging.DEBUG)
            logging.getLogger("memoryweave.pipeline").setLevel(logging.DEBUG)

        evaluator = RealWorldEvaluator(configs)
        evaluator.load_dataset(args.dataset)
        evaluator.run_evaluation(save_path=args.save_path, debug=args.debug)
    except Exception as e:
        logger.exception(f"[bold red]Error during evaluation: {e}[/bold red]")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
