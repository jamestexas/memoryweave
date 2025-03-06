import concurrent.futures
import logging
import os
import time
from importlib.util import find_spec

import numpy as np
from rich.logging import RichHandler

from memoryweave.benchmarks.utils.perf_timer import timer

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(markup=True, rich_tracebacks=True, show_path=True)],
)
logger = logging.getLogger(__name__)


def determine_optimal_workers() -> int:
    """Determine optimal number of worker threads based on system resources."""
    cpu_count = os.cpu_count() or 2  # Default to 2 if CPU count is not available
    if find_spec("psutil") is None:
        max_workers = min(cpu_count + 1, 8)  # CPU count + 1, up to 8
        logger.warning(f"[bold yellow]psutil not found, using default of {max_workers} workers[/]")
        return max_workers
    try:
        import psutil

        available_memory_gb = psutil.virtual_memory().available / (1024 * 1024 * 1024)
        if available_memory_gb > 8:
            max_workers = min(cpu_count * 2, 16)
        elif available_memory_gb > 4:
            max_workers = min(cpu_count, 8)
        else:
            max_workers = max(2, cpu_count // 2)
        logger.debug(f"Determined optimal workers: {max_workers}")
        return max_workers

    except Exception as e:
        logger.warning(f"Error determining optimal worker count: {e}. Using default of 4.")
        return 4


@timer
class RetrievalOrchestrator:
    """Orchestrates the memory retrieval process with parallel processing."""

    @timer()
    def __init__(
        self,
        strategy,
        activation_manager=None,
        temporal_context=None,
        semantic_coherence_processor=None,
        memory_store_adapter=None,
        debug=False,
        max_workers=4,  # Number of workers for parallel processing
        enable_cache=True,  # Whether to enable query caching
        max_cache_size=50,  # Maximum number of cached queries
    ):
        """
        Initialize the retrieval orchestrator.
        """
        logger.debug("Initializing RetrievalOrchestrator...")
        self.strategy = strategy
        self.activation_manager = activation_manager
        self.temporal_context = temporal_context
        self.semantic_coherence_processor = semantic_coherence_processor
        self.memory_store_adapter = memory_store_adapter
        self.debug = debug
        if max_workers is None:
            max_workers = determine_optimal_workers()
            logger.info(f"Using {max_workers} worker threads for parallel retrieval")

        self.max_workers = max_workers
        self.enable_cache = enable_cache
        self.max_cache_size = max_cache_size

        # Create thread pool executor
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        logger.debug(f"ThreadPoolExecutor created with {max_workers} workers.")

        # Initialize query cache
        self.query_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0, "last_query_time": 0}

    @timer("memory_search")
    def retrieve(
        self,
        query_embedding,
        query,
        query_type=None,
        expanded_keywords=None,
        entities=None,
        adapted_params=None,
        top_k=10,
    ):
        """
        Execute the retrieval process with parallel processing for improved performance.
        """
        start_time = time.time()
        logger.debug(f"[cyan]Processing retrieval for query:[/] '{query}'")

        # Generate cache key if caching is enabled
        cache_key = None
        if self.enable_cache:
            cache_key = self._generate_cache_key(
                query_embedding, query, query_type, expanded_keywords, entities, top_k
            )
            if cache_key in self.query_cache:
                self.cache_stats["hits"] += 1
                logger.debug(f"[green]Cache hit for query:[/] '{query}'")
                return self.query_cache[cache_key]
            self.cache_stats["misses"] += 1
            logger.debug(f"[red]Cache miss for query:[/] '{query}'")

        # Prepare retrieval context
        current_time = time.time()
        retrieval_context = {
            "query": query,
            "query_type": query_type,
            "important_keywords": expanded_keywords or [],
            "extracted_entities": entities or [],
            "current_time": current_time,
            "memory_store": self.memory_store_adapter,
        }
        if adapted_params:
            retrieval_context["adapted_retrieval_params"] = adapted_params

        logger.debug("Submitting main retrieval task to executor.")
        try:
            future_memories = self.executor.submit(
                self.strategy.retrieve,
                query_embedding=query_embedding,
                top_k=top_k,
                context=retrieval_context,
                query=query,
            )
            logger.debug("Main retrieval task submitted.")

            future_activations = None
            if self.activation_manager:
                logger.debug("Submitting activation prefetch task to executor.")
                future_activations = self.executor.submit(self._prepare_activations, top_k)

            try:
                memories = future_memories.result(timeout=9.0)
                logger.debug("Main retrieval task completed.")
                activations_ready = future_activations.done() if future_activations else False
            except concurrent.futures.TimeoutError:
                logger.error("Retrieval strategy timeout, using fallback")
                memories = self._fallback_retrieval(query_embedding, top_k)
                activations_ready = False

            if activations_ready:
                activation_data = future_activations.result()
                retrieval_context["prefetched_activations"] = activation_data
                logger.debug("Activation data retrieved and added to context.")

            processed_memories = self.post_process_results(
                memories, query_embedding, query, current_time, retrieval_context
            )

            if self.enable_cache and cache_key is not None:
                if len(self.query_cache) >= self.max_cache_size:
                    oldest_key = next(iter(self.query_cache))
                    del self.query_cache[oldest_key]
                    logger.debug("Cache full, removed oldest entry.")
                self.query_cache[cache_key] = processed_memories
                logger.debug("Result cached.")

            query_time = time.time() - start_time
            self.cache_stats["last_query_time"] = query_time
            if self.debug:
                logger.debug(f"Retrieval completed in {query_time:.3f}s")
                logger.debug(f"Retrieved {len(processed_memories)} memories")

            return processed_memories

        except Exception as e:
            logger.error(f"Error using primary retrieval strategy: {e}")
            import traceback

            traceback.print_exc()
            memories = self._fallback_retrieval(query_embedding, top_k)
            processed_memories = self.post_process_results(
                memories, query_embedding, query, current_time
            )
            query_time = time.time() - start_time
            self.cache_stats["last_query_time"] = query_time
            logger.info(
                f"[bold cyan]Retrieved {len(memories)} memories[/] in [green]{query_time:.3f}s[/]"
            )
            return processed_memories

    def _generate_cache_key(self, query_embedding, query, query_type, keywords, entities, top_k):
        """
        Generate a cache key for a query.
        """
        try:
            emb_hash = hash(tuple(np.round(query_embedding[:10], 2)))
        except Exception:
            emb_hash = hash(str(query))
        key = hash(
            (
                emb_hash,
                query,
                query_type,
                tuple(keywords) if keywords else None,
                tuple(entities) if entities else None,
                top_k,
            )
        )
        logger.debug(f"Generated cache key: {key} for query: '{query}'")
        return key

    def _prepare_activations(self, top_k):
        """Pre-fetch activations to speed up post-processing."""
        logger.debug("Preparing activations for post-processing.")
        if self.activation_manager:
            return self.activation_manager.get_activated_memories(threshold=0.1)
        return {}

    def _fallback_retrieval(self, query_embedding, top_k=10):
        """Fallback retrieval method when primary strategy fails."""
        if not self.memory_store_adapter:
            return []
        try:
            logger.warning("Primary retrieval failed, using fallback method")
            fallback_results = self.memory_store_adapter.search_by_vector(
                query_embedding, limit=top_k
            )
            memories = [
                {
                    "memory_id": item.get("id"),
                    "relevance_score": item.get("score", 0.5),
                    "content": item.get("content", ""),
                    "metadata": item.get("metadata", {}),
                }
                for item in fallback_results
            ]
            logger.debug(f"Fallback retrieved {len(memories)} memories")
            return memories
        except Exception as e:
            logger.error(f"Even fallback retrieval failed: {e}")
            import traceback

            traceback.print_exc()
            return []

    def post_process_results(
        self, memories, query_embedding, query, current_time=None, context=None
    ):
        """
        Apply consistent post-processing to retrieval results with parallel execution where possible.
        """
        if not memories:
            logger.debug("No memories to process; returning empty list.")
            return []

        if current_time is None:
            current_time = time.time()

        if context is None:
            context = {}

        processed_memories = memories.copy()

        if len(memories) > 20 and self.max_workers > 1:
            logger.debug("Processing a large number of memories in parallel.")
            if self.activation_manager:
                _prefetched_activations = context.get("prefetched_activations")

                def process_memory_activation(memory):
                    memory_id = memory.get("memory_id")
                    if memory_id is not None and self.activation_manager:
                        relevance = memory.get("relevance_score", 0.5)
                        activation_boost = min(0.2 + (relevance * 0.3), 0.5)
                        self.activation_manager.activate_memory(
                            memory_id, activation_boost, spread=True
                        )
                        if self.temporal_context:
                            self.temporal_context.update_timestamp(memory_id, current_time)
                    return memory

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_workers
                ) as executor:
                    processed_memories = list(executor.map(process_memory_activation, memories))
                logger.debug("Parallel activation update complete.")

            if self.semantic_coherence_processor and len(processed_memories) > 1:
                try:
                    logger.debug("Applying semantic coherence processing.")
                    coherent_memories = self.semantic_coherence_processor.process_results(
                        processed_memories, query_embedding, {"query": query}
                    )
                    if coherent_memories:
                        processed_memories = coherent_memories
                        logger.debug(
                            f"Semantic coherence applied; {len(processed_memories)} results remain."
                        )
                except Exception as e:
                    logger.error(f"Error in semantic coherence processing: {e}")

            if self.temporal_context and query:
                try:
                    base_context = {"current_time": current_time}
                    logger.debug("Applying temporal context adjustments.")
                    processed_memories = self.temporal_context.apply_temporal_context(
                        query=query, results=processed_memories, context=base_context
                    )
                    logger.debug(
                        f"Temporal context applied; {len(processed_memories)} results remain."
                    )
                except Exception as e:
                    logger.error(f"Error applying temporal context: {e}")

        else:
            logger.debug("Processing memories sequentially.")
            for memory in processed_memories:
                memory_id = memory.get("memory_id")
                if memory_id is not None and self.activation_manager:
                    relevance = memory.get("relevance_score", 0.5)
                    activation_boost = min(0.2 + (relevance * 0.3), 0.5)
                    self.activation_manager.activate_memory(
                        memory_id, activation_boost, spread=True
                    )
                    if self.temporal_context:
                        self.temporal_context.update_timestamp(memory_id, current_time)

            if self.semantic_coherence_processor and len(processed_memories) > 1:
                try:
                    logger.debug("Applying semantic coherence sequentially.")
                    coherent_memories = self.semantic_coherence_processor.process_results(
                        processed_memories, query_embedding, {"query": query}
                    )
                    if coherent_memories:
                        processed_memories = coherent_memories
                        logger.debug(
                            f"Semantic coherence applied; {len(processed_memories)} results remain."
                        )
                except Exception as e:
                    logger.error(f"Error in semantic coherence processing: {e}")

            if self.temporal_context and query:
                try:
                    base_context = {"current_time": current_time}
                    logger.debug("Applying temporal context sequentially.")
                    processed_memories = self.temporal_context.apply_temporal_context(
                        query=query, results=processed_memories, context=base_context
                    )
                    logger.debug(
                        f"Temporal context applied; {len(processed_memories)} results remain."
                    )
                except Exception as e:
                    logger.error(f"Error applying temporal context: {e}")

        processed_memories.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        logger.debug("Post-processing complete; results sorted by relevance.")
        return processed_memories

    def get_cache_stats(self):
        """Get statistics about the query cache."""
        return {
            "cache_size": len(self.query_cache),
            "max_cache_size": self.max_cache_size,
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "hit_ratio": self.cache_stats["hits"]
            / max(1, (self.cache_stats["hits"] + self.cache_stats["misses"])),
            "last_query_time": self.cache_stats["last_query_time"],
        }

    def clear_cache(self):
        """Clear the query cache."""
        self.query_cache = {}
        self.cache_stats["hits"] = 0
        self.cache_stats["misses"] = 0
        logger.debug("Cache cleared.")
