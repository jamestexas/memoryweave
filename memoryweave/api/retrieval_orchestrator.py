import concurrent.futures
import logging
import os
import time
from importlib.util import find_spec

import numpy as np
from rich.logging import RichHandler

logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(markup=True)]
)
logger = logging.getLogger(__name__)


def determine_optimal_workers() -> int:
    """Determine optimal number of worker threads based on system resources."""
    cpu_count = os.cpu_count() or 2  # Default to 2 if CPU count is not available
    if find_spec("psutil") is None:
        max_workers = min(cpu_count + 1, 8)  # CPU count + 1, up to 8
        logger.warning(f"[bold yellow]psutil not found, using default of {max_workers} workers[/]")
        return max_workers
    # If we get here, psutil should be available and we can dynamically determine worker count
    try:
        import psutil

        # Try to get available memory with psutil
        # Get available memory in GB
        available_memory_gb = psutil.virtual_memory().available / (1024 * 1024 * 1024)

        # More workers for higher memory systems, up to 2x CPU count
        if available_memory_gb > 8:  # More than 8GB available
            max_workers = min(cpu_count * 2, 16)  # Up to 16 workers
        elif available_memory_gb > 4:  # 4-8GB available
            max_workers = min(cpu_count, 8)  # Up to 8 workers
        else:  # Less than 4GB available
            max_workers = max(2, cpu_count // 2)  # At least 2 workers
        return max_workers

    except Exception as e:
        # Fallback to safe default
        logger.warning(f"Error determining optimal worker count: {e}. Using default of 4.")
        max_workers = 4
    return max_workers


class RetrievalOrchestrator:
    """Orchestrates the memory retrieval process with parallel processing."""

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

        Args:
            strategy: The retrieval strategy to use
            activation_manager: Optional activation manager
            temporal_context: Optional temporal context processor
            semantic_coherence_processor: Optional semantic coherence processor
            memory_store_adapter: Optional memory store adapter
            debug: Whether to enable debug logging
            max_workers: Maximum number of concurrent worker threads
            enable_cache: Whether to enable query caching
            max_cache_size: Maximum number of cached queries
        """
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

        # Prepare thread pool executor for parallel processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        # Cache for recent queries
        self.query_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0, "last_query_time": 0}

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

        Args:
            query_embedding: Query embedding vector
            query: Original query text
            query_type: Type of query
            expanded_keywords: Expanded keywords for the query
            entities: Extracted entities from the query
            adapted_params: Adapted retrieval parameters
            top_k: Number of results to return

        Returns:
            List of retrieved memories with relevance scores
        """
        start_time = time.time()

        # Generate cache key if caching is enabled
        cache_key = None
        if self.enable_cache:
            cache_key = self._generate_cache_key(
                query_embedding, query, query_type, expanded_keywords, entities, top_k
            )

            # Check cache
            if cache_key in self.query_cache:
                self.cache_stats["hits"] += 1

                # Log query time if debugging
                if self.debug:
                    logger.debug(f"Cache hit for query: '{query}'")

                return self.query_cache[cache_key]

            self.cache_stats["misses"] += 1

        # Prepare context for retrieval (shared across all workers)
        current_time = time.time()
        retrieval_context = {
            "query": query,
            "query_type": query_type,
            "important_keywords": expanded_keywords or [],
            "extracted_entities": entities or [],
            "current_time": current_time,
            "memory_store": self.memory_store_adapter,
        }

        # Add adapted parameters if available
        if adapted_params:
            retrieval_context["adapted_retrieval_params"] = adapted_params

        # Execute retrieval with strategy and parallel processing for sub-tasks
        try:
            # Submit main retrieval task
            future_memories = self.executor.submit(
                self.strategy.retrieve,
                query_embedding=query_embedding,
                top_k=top_k,
                context=retrieval_context,
                query=query,
            )

            # While waiting for retrieval, prepare activation data in parallel
            future_activations = None
            if self.activation_manager:
                future_activations = self.executor.submit(self._prepare_activations, top_k)

            # Wait for main retrieval to complete (with timeout)
            try:
                memories = future_memories.result(timeout=9.0)  # 9 second timeout

                # If we got activations, they'll be used in post-processing
                activations_ready = future_activations.done() if future_activations else False

            except concurrent.futures.TimeoutError:
                logger.error("Retrieval strategy timeout, using fallback")
                memories = self._fallback_retrieval(query_embedding, top_k)
                activations_ready = False

            # Post-process results (may use prepared activations)
            if activations_ready:
                activation_data = future_activations.result()
                # Store activations in context for post-processing
                retrieval_context["prefetched_activations"] = activation_data

            processed_memories = self.post_process_results(
                memories, query_embedding, query, current_time, retrieval_context
            )

            # Cache the result if caching is enabled
            if self.enable_cache and cache_key is not None:
                if len(self.query_cache) >= self.max_cache_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self.query_cache))
                    del self.query_cache[oldest_key]

                self.query_cache[cache_key] = processed_memories

            # Update query time statistics
            query_time = time.time() - start_time
            self.cache_stats["last_query_time"] = query_time

            # Log retrieval time if debugging
            if self.debug:
                logger.debug(f"Retrieval completed in {query_time:.3f}s")
                logger.debug(f"Retrieved {len(processed_memories)} memories")

            return processed_memories

        except Exception as e:
            logger.error(f"Error using primary retrieval strategy: {e}")
            import traceback

            traceback.print_exc()

            # Attempt fallback retrieval
            memories = self._fallback_retrieval(query_embedding, top_k)

            # Post-process results
            processed_memories = self.post_process_results(
                memories, query_embedding, query, current_time
            )

            # Update query time statistics
            query_time = time.time() - start_time
            self.cache_stats["last_query_time"] = query_time

            return processed_memories

    def _generate_cache_key(self, query_embedding, query, query_type, keywords, entities, top_k):
        """
        Generate a cache key for a query.

        Creates a hash-based key using essential query features.
        """
        # Use a simplified hash for embedding to avoid floating point issues
        # Hash only the first few dimensions after rounding to save memory and computation
        try:
            emb_hash = hash(tuple(np.round(query_embedding[:10], 2)))
        except Exception:
            # Fallback if hashing the embedding fails
            emb_hash = hash(str(query))

        # Create composite key from query features
        return hash(
            (
                emb_hash,
                query,
                query_type,
                tuple(keywords) if keywords else None,
                tuple(entities) if entities else None,
                top_k,
            )
        )

    def _prepare_activations(self, top_k):
        """Pre-fetch activations to speed up post-processing."""
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

        Args:
            memories: Retrieved memories to post-process
            query_embedding: Query embedding vector
            query: Original query text
            current_time: Current timestamp (defaults to now)
            context: Additional context for post-processing

        Returns:
            Post-processed memories
        """
        if not memories:
            return []

        if current_time is None:
            current_time = time.time()

        if context is None:
            context = {}

        # Create a copy to avoid modifying original data
        processed_memories = memories.copy()

        # For larger result sets, use parallel processing
        if len(memories) > 20 and self.max_workers > 1:
            # Step 1: Update activations and timestamps (can be parallelized)
            if self.activation_manager:
                # We may have prefetched activations
                _prefetched_activations = context.get("prefetched_activations")

                def process_memory_activation(memory):
                    """Process activation for a single memory."""
                    memory_id = memory.get("memory_id")
                    if memory_id is not None and self.activation_manager:
                        # Update activation based on relevance
                        relevance = memory.get("relevance_score", 0.5)
                        activation_boost = min(0.2 + (relevance * 0.3), 0.5)
                        self.activation_manager.activate_memory(
                            memory_id, activation_boost, spread=True
                        )

                        # Update temporal context
                        if self.temporal_context:
                            self.temporal_context.update_timestamp(memory_id, current_time)

                    return memory

                # Process in parallel if we have multiple memories
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_workers
                ) as executor:
                    # Map function to all memories
                    processed_memories = list(executor.map(process_memory_activation, memories))

            # Step 2: Apply semantic coherence if enabled (must be sequential)
            if self.semantic_coherence_processor and len(processed_memories) > 1:
                try:
                    coherent_memories = self.semantic_coherence_processor.process_results(
                        processed_memories, query_embedding, {"query": query}
                    )
                    if coherent_memories:
                        processed_memories = coherent_memories
                        logger.debug(
                            f"Applied semantic coherence, now have {len(processed_memories)} results"
                        )
                except Exception as e:
                    logger.error(f"Error in semantic coherence processing: {e}")

            # Step 3: Apply temporal context if available (can be sequential)
            if self.temporal_context and query:
                try:
                    base_context = {"current_time": current_time}
                    processed_memories = self.temporal_context.apply_temporal_context(
                        query=query, results=processed_memories, context=base_context
                    )
                    logger.debug(
                        f"Applied temporal context, now have {len(processed_memories)} results"
                    )
                except Exception as e:
                    logger.error(f"Error applying temporal context: {e}")

        else:
            # For smaller sets, process sequentially (original implementation)
            # Update activations and timestamps
            for memory in processed_memories:
                memory_id = memory.get("memory_id")
                if memory_id is not None and self.activation_manager:
                    # Update activation based on relevance
                    relevance = memory.get("relevance_score", 0.5)
                    activation_boost = min(0.2 + (relevance * 0.3), 0.5)
                    self.activation_manager.activate_memory(
                        memory_id, activation_boost, spread=True
                    )

                    # Update temporal context
                    if self.temporal_context:
                        self.temporal_context.update_timestamp(memory_id, current_time)

            # Apply semantic coherence if enabled
            if self.semantic_coherence_processor and len(processed_memories) > 1:
                try:
                    coherent_memories = self.semantic_coherence_processor.process_results(
                        processed_memories, query_embedding, {"query": query}
                    )
                    if coherent_memories:
                        processed_memories = coherent_memories
                        logger.debug(
                            f"Applied semantic coherence, now have {len(processed_memories)} results"
                        )
                except Exception as e:
                    logger.error(f"Error in semantic coherence processing: {e}")

            # Apply temporal context if available
            if self.temporal_context and query:
                try:
                    base_context = {"current_time": current_time}
                    processed_memories = self.temporal_context.apply_temporal_context(
                        query=query, results=processed_memories, context=base_context
                    )
                    logger.debug(
                        f"Applied temporal context, now have {len(processed_memories)} results"
                    )
                except Exception as e:
                    logger.error(f"Error applying temporal context: {e}")

        # Sort by relevance score
        processed_memories.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)

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
