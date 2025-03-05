import logging
import time

from rich.logging import RichHandler

logging.basicConfig(
    level="INFO", format="%(message)s", datefmt="[%X]", handlers=[RichHandler(markup=True)]
)
logger = logging.getLogger(__name__)


class RetrievalOrchestrator:
    """Orchestrates the memory retrieval process."""

    def __init__(
        self,
        strategy,
        activation_manager=None,
        temporal_context=None,
        semantic_coherence_processor=None,
        memory_store_adapter=None,
        debug=False,
    ):
        self.strategy = strategy
        self.activation_manager = activation_manager
        self.temporal_context = temporal_context
        self.semantic_coherence_processor = semantic_coherence_processor
        self.memory_store_adapter = memory_store_adapter
        self.debug = debug

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
        """Execute the retrieval process with all necessary steps."""
        # Prepare context for retrieval
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

        # Execute retrieval with strategy
        try:
            logger.debug("Retrieving memories using strategy")
            memories = self.strategy.retrieve(
                query_embedding=query_embedding,
                top_k=top_k,
                context=retrieval_context,
            )
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

        return processed_memories

    def post_process_results(self, memories, query_embedding, query, current_time=None):
        """Apply consistent post-processing to retrieval results."""
        if not memories:
            return []

        if current_time is None:
            current_time = time.time()

        # Update activations and timestamps
        for memory in memories:
            memory_id = memory.get("memory_id")
            if memory_id is not None and self.activation_manager:
                # Update activation based on relevance
                relevance = memory.get("relevance_score", 0.5)
                activation_boost = min(0.2 + (relevance * 0.3), 0.5)
                self.activation_manager.activate_memory(memory_id, activation_boost, spread=True)

                # Update temporal context
                if self.temporal_context:
                    self.temporal_context.update_timestamp(memory_id, current_time)

        # Apply semantic coherence if enabled
        if self.semantic_coherence_processor and len(memories) > 1:
            try:
                coherent_memories = self.semantic_coherence_processor.process_results(
                    memories, query_embedding, {"query": query}
                )
                if coherent_memories:
                    memories = coherent_memories
                    logger.debug(f"Applied semantic coherence, now have {len(memories)} results")
            except Exception as e:
                logger.error(f"Error in semantic coherence processing: {e}")

        # Apply temporal context if available
        if self.temporal_context and query:
            try:
                base_context = {"current_time": current_time}
                memories = self.temporal_context.apply_temporal_context(
                    query=query, results=memories, context=base_context
                )
                logger.debug(f"Applied temporal context, now have {len(memories)} results")
            except Exception as e:
                logger.error(f"Error applying temporal context: {e}")

        # Sort by relevance score
        memories.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)

        return memories

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
