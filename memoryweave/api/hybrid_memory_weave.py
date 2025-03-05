"""
Hybrid MemoryWeave API with adaptive chunking for memory efficiency.

This module provides an efficient middle-ground between standard single-vector
and full chunking approaches, optimizing memory usage while maintaining
retrieval quality.
"""

import logging
import time
from typing import Any

import numpy as np

from memoryweave.api.llm_provider import LLMProvider
from memoryweave.api.memory_weave import MemoryWeaveAPI
from memoryweave.components.retrieval_strategies.hybrid_fabric_strategy import HybridFabricStrategy
from memoryweave.components.text_chunker import TextChunker
from memoryweave.storage.hybrid_memory_store import HybridMemoryAdapter, HybridMemoryStore

logger = logging.getLogger(__name__)


class HybridMemoryWeaveAPI(MemoryWeaveAPI):
    """
    Memory-efficient MemoryWeave API with adaptive chunking and hierarchical embeddings.

    This API implements a middle-ground approach between single-vector and full chunking:
    - Uses full embedding for short texts
    - Creates strategic, sparse chunks for longer texts
    - Implements importance-based chunking with keyword enhancement
    - Maintains low memory footprint while preserving context awareness
    """

    def __init__(
        self,
        model_name: str = "unsloth/Llama-3.2-3B-Instruct",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        max_memories: int = 1000,
        enable_category_management: bool = True,
        enable_personal_attributes: bool = True,
        enable_semantic_coherence: bool = True,
        enable_dynamic_thresholds: bool = True,
        consolidation_interval: int = 100,
        show_progress_bar: bool = False,
        debug: bool = False,
        llm_provider: LLMProvider | None = None,
        **model_kwargs,
    ):
        """
        Initialize the HybridMemoryWeaveAPI.

        Args:
            model_name: Name of the language model to use
            embedding_model_name: Name of the embedding model to use
            device: Device to use for computation
            max_memories: Maximum number of memories to store
            enable_category_management: Whether to enable category management
            enable_personal_attributes: Whether to enable personal attribute extraction
            enable_semantic_coherence: Whether to enable semantic coherence checking
            enable_dynamic_thresholds: Whether to enable dynamic threshold adjustment
            consolidation_interval: Interval for memory consolidation
            show_progress_bar: Whether to show progress bars for embedding generation
            debug: Whether to enable debug logging
            **model_kwargs: Additional arguments for the language model
        """
        # Initialize chunking components first
        self.text_chunker = TextChunker()
        self.text_chunker.initialize({
            "chunk_size": 300,  # Larger chunks than the full chunked version
            "chunk_overlap": 30,  # Less overlap to save memory
            "min_chunk_size": 50,
            "respect_paragraphs": True,
            "respect_sentences": True,
        })

        # Initialize hybrid memory store
        self.hybrid_memory_store = HybridMemoryStore()
        self.hybrid_memory_adapter = HybridMemoryAdapter(self.hybrid_memory_store)

        # Set thresholds for adaptive chunking
        self.adaptive_chunk_threshold = 800  # Character count that triggers chunking
        self.max_chunks_per_memory = 3  # Maximum number of chunks per memory
        self.importance_threshold = 0.6  # Threshold for keeping chunks
        self.enable_auto_chunking = True  # Whether to automatically chunk large texts

        # Override default memory store to use hybrid version
        self._memory_store_override = self.hybrid_memory_store
        self._memory_adapter_override = self.hybrid_memory_adapter
        self.llm_provider = llm_provider
        # Call parent constructor with overrides
        super().__init__(
            model_name=model_name,
            embedding_model_name=embedding_model_name,
            device=device,
            max_memories=max_memories,
            enable_category_management=enable_category_management,
            enable_personal_attributes=enable_personal_attributes,
            enable_semantic_coherence=enable_semantic_coherence,
            enable_dynamic_thresholds=enable_dynamic_thresholds,
            consolidation_interval=consolidation_interval,
            show_progress_bar=show_progress_bar,
            debug=debug,
            llm_provider=self.llm_provider,
            **model_kwargs,
        )

        # Replace strategy with hybrid version after parent initialization
        self._setup_hybrid_strategy()

        # Track which memories are chunked
        self.chunked_memory_ids = set()

    def _setup_hybrid_strategy(self):
        """Set up the hybrid fabric strategy to replace the standard strategy."""
        # Create hybrid strategy
        self.hybrid_strategy = HybridFabricStrategy(
            memory_store=self.hybrid_memory_adapter,
            associative_linker=self.associative_linker,
            temporal_context=self.temporal_context,
            activation_manager=self.activation_manager,
        )

        # Initialize with optimized parameters
        params = {
            "confidence_threshold": self.strategy.confidence_threshold,
            "similarity_weight": 0.5,
            "associative_weight": 0.2,
            "temporal_weight": 0.2,
            "activation_weight": 0.1,
            "use_keyword_filtering": True,
            "keyword_boost_factor": 0.3,
            "max_chunks_per_memory": self.max_chunks_per_memory,
            "prioritize_full_embeddings": True,
            "debug": self.debug,
        }
        self.hybrid_strategy.initialize(params)

        # Replace the strategy
        self.strategy = self.hybrid_strategy

        # Update retrieval orchestrator
        if hasattr(self, "retrieval_orchestrator"):
            self.retrieval_orchestrator.strategy = self.strategy

    def add_memory(self, text: str, metadata: dict[str, Any] = None) -> str:
        """
        Store a memory with adaptive chunking for efficient memory usage.

        This method adaptively chunks large texts while keeping the total
        number of embeddings low through importance filtering.

        Args:
            text: The text to store
            metadata: Optional metadata for the memory

        Returns:
            Memory ID of the stored memory
        """
        logger.debug(f"Adding memory with adaptive chunking: {text[:100]}...")

        # Add default metadata if not provided
        if metadata is None:
            metadata = {"type": "manual", "created_at": time.time(), "importance": 0.6}
        elif "created_at" not in metadata:
            metadata["created_at"] = time.time()

        # Determine if we should chunk this text
        should_chunk = self.enable_auto_chunking and len(text) > self.adaptive_chunk_threshold

        # For small texts, use standard storage
        if not should_chunk:
            # Create a single embedding
            embedding = self.embedding_model.encode(text, show_progress_bar=self.show_progress_bar)

            # Add to memory store
            mem_id = self.hybrid_memory_store.add(embedding, text, metadata)
            
            # Critical fix: Invalidate BOTH adapters to ensure consistency
            self.hybrid_memory_adapter.invalidate_cache()
            # Also invalidate the standard adapter if we have access to it
            if hasattr(self, "memory_store_adapter"):
                self.memory_store_adapter.invalidate_cache()

            # Add to category if enabled
            if self.category_manager:
                self.category_manager.add_to_category(mem_id, embedding)

            logger.debug(f"Added memory {mem_id} without chunking")
            return mem_id

        # For large texts, use adaptive chunking
        # 1. Create chunks
        chunks = self.text_chunker.create_chunks(text, metadata)

        # 2. Calculate full embedding for the entire text
        full_embedding = self.embedding_model.encode(text, show_progress_bar=self.show_progress_bar)

        # 3. Select important chunks (adaptive chunking)
        selected_chunks, chunk_embeddings = self._select_important_chunks(chunks, full_embedding)

        # If there are no important chunks, just use the full embedding
        if not selected_chunks:
            mem_id = self.hybrid_memory_store.add(full_embedding, text, metadata)
            # Critical fix: Invalidate BOTH adapters
            self.hybrid_memory_adapter.invalidate_cache()
            if hasattr(self, "memory_store_adapter"):
                self.memory_store_adapter.invalidate_cache()
            return mem_id

        # 4. Add to hybrid memory store
        mem_id = self.hybrid_memory_store.add_hybrid(
            full_embedding=full_embedding,
            chunks=selected_chunks,
            chunk_embeddings=chunk_embeddings,
            original_content=text,
            metadata=metadata,
        )
        
        # Critical fix: Invalidate BOTH adapters
        self.hybrid_memory_adapter.invalidate_cache()
        if hasattr(self, "memory_store_adapter"):
            self.memory_store_adapter.invalidate_cache()

        # Track as chunked memory
        self.chunked_memory_ids.add(mem_id)

        # Add to category if enabled (using full embedding)
        if self.category_manager:
            self.category_manager.add_to_category(mem_id, full_embedding)

        # Track memories since consolidation
        self.memories_since_consolidation += 1

        # Perform consolidation if needed
        if self.memories_since_consolidation >= self.consolidation_interval:
            self._consolidate_memories()

        logger.debug(f"Added hybrid memory {mem_id} with {len(selected_chunks)} selected chunks")
        return mem_id

    def _select_important_chunks(
        self, chunks: list[dict[str, Any]], full_embedding: np.ndarray
    ) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
        """
        Select the most important chunks to embed and store.

        This method rates chunks by information density and importance,
        keeping only the most valuable chunks to reduce memory usage.

        Args:
            chunks: list of chunk dictionaries
            full_embedding: Embedding of the full text

        Returns:
            tuple of (selected_chunks, chunk_embeddings)
        """
        if not chunks:
            return [], []

        # Limit total number of chunks
        if len(chunks) > self.max_chunks_per_memory:
            # Extract keywords from chunks to help identify important ones
            all_keywords = []
            for chunk in chunks:
                # Use query analyzer to extract keywords if available
                chunk_text = chunk["text"]
                if hasattr(self, "query_analyzer") and self.query_analyzer:
                    keywords = self.query_analyzer.extract_keywords(chunk_text)
                    all_keywords.append(keywords)
                else:
                    # Simple keyword extraction fallback
                    words = chunk_text.lower().split()
                    # Filter out common words and keep longer words
                    keywords = [w for w in words if len(w) > 5 and w.isalpha()]
                    all_keywords.append(keywords[:10])  # Limit to 10 keywords

            # Calculate chunk scores
            chunk_scores = []
            for i, chunk in enumerate(chunks):
                # Score based on multiple factors
                score = 0.0

                # 1. Position score - first and last chunks often have key information
                if i == 0 or i == len(chunks) - 1:
                    score += 0.3

                # 2. Keyword density
                if i < len(all_keywords):
                    score += min(0.5, len(all_keywords[i]) * 0.05)

                # 3. Length score - longer chunks may have more information
                score += min(0.2, len(chunk["text"]) / 1000)

                chunk_scores.append(score)

            # Select top N chunks based on scores
            sorted_indices = np.argsort(chunk_scores)[::-1]
            top_indices = sorted_indices[: self.max_chunks_per_memory]
            chunks = [chunks[i] for i in top_indices]

        # Create embeddings for the selected chunks
        chunk_embeddings = []
        for chunk in chunks:
            chunk_text = chunk["text"]
            embedding = self.embedding_model.encode(
                chunk_text, show_progress_bar=self.show_progress_bar
            )
            chunk_embeddings.append(embedding)

        return chunks, chunk_embeddings

    def add_conversation_memory(
        self, turns: list[dict[str, str]], metadata: dict[str, Any] = None
    ) -> str:
        """
        Add a conversation memory with efficient chunking.

        Args:
            turns: list of conversation turns with "role" and "content" keys
            metadata: Optional metadata for the memory

        Returns:
            Memory ID of the stored memory
        """
        if not turns:
            raise ValueError("Conversation must contain at least one turn")

        # Build full text
        full_text = "\n".join(
            f"{turn.get('role', 'unknown')}: {turn.get('content', '')}" for turn in turns
        )

        # Create metadata if not provided
        if metadata is None:
            metadata = {
                "type": "conversation",
                "created_at": time.time(),
                "importance": 0.7,
                "turn_count": len(turns),
            }
        else:
            metadata = metadata.copy()
            metadata["type"] = metadata.get("type", "conversation")
            metadata["turn_count"] = len(turns)

        # For conversations, use a different chunking strategy
        # Each turn is important, so we want to preserve all of them
        total_length = len(full_text)
        if total_length > self.adaptive_chunk_threshold:
            # Create the full embedding
            full_embedding = self.embedding_model.encode(
                full_text, show_progress_bar=self.show_progress_bar
            )

            # Process conversation into chunks based on turns
            if len(turns) > self.max_chunks_per_memory:
                # If there are too many turns, group them
                chunks = self._adaptive_conversation_chunking(turns, metadata)
            else:
                # Otherwise, process each turn as its own chunk
                chunks = []
                for i, turn in enumerate(turns):
                    role = turn.get("role", "unknown")
                    content = turn.get("content", "")
                    chunk_text = f"{role}: {content}"

                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "chunk_index": i,
                        "is_conversation": True,
                        "role": role,
                    })

                    chunks.append({"text": chunk_text, "metadata": chunk_metadata})

            # Create embeddings for each chunk
            chunk_embeddings = []
            for chunk in chunks:
                chunk_text = chunk["text"]
                embedding = self.embedding_model.encode(
                    chunk_text, show_progress_bar=self.show_progress_bar
                )
                chunk_embeddings.append(embedding)

            # Add as hybrid memory
            mem_id = self.hybrid_memory_store.add_hybrid(
                full_embedding=full_embedding,
                chunks=chunks,
                chunk_embeddings=chunk_embeddings,
                original_content=full_text,
                metadata=metadata,
            )
        else:
            # If the conversation is short, just use a single embedding
            embedding = self.embedding_model.encode(
                full_text, show_progress_bar=self.show_progress_bar
            )
            mem_id = self.hybrid_memory_store.add(embedding, full_text, metadata)

        # Update caches
        self.hybrid_memory_adapter.invalidate_cache()

        # If it was chunked, track it
        if total_length > self.adaptive_chunk_threshold:
            self.chunked_memory_ids.add(mem_id)

        # Add to category if enabled
        if self.category_manager:
            # Use the full embedding for categorization
            if total_length > self.adaptive_chunk_threshold:
                self.category_manager.add_to_category(mem_id, full_embedding)
            else:
                self.category_manager.add_to_category(mem_id, embedding)

        logger.debug(f"Added conversation memory {mem_id}")
        return mem_id

    def _adaptive_conversation_chunking(
        self, turns: list[dict[str, str]], metadata: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Adaptively chunk a conversation to preserve important context.

        This method groups turns intelligently to create meaningful chunks
        while staying within memory limits.

        Args:
            turns: list of conversation turns
            metadata: Memory metadata

        Returns:
            list of chunk dictionaries
        """
        # If there are too many turns, we need to group them
        max_chunks = self.max_chunks_per_memory
        turn_count = len(turns)

        if turn_count <= max_chunks:
            # No grouping needed
            chunks = []
            for i, turn in enumerate(turns):
                role = turn.get("role", "unknown")
                content = turn.get("content", "")
                chunk_text = f"{role}: {content}"

                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "is_conversation": True,
                    "role": role,
                })

                chunks.append({"text": chunk_text, "metadata": chunk_metadata})
            return chunks

        # Strategy: Always keep the first and last turns, group the middle ones
        chunks = []

        # Always include the first turn
        first_turn = turns[0]
        first_role = first_turn.get("role", "unknown")
        first_content = first_turn.get("content", "")
        first_text = f"{first_role}: {first_content}"

        first_metadata = metadata.copy()
        first_metadata.update({
            "chunk_index": 0,
            "is_conversation": True,
            "role": first_role,
            "is_first": True,
        })

        chunks.append({"text": first_text, "metadata": first_metadata})

        # For middle turns, create groups
        remaining_chunks = max_chunks - 2  # -2 for first and last
        middle_turns = turns[1:-1]
        middle_count = len(middle_turns)

        if middle_count > 0:
            # Determine group size
            group_size = (middle_count + remaining_chunks - 1) // remaining_chunks

            # Create groups
            for i in range(0, middle_count, group_size):
                group = middle_turns[i : i + group_size]
                group_text = "\n".join(
                    f"{turn.get('role', 'unknown')}: {turn.get('content', '')}" for turn in group
                )

                group_metadata = metadata.copy()
                group_metadata.update({
                    "chunk_index": len(chunks),
                    "is_conversation": True,
                    "turn_range": (i + 1, min(i + group_size, middle_count)),
                })

                chunks.append({"text": group_text, "metadata": group_metadata})

        # Always include the last turn
        last_turn = turns[-1]
        last_role = last_turn.get("role", "unknown")
        last_content = last_turn.get("content", "")
        last_text = f"{last_role}: {last_content}"

        last_metadata = metadata.copy()
        last_metadata.update({
            "chunk_index": len(chunks),
            "is_conversation": True,
            "role": last_role,
            "is_last": True,
        })

        chunks.append({"text": last_text, "metadata": last_metadata})

        return chunks

    def configure_chunking(self, **kwargs) -> None:
        """
        Configure chunking parameters.

        Args:
            **kwargs: Chunking parameters to configure
        """
        # API-level parameters
        if "adaptive_chunk_threshold" in kwargs:
            self.adaptive_chunk_threshold = kwargs["adaptive_chunk_threshold"]
        if "enable_auto_chunking" in kwargs:
            self.enable_auto_chunking = kwargs["enable_auto_chunking"]
        if "max_chunks_per_memory" in kwargs:
            self.max_chunks_per_memory = kwargs["max_chunks_per_memory"]
        if "importance_threshold" in kwargs:
            self.importance_threshold = kwargs["importance_threshold"]

        # TextChunker parameters
        chunker_params = {}
        if "chunk_size" in kwargs:
            chunker_params["chunk_size"] = kwargs["chunk_size"]
        if "chunk_overlap" in kwargs:
            chunker_params["chunk_overlap"] = kwargs["chunk_overlap"]
        if "min_chunk_size" in kwargs:
            chunker_params["min_chunk_size"] = kwargs["min_chunk_size"]
        if "respect_paragraphs" in kwargs:
            chunker_params["respect_paragraphs"] = kwargs["respect_paragraphs"]
        if "respect_sentences" in kwargs:
            chunker_params["respect_sentences"] = kwargs["respect_sentences"]

        if chunker_params:
            self.text_chunker.initialize(chunker_params)

        # HybridFabricStrategy parameters
        strategy_params = {}
        if "keyword_boost_factor" in kwargs:
            strategy_params["keyword_boost_factor"] = kwargs["keyword_boost_factor"]
        if "prioritize_full_embeddings" in kwargs:
            strategy_params["prioritize_full_embeddings"] = kwargs["prioritize_full_embeddings"]
        if "use_keyword_filtering" in kwargs:
            strategy_params["use_keyword_filtering"] = kwargs["use_keyword_filtering"]

        if strategy_params:
            current_params = {
                "confidence_threshold": self.strategy.confidence_threshold,
                "similarity_weight": self.strategy.similarity_weight,
                "associative_weight": self.strategy.associative_weight,
                "temporal_weight": self.strategy.temporal_weight,
                "activation_weight": self.strategy.activation_weight,
                "max_chunks_per_memory": self.max_chunks_per_memory,
                "debug": self.debug,
            }
            # Update with new parameters
            current_params.update(strategy_params)
            # Re-initialize strategy
            self.strategy.initialize(current_params)

    def get_chunking_statistics(self) -> dict[str, Any]:
        """
        Get statistics about memory usage and chunking.

        Returns:
            dictionary with chunking statistics
        """
        try:
            stats = {
                "total_memories": len(self.hybrid_memory_store.get_all()),
                "chunked_memories": len(self.chunked_memory_ids),
                "total_chunks": self.hybrid_memory_store.get_chunk_count(),
                "avg_chunks_per_memory": self.hybrid_memory_store.get_average_chunks_per_memory(),
                "adaptive_chunk_threshold": self.adaptive_chunk_threshold,
                "max_chunks_per_memory": self.max_chunks_per_memory,
                "enable_auto_chunking": self.enable_auto_chunking,
                "memory_usage": {
                    "full_embeddings": len(self.hybrid_memory_store.get_all()),
                    "chunk_embeddings": self.hybrid_memory_store.get_chunk_count(),
                    "total_embeddings": (
                        len(self.hybrid_memory_store.get_all())
                        + self.hybrid_memory_store.get_chunk_count()
                    ),
                },
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting chunking statistics: {e}")
            return {}

    def chat(self, user_message: str, max_new_tokens: int = 512) -> str:
        """
        Process user message and generate a response using hybrid memory retrieval.

        This method leverages the hybrid approach for more efficient memory usage.

        Args:
            user_message: User's message
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            Assistant's response
        """
        start_time = time.time()

        # Step 1: Query analysis
        query_info = self._analyze_query(user_message)
        _query_obj, adapted_params, expanded_keywords, query_type, entities = query_info

        # Step 2: Compute query embedding
        query_embedding = self._compute_embedding(user_message)
        if query_embedding is None:
            return "Sorry, an error occurred while processing your request."

        # Step 3: Add keyword filtering if available
        if hasattr(self, "query_analyzer") and self.query_analyzer:
            keywords = self.query_analyzer.extract_keywords(user_message)
            if keywords and len(keywords) > 0:
                if adapted_params is None:
                    adapted_params = {}
                adapted_params["important_keywords"] = keywords

        # Step 4: Retrieve memories
        relevant_memories = self.retrieval_orchestrator.retrieve(
            query_embedding=query_embedding,
            query=user_message,
            query_type=query_type,
            expanded_keywords=expanded_keywords,
            entities=entities,
            adapted_params=adapted_params,
            top_k=adapted_params.get("max_results", 10),
        )

        # Step 5: Extract personal attributes if enabled
        self._extract_personal_attributes(user_message, time.time())

        # Step 6: Construct prompt
        prompt = self.prompt_builder.build_chat_prompt(
            user_message=user_message,
            memories=relevant_memories,
            conversation_history=self.conversation_history,
            query_type=query_type,
        )
        if self.debug:
            print("===== Prompt Start =====")
            print(prompt)
            print("===== Prompt End =====")
        # Step 7: Generate response
        assistant_reply = self.llm_provider.generate(prompt=prompt, max_new_tokens=max_new_tokens)

        # Step 8: Store interaction
        self._store_hybrid_interaction(user_message, assistant_reply, time.time())

        # Step 9: Update history and statistics
        self._update_conversation_history(user_message, assistant_reply)
        self._update_retrieval_stats(start_time, len(relevant_memories))

        return assistant_reply

    def _store_hybrid_interaction(self, user_message: str, assistant_reply: str, timestamp: float):
        """
        Store conversation messages efficiently with hybrid approach.

        Args:
            user_message: User's message
            assistant_reply: Assistant's reply
            timestamp: Timestamp when the interaction occurred
        """
        try:
            # Determine if each message warrants chunking
            user_should_chunk = len(user_message) > self.adaptive_chunk_threshold
            assistant_should_chunk = len(assistant_reply) > self.adaptive_chunk_threshold

            # Store user message
            user_metadata = {
                "type": "user_message",
                "created_at": timestamp,
                "conversation_id": id(self.conversation_history),
                "importance": 0.7,
            }

            if user_should_chunk:
                # Create full embedding and selected chunks
                user_embedding = self.embedding_model.encode(
                    user_message, show_progress_bar=self.show_progress_bar
                )
                user_chunks = self.text_chunker.create_chunks(user_message, user_metadata)
                selected_chunks, chunk_embeddings = self._select_important_chunks(
                    user_chunks, user_embedding
                )

                # Add as hybrid memory
                user_mem_id = self.hybrid_memory_store.add_hybrid(
                    full_embedding=user_embedding,
                    chunks=selected_chunks,
                    chunk_embeddings=chunk_embeddings,
                    original_content=user_message,
                    metadata=user_metadata,
                )
                self.chunked_memory_ids.add(user_mem_id)
            else:
                # Add as regular memory
                user_emb = self.embedding_model.encode(
                    user_message, show_progress_bar=self.show_progress_bar
                )
                user_mem_id = self.hybrid_memory_store.add(user_emb, user_message, user_metadata)

            # Store assistant message
            assistant_metadata = {
                "type": "assistant_message",
                "created_at": timestamp,
                "conversation_id": id(self.conversation_history),
                "importance": 0.5,
            }

            if assistant_should_chunk:
                # Create full embedding and selected chunks
                assistant_embedding = self.embedding_model.encode(
                    assistant_reply, show_progress_bar=self.show_progress_bar
                )
                assistant_chunks = self.text_chunker.create_chunks(
                    assistant_reply, assistant_metadata
                )
                selected_chunks, chunk_embeddings = self._select_important_chunks(
                    assistant_chunks, assistant_embedding
                )

                # Add as hybrid memory
                assistant_mem_id = self.hybrid_memory_store.add_hybrid(
                    full_embedding=assistant_embedding,
                    chunks=selected_chunks,
                    chunk_embeddings=chunk_embeddings,
                    original_content=assistant_reply,
                    metadata=assistant_metadata,
                )
                self.chunked_memory_ids.add(assistant_mem_id)
            else:
                # Add as regular memory
                assistant_emb = self.embedding_model.encode(
                    assistant_reply, show_progress_bar=self.show_progress_bar
                )
                assistant_mem_id = self.hybrid_memory_store.add(
                    assistant_emb, assistant_reply, assistant_metadata
                )

            # Create associative link between messages
            if self.associative_linker:
                self.associative_linker.create_associative_link(user_mem_id, assistant_mem_id, 0.9)

            # Invalidate cache
            self.hybrid_memory_adapter.invalidate_cache()

        except Exception as e:
            logger.error(f"Error storing hybrid conversation in memory: {e}")
