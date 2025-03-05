"""
Enhanced MemoryWeave API with chunking support for large contexts.

This module extends the base MemoryWeaveAPI to support text chunking,
enabling better handling of large contexts like long conversations,
documents, or detailed memories.
"""

import logging
import time
from typing import Any

import numpy as np

from memoryweave.api.llm_provider import LLMProvider
from memoryweave.api.memory_weave import MemoryWeaveAPI
from memoryweave.components.retrieval_strategies.chunked_fabric_strategy import (
    ChunkedFabricStrategy,
)
from memoryweave.components.text_chunker import TextChunker
from memoryweave.storage.chunked_memory_store import ChunkedMemoryAdapter, ChunkedMemoryStore

logger = logging.getLogger(__name__)


class ChunkedMemoryWeaveAPI(MemoryWeaveAPI):
    """
    Enhanced MemoryWeave API with chunking support for large contexts.

    This API extends the base MemoryWeaveAPI with features for:
    1. Chunking large texts for better embedding representation
    2. Managing multi-vector memories
    3. Enhanced retrieval for large contexts
    4. Improved conversation history handling
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
        Initialize the ChunkedMemoryWeaveAPI.

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
            "chunk_size": 200,
            "chunk_overlap": 50,
            "min_chunk_size": 30,
            "respect_paragraphs": True,
            "respect_sentences": True,
        })
        self.llm_provider = llm_provider
        # Replace standard memory store with chunked version
        self.chunked_memory_store = ChunkedMemoryStore()
        self.chunked_memory_adapter = ChunkedMemoryAdapter(self.chunked_memory_store)

        # Override default memory store to use chunked version
        self._memory_store_override = self.chunked_memory_store
        self._memory_adapter_override = self.chunked_memory_adapter

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

        # Replace strategy with chunked version after parent initialization
        self._setup_chunked_strategy()

        # Chunking configuration
        self.auto_chunk_threshold = 500  # Character count that triggers automatic chunking
        self.enable_auto_chunking = True  # Whether to automatically chunk large texts
        self.max_chunk_count = 10  # Maximum number of chunks per memory

        # Track which memories are chunked
        self.chunked_memory_ids = set()

    def _setup_chunked_strategy(self):
        """Set up the chunked fabric strategy to replace the standard strategy."""
        # Create chunked strategy
        self.chunked_strategy = ChunkedFabricStrategy(
            memory_store=self.chunked_memory_adapter,
            associative_linker=self.associative_linker,
            temporal_context=self.temporal_context,
            activation_manager=self.activation_manager,
        )

        # Initialize with same parameters as original strategy
        params = {
            "confidence_threshold": self.strategy.confidence_threshold,
            "similarity_weight": self.strategy.similarity_weight,
            "associative_weight": self.strategy.associative_weight,
            "temporal_weight": self.strategy.temporal_weight,
            "activation_weight": self.strategy.activation_weight,
            "max_associative_hops": self.strategy.max_associative_hops,
            "debug": self.debug,
            # Add chunking-specific parameters
            "chunk_weight_decay": 0.8,
            "max_chunks_per_memory": 3,
            "combine_chunk_scores": True,
            "prioritize_coherent_chunks": True,
        }
        self.chunked_strategy.initialize(params)

        # Replace the strategy
        self.strategy = self.chunked_strategy

        # Update retrieval orchestrator to use the new strategy
        if hasattr(self, "retrieval_orchestrator"):
            self.retrieval_orchestrator.strategy = self.strategy

    def add_memory(self, text: str, metadata: dict[str, Any] = None) -> str:
        """
        Store a memory with chunking for large texts.

        This method automatically chunks large texts and stores multiple
        embeddings per memory for better representation.

        Args:
            text: The text to store
            metadata: Optional metadata for the memory

        Returns:
            Memory ID of the stored memory
        """
        logger.debug(f"Adding memory: {text[:100]}...")

        # Add default metadata if not provided
        if metadata is None:
            metadata = {"type": "manual", "created_at": time.time(), "importance": 0.6}
        elif "created_at" not in metadata:
            metadata["created_at"] = time.time()

        # Determine if we should chunk this text
        should_chunk = self.enable_auto_chunking and len(text) > self.auto_chunk_threshold

        # For small texts, use standard storage
        if not should_chunk:
            # Create a single embedding
            embedding = self.embedding_model.encode(text, show_progress_bar=self.show_progress_bar)

            # Add to memory store
            mem_id = self.chunked_memory_store.add(embedding, text, metadata)

            # Critical fix: Invalidate BOTH adapters
            self.chunked_memory_adapter.invalidate_cache()
            # Also invalidate the standard adapter if we have access to it
            if hasattr(self, "memory_store_adapter"):
                self.memory_store_adapter.invalidate_cache()

            # Add to category if enabled
            if self.category_manager:
                self.category_manager.add_to_category(mem_id, embedding)

            logger.debug(f"Added memory {mem_id} without chunking")
            return mem_id

        # For large texts, use chunked storage
        # Create chunks
        chunks = self.text_chunker.create_chunks(text, metadata)

        # Limit number of chunks if needed
        if len(chunks) > self.max_chunk_count:
            logger.debug(f"Limiting chunks from {len(chunks)} to {self.max_chunk_count}")
            chunks = chunks[: self.max_chunk_count]

        # Create embeddings for each chunk
        embeddings = []
        for chunk in chunks:
            chunk_text = chunk["text"]
            embedding = self.embedding_model.encode(
                chunk_text, show_progress_bar=self.show_progress_bar
            )
            embeddings.append(embedding)

        # Add to chunked memory store
        mem_id = self.chunked_memory_store.add_chunked(chunks, embeddings, text, metadata)

        # Critical fix: Invalidate BOTH adapters
        self.chunked_memory_adapter.invalidate_cache()
        if hasattr(self, "memory_store_adapter"):
            self.memory_store_adapter.invalidate_cache()

        # Track as chunked memory
        self.chunked_memory_ids.add(mem_id)

        # Add to category if enabled (using combined embedding)
        if self.category_manager:
            # Use average embedding for categorization
            combined_embedding = np.mean(embeddings, axis=0)
            self.category_manager.add_to_category(mem_id, combined_embedding)

        # Track memories since consolidation
        self.memories_since_consolidation += 1

        # Perform consolidation if needed
        if self.memories_since_consolidation >= self.consolidation_interval:
            self._consolidate_memories()

        logger.debug(f"Added chunked memory {mem_id} with {len(chunks)} chunks")
        return mem_id

    def add_conversation_memory(
        self, turns: list[dict[str, str]], metadata: dict[str, Any] = None
    ) -> str:
        """
        Add a conversation memory with specialized conversation chunking.

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

        # Create conversation chunks
        chunks = self.text_chunker.process_conversation(turns)

        # Create embeddings for each chunk
        embeddings = []
        for chunk in chunks:
            chunk_text = chunk["text"]
            embedding = self.embedding_model.encode(
                chunk_text, show_progress_bar=self.show_progress_bar
            )
            embeddings.append(embedding)

        # Add to chunked memory store
        mem_id = self.chunked_memory_store.add_chunked(chunks, embeddings, full_text, metadata)
        self.chunked_memory_adapter.invalidate_cache()

        # Track as chunked memory
        self.chunked_memory_ids.add(mem_id)

        # Add to category if enabled (using combined embedding)
        if self.category_manager:
            # Use average embedding for categorization
            combined_embedding = np.mean(embeddings, axis=0)
            self.category_manager.add_to_category(mem_id, combined_embedding)

        logger.debug(f"Added conversation memory {mem_id} with {len(chunks)} chunks")
        return mem_id

    def chat(self, user_message: str, max_new_tokens: int = 512) -> str:
        """
        Process user message and generate a response using memory retrieval.

        This method is enhanced to handle chunking of both the query and
        the retrievable memories.

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

        # Step 2: Determine if query needs chunking
        should_chunk_query = len(user_message) > self.auto_chunk_threshold

        # Step 3: Handle query embedding differently based on size
        if should_chunk_query:
            # Create query chunks
            query_chunks = self.text_chunker.create_chunks(user_message)

            # Create embeddings for each chunk
            query_embeddings = []
            for chunk in query_chunks:
                chunk_text = chunk["text"]
                embedding = self.embedding_model.encode(
                    chunk_text, show_progress_bar=self.show_progress_bar
                )
                query_embeddings.append(embedding)

            # Use the first chunk as primary embedding, but store all for reference
            query_embedding = query_embeddings[0]
            adapted_params["query_chunks"] = query_chunks
            adapted_params["query_embeddings"] = query_embeddings

            logger.debug(f"Chunked query into {len(query_chunks)} chunks")
        else:
            # Use standard embedding for small queries
            query_embedding = self._compute_embedding(user_message)
            if query_embedding is None:
                return "Sorry, an error occurred while processing your request."

        # Step 4: Retrieve memories
        # The chunked strategy handles both chunked and non-chunked queries
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

        # Step 8: Store conversation as chunked memory
        self._store_chunked_interaction(user_message, assistant_reply, time.time())

        # Step 9: Update history and statistics
        self._update_conversation_history(user_message, assistant_reply)
        self._update_retrieval_stats(start_time, len(relevant_memories))

        return assistant_reply

    def _store_chunked_interaction(self, user_message: str, assistant_reply: str, timestamp: float):
        """
        Store conversation messages as chunked memories when appropriate.

        Args:
            user_message: User's message
            assistant_reply: Assistant's reply
            timestamp: Timestamp when the interaction occurred
        """
        try:
            # Determine if we should chunk these messages
            should_chunk_user = len(user_message) > self.auto_chunk_threshold
            should_chunk_assistant = len(assistant_reply) > self.auto_chunk_threshold

            # User message handling
            user_metadata = {
                "type": "user_message",
                "created_at": timestamp,
                "conversation_id": id(self.conversation_history),
                "importance": 0.7,
            }

            if should_chunk_user:
                # Create user message chunks
                user_chunks = self.text_chunker.create_chunks(user_message, user_metadata)

                # Create embeddings for each chunk
                user_embeddings = []
                for chunk in user_chunks:
                    chunk_text = chunk["text"]
                    embedding = self.embedding_model.encode(
                        chunk_text, show_progress_bar=self.show_progress_bar
                    )
                    user_embeddings.append(embedding)

                # Add as chunked memory
                user_mem_id = self.chunked_memory_store.add_chunked(
                    user_chunks, user_embeddings, user_message, user_metadata
                )
                self.chunked_memory_ids.add(user_mem_id)
            else:
                # Add as regular memory
                user_emb = self.embedding_model.encode(
                    user_message, show_progress_bar=self.show_progress_bar
                )
                user_mem_id = self.chunked_memory_store.add(user_emb, user_message, user_metadata)

            # Assistant message handling
            assistant_metadata = {
                "type": "assistant_message",
                "created_at": timestamp,
                "conversation_id": id(self.conversation_history),
                "importance": 0.5,
            }

            if should_chunk_assistant:
                # Create assistant message chunks
                assistant_chunks = self.text_chunker.create_chunks(
                    assistant_reply, assistant_metadata
                )

                # Create embeddings for each chunk
                assistant_embeddings = []
                for chunk in assistant_chunks:
                    chunk_text = chunk["text"]
                    embedding = self.embedding_model.encode(
                        chunk_text, show_progress_bar=self.show_progress_bar
                    )
                    assistant_embeddings.append(embedding)

                # Add as chunked memory
                assistant_mem_id = self.chunked_memory_store.add_chunked(
                    assistant_chunks, assistant_embeddings, assistant_reply, assistant_metadata
                )
                self.chunked_memory_ids.add(assistant_mem_id)
            else:
                # Add as regular memory
                assistant_emb = self.embedding_model.encode(
                    assistant_reply, show_progress_bar=self.show_progress_bar
                )
                assistant_mem_id = self.chunked_memory_store.add(
                    assistant_emb, assistant_reply, assistant_metadata
                )

            # Create associative link between messages
            if self.associative_linker:
                self.associative_linker.create_associative_link(user_mem_id, assistant_mem_id, 0.9)

            # Invalidate cache
            self.chunked_memory_adapter.invalidate_cache()

        except Exception as e:
            logger.error(f"Error storing chunked conversation in memory: {e}")

    def get_memory_chunks(self, memory_id: str) -> list[dict[str, Any]]:
        """
        Get chunks for a specific memory.

        Args:
            memory_id: ID of the memory

        Returns:
            list of chunks with their text and metadata
        """
        try:
            chunks = self.chunked_memory_store.get_chunks(memory_id)
            result = []

            for chunk in chunks:
                result.append({
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "chunk_index": chunk.chunk_index,
                })

            return result
        except Exception as e:
            logger.error(f"Error getting memory chunks: {e}")
            return []

    def get_chunking_statistics(self) -> dict[str, Any]:
        """
        Get statistics about chunking.

        Returns:
            dictionary with chunking statistics
        """
        try:
            stats = {
                "total_memories": len(self.chunked_memory_store.get_all()),
                "chunked_memories": len(self.chunked_memory_ids),
                "total_chunks": self.chunked_memory_store.get_chunk_count(),
                "avg_chunks_per_memory": self.chunked_memory_store.get_average_chunks_per_memory(),
                "auto_chunk_threshold": self.auto_chunk_threshold,
                "enable_auto_chunking": self.enable_auto_chunking,
            }

            return stats
        except Exception as e:
            logger.error(f"Error getting chunking statistics: {e}")
            return {}

    def configure_chunking(self, **kwargs) -> None:
        """
        Configure chunking parameters.

        Args:
            **kwargs: Chunking parameters to configure
        """
        # API-level parameters
        if "auto_chunk_threshold" in kwargs:
            self.auto_chunk_threshold = kwargs["auto_chunk_threshold"]
        if "enable_auto_chunking" in kwargs:
            self.enable_auto_chunking = kwargs["enable_auto_chunking"]
        if "max_chunk_count" in kwargs:
            self.max_chunk_count = kwargs["max_chunk_count"]

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

        # ChunkedFabricStrategy parameters
        strategy_params = {}
        if "chunk_weight_decay" in kwargs:
            strategy_params["chunk_weight_decay"] = kwargs["chunk_weight_decay"]
        if "max_chunks_per_memory" in kwargs:
            strategy_params["max_chunks_per_memory"] = kwargs["max_chunks_per_memory"]
        if "combine_chunk_scores" in kwargs:
            strategy_params["combine_chunk_scores"] = kwargs["combine_chunk_scores"]
        if "prioritize_coherent_chunks" in kwargs:
            strategy_params["prioritize_coherent_chunks"] = kwargs["prioritize_coherent_chunks"]

        if strategy_params:
            current_params = {
                "confidence_threshold": self.strategy.confidence_threshold,
                "similarity_weight": self.strategy.similarity_weight,
                "associative_weight": self.strategy.associative_weight,
                "temporal_weight": self.strategy.temporal_weight,
                "activation_weight": self.strategy.activation_weight,
                "max_associative_hops": self.strategy.max_associative_hops,
                "debug": self.debug,
            }
            # Update with new parameters
            current_params.update(strategy_params)
            # Re-initialize strategy
            self.strategy.initialize(current_params)
