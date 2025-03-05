import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from rich.logging import RichHandler

from memoryweave.api.llm_provider import LLMProvider
from memoryweave.api.memory_store import MemoryStoreAdapter, get_device
from memoryweave.api.prompt_builder import PromptBuilder
from memoryweave.api.retrieval_orchestrator import RetrievalOrchestrator
from memoryweave.api.streaming import StreamingHandler
from memoryweave.components.activation import ActivationManager
from memoryweave.components.associative_linking import AssociativeMemoryLinker
from memoryweave.components.category_manager import CategoryManager
from memoryweave.components.personal_attributes import PersonalAttributeManager
from memoryweave.components.post_processors import SemanticCoherenceProcessor
from memoryweave.components.query_adapter import QueryTypeAdapter
from memoryweave.components.retrieval_strategies.contextual_fabric_strategy import (
    ContextualFabricStrategy,
)
from memoryweave.components.retriever import _get_embedder
from memoryweave.components.temporal_context import TemporalContextBuilder
from memoryweave.interfaces.retrieval import QueryType
from memoryweave.query.analyzer import SimpleQueryAnalyzer
from memoryweave.storage.memory_store import MemoryStore

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(markup=True)],
)
logger = logging.getLogger(__name__)
logging.getLogger("memoryweave.components.post_processors").setLevel(logging.WARNING)

DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class MemoryWeaveAPI:
    """
    A unified API for MemoryWeave, managing memory and retrieval in a simple interface.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
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
        """Initialize MemoryWeave with an LLM, embeddings, and memory components."""
        self.debug = debug
        self.device = get_device(device)
        self.show_progress_bar = show_progress_bar

        # Configure logging
        if debug:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)

        # Initialize LLM provider
        self.llm_provider = llm_provider or LLMProvider(model_name, self.device, **model_kwargs)

        # Initialize streaming handler
        self.streaming_handler = StreamingHandler(self.llm_provider)

        # Initialize prompt builder
        self.prompt_builder = PromptBuilder()

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = _get_embedder(model_name=embedding_model_name, device=self.device)

        # Initialize memory components
        self.memory_store = MemoryStore()
        self.memory_store_adapter = MemoryStoreAdapter(self.memory_store)

        # Initialize associative linker
        self.associative_linker = AssociativeMemoryLinker(self.memory_store)

        # Initialize temporal context builder
        self.temporal_context = TemporalContextBuilder(self.memory_store)

        # Initialize activation manager
        self.activation_manager = ActivationManager()

        # Initialize query components
        self.query_analyzer = SimpleQueryAnalyzer()
        self.query_analyzer.initialize({"min_keyword_length": 3, "max_keywords": 10})

        self.query_adapter = QueryTypeAdapter()
        self.query_adapter.initialize({"apply_keyword_boost": True, "scale_params_by_length": True})

        # Initialize optional components
        self.category_manager = None
        if enable_category_management:
            self.category_manager = CategoryManager()
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            self.category_manager.initialize(
                config=dict(
                    vigilance_threshold=0.85,
                    embedding_dim=embedding_dim,
                ),
            )

        # Initialize personal attribute manager
        self.personal_attribute_manager = None
        if enable_personal_attributes:
            self.personal_attribute_manager = PersonalAttributeManager()
            self.personal_attribute_manager.initialize()

        # Initialize semantic coherence processor
        self.semantic_coherence_processor = None
        if enable_semantic_coherence:
            self.semantic_coherence_processor = SemanticCoherenceProcessor()
            self.semantic_coherence_processor.initialize()

        # Initialize retrieval strategy
        self.strategy = ContextualFabricStrategy(
            memory_store=self.memory_store_adapter,
            associative_linker=self.associative_linker,
            temporal_context=self.temporal_context,
            activation_manager=self.activation_manager,
        )
        self.strategy.initialize({
            "confidence_threshold": 0.1,
            "similarity_weight": 0.4,
            "associative_weight": 0.3,
            "temporal_weight": 0.2,
            "activation_weight": 0.1,
            "max_associative_hops": 2,
            "debug": debug,
        })

        # Initialize retrieval orchestrator
        self.retrieval_orchestrator = RetrievalOrchestrator(
            strategy=self.strategy,
            activation_manager=self.activation_manager,
            temporal_context=self.temporal_context,
            semantic_coherence_processor=self.semantic_coherence_processor,
            memory_store_adapter=self.memory_store_adapter,
            debug=debug,
        )

        # Memory management settings
        self.max_memories = max_memories
        self.consolidation_interval = consolidation_interval
        self.memories_since_consolidation = 0

        # Conversation tracking
        self.conversation_history = []
        self.retrieval_stats = {
            "total_queries": 0,
            "successful_retrievals": 0,
            "avg_query_time": 0,
            "avg_results_count": 0,
        }

    def add_memory(self, text: str, metadata: dict[str, Any] = None) -> str:
        """Store a memory with consistent handling of metadata."""
        logger.debug(f"Adding memory: {text}")

        # Create embedding
        embedding = self.embedding_model.encode(text, show_progress_bar=self.show_progress_bar)

        # Add default metadata if not provided
        if metadata is None:
            metadata = {"type": "manual", "created_at": time.time(), "importance": 0.6}
        elif "created_at" not in metadata:
            metadata["created_at"] = time.time()

        # Add to memory store
        mem_id = self.memory_store.add(embedding, text, metadata)
        self.memory_store_adapter.invalidate_cache()

        # Add to category if enabled
        if self.category_manager:
            self.category_manager.add_to_category(mem_id, embedding)

        # Track memories since consolidation
        self.memories_since_consolidation += 1

        # Perform consolidation if needed
        if self.memories_since_consolidation >= self.consolidation_interval:
            self._consolidate_memories()

        logger.debug(f"Added memory {mem_id}: {text}")
        return mem_id

    def add_memories(
        self, texts: list[str], metadata_list: list[dict[str, Any]] | None = None
    ) -> list[str]:
        """Add multiple memories efficiently."""
        if not texts:
            return []

        # Create metadata list if not provided
        if metadata_list is None:
            metadata_list = [None] * len(texts)
        elif len(metadata_list) != len(texts):
            raise ValueError("metadata_list must have the same length as texts")

        # Add each memory
        memory_ids = []
        for text, metadata in zip(texts, metadata_list):
            memory_id = self.add_memory(text, metadata)
            memory_ids.append(memory_id)

        return memory_ids

    def chat(self, user_message: str, max_new_tokens: int = 512) -> str:
        """Process user message and generate a response using memory retrieval."""
        start_time = time.time()

        # Step 1: Query analysis
        query_info = self._analyze_query(user_message)
        _query_obj: dict[str, Any]
        adapted_params: dict[str, Any]
        expanded_keywords: list[str]

        _query_obj, adapted_params, expanded_keywords, query_type, entities = query_info

        # Step 2: Compute query embedding

        if (query_embedding := self._compute_embedding(user_message)) is None:
            return "Sorry, an error occurred while processing your request."

        # Step 3: Retrieve memories
        relevant_memories = self.retrieval_orchestrator.retrieve(
            query_embedding=query_embedding,
            query=user_message,
            query_type=query_type,
            expanded_keywords=expanded_keywords,
            entities=entities,
            adapted_params=adapted_params,
            top_k=adapted_params.get("max_results", 10),
        )

        # Step 4: Extract personal attributes if enabled
        self._extract_personal_attributes(user_message, time.time())

        # Step 5: Construct prompt
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
        # Step 6: Generate response
        assistant_reply = self.llm_provider.generate(prompt=prompt, max_new_tokens=max_new_tokens)

        # Step 7: Update history and statistics
        self._store_interaction(user_message, assistant_reply, time.time())
        self._update_conversation_history(user_message, assistant_reply)
        self._update_retrieval_stats(start_time, len(relevant_memories))

        return assistant_reply

    async def chat_stream(
        self, user_message: str, max_new_tokens: int = 512
    ) -> AsyncGenerator[str, None]:
        """Process user message and stream the response."""
        start_time = time.time()

        # Step 1: Query analysis
        query_info = self._analyze_query(user_message)
        query_obj, adapted_params, expanded_keywords, query_type, entities = query_info

        # Step 2: Compute query embedding
        query_embedding = self._compute_embedding(user_message)
        if query_embedding is None:
            yield "Sorry, an error occurred while processing your request."
            return

        # Step 3: Retrieve memories
        now = time.time()
        relevant_memories = self.retrieval_orchestrator.retrieve(
            query_embedding=query_embedding,
            query=user_message,
            query_type=query_type,
            expanded_keywords=expanded_keywords,
            entities=entities,
            adapted_params=adapted_params,
            top_k=adapted_params.get("max_results", 10),
        )

        # Step 4: Extract personal attributes if enabled
        self._extract_personal_attributes(user_message, now)

        # Step 5: Construct prompt
        prompt = self.prompt_builder.build_chat_prompt(
            user_message=user_message,
            memories=relevant_memories,
            conversation_history=self.conversation_history,
            query_type=query_type,
        )

        # Step 6: Stream response
        full_response = []
        async for token in self.streaming_handler.stream(prompt, max_new_tokens):
            full_response.append(token)
            yield token

        # Step 7: Update history and statistics
        assistant_reply = "".join(full_response)
        self._store_interaction(user_message, assistant_reply, now)
        self._update_conversation_history(user_message, assistant_reply)
        self._update_retrieval_stats(start_time, len(relevant_memories))

    def retrieve(
        self,
        query: str,
        query_embedding=None,
        top_k: int = 10,
        confidence_threshold: float = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Direct access to the retrieval system."""
        # Compute embedding if not provided
        if query_embedding is None:
            query_embedding = self._compute_embedding(query)
            if query_embedding is None:
                return []

        # Analyze query to get type and keywords
        query_info = self._analyze_query(query)
        _, adapted_params, expanded_keywords, query_type, entities = query_info

        # Override with any provided parameters
        if confidence_threshold is not None:
            if adapted_params is None:
                adapted_params = {}
            adapted_params["confidence_threshold"] = confidence_threshold

        # Add any custom parameters from kwargs
        for key, value in kwargs.items():
            if adapted_params is None:
                adapted_params = {}
            adapted_params[key] = value

        # Use retrieval orchestrator for consistent handling
        memories = self.retrieval_orchestrator.retrieve(
            query_embedding=query_embedding,
            query=query,
            query_type=query_type,
            expanded_keywords=expanded_keywords,
            entities=entities,
            adapted_params=adapted_params,
            top_k=top_k,
        )

        return memories

    # Helper methods
    def _analyze_query(self, user_message: str):
        """Analyze query to extract type, keywords, and parameters."""
        try:
            query_type = self.query_analyzer.analyze(user_message)
            keywords = self.query_analyzer.extract_keywords(user_message)
            entities = self.query_analyzer.extract_entities(user_message)

            query_obj = {
                "text": user_message,
                "query_type": query_type,
                "extracted_keywords": keywords,
                "extracted_entities": entities,
            }

            adapted_params = self.query_adapter.adapt_parameters(query_obj)

            # Expand keywords if possible
            if hasattr(self, "keyword_expander") and self.keyword_expander:
                expanded_obj = self.keyword_expander.expand(query_obj)
                expanded_keywords = expanded_obj.get("extracted_keywords", keywords)
            else:
                expanded_keywords = keywords

            logger.debug(f"Query type: {query_type}")
            logger.debug(f"Keywords: {keywords}")
            logger.debug(f"Expanded keywords: {expanded_keywords}")
            logger.debug(f"Entities: {entities}")

        except Exception as e:
            logger.error(f"Error during query analysis: {e}")
            # Fallback defaults
            query_type = QueryType.UNKNOWN
            keywords = []
            entities = []
            expanded_keywords = []
            adapted_params = {"confidence_threshold": 0.1, "max_results": 10}
            query_obj = {"text": user_message}

        return query_obj, adapted_params, expanded_keywords, query_type, entities

    def _compute_embedding(self, text: str):
        """Compute embedding for text."""
        try:
            return self.embedding_model.encode(text, show_progress_bar=self.show_progress_bar)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _extract_personal_attributes(self, user_message: str, timestamp: float):
        """Extract and store personal attributes."""
        if self.personal_attribute_manager:
            try:
                attributes = self.personal_attribute_manager.extract_attributes(user_message)
                if attributes:
                    logger.debug(f"Extracted personal attributes: {attributes}")
                    for attr_type, attr_value in attributes.items():
                        attr_text = f"The user's {attr_type} is {attr_value}."
                        if len(str(attr_value)) > 1:
                            self.add_memory(
                                attr_text,
                                {
                                    "type": "personal_attribute",
                                    "attribute_type": attr_type,
                                    "attribute_value": attr_value,
                                    "created_at": timestamp,
                                    "importance": 0.9,
                                },
                            )
            except Exception as e:
                logger.error(f"Error extracting personal attributes: {e}")

    def _store_interaction(self, user_message: str, assistant_reply: str, timestamp: float):
        """Store conversation messages as memories."""
        try:
            # Create embeddings
            _user_emb = self.embedding_model.encode(
                user_message, show_progress_bar=self.show_progress_bar
            )
            _assistant_emb = self.embedding_model.encode(
                assistant_reply, show_progress_bar=self.show_progress_bar
            )

            # Store user message
            user_mem_id = self.add_memory(
                user_message,
                {
                    "type": "user_message",
                    "created_at": timestamp,
                    "conversation_id": id(self.conversation_history),
                    "importance": 0.7,
                },
            )

            # Store assistant message
            assistant_mem_id = self.add_memory(
                assistant_reply,
                {
                    "type": "assistant_message",
                    "created_at": timestamp,
                    "conversation_id": id(self.conversation_history),
                    "importance": 0.5,
                },
            )

            # Create associative link between messages
            if self.associative_linker:
                self.associative_linker.create_associative_link(user_mem_id, assistant_mem_id, 0.9)

        except Exception as e:
            logger.error(f"Error storing conversation in memory: {e}")

    def _update_conversation_history(self, user_message: str, assistant_reply: str):
        """Add messages to conversation history."""
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_reply})

    def _update_retrieval_stats(self, start_time: float, results_count: int):
        """Update retrieval performance statistics."""
        query_time = time.time() - start_time
        stats = self.retrieval_stats
        stats["total_queries"] += 1
        stats["successful_retrievals"] += 1 if results_count > 0 else 0
        n = stats["total_queries"]
        stats["avg_query_time"] = ((n - 1) * stats["avg_query_time"] + query_time) / n
        stats["avg_results_count"] = ((n - 1) * stats["avg_results_count"] + results_count) / n

    def _consolidate_memories(self):
        """Consolidate memories to stay within capacity limits."""
        # Skip if no consolidation needed
        if len(self.memory_store.get_all()) <= self.max_memories:
            self.memories_since_consolidation = 0
            logger.debug("Memory consolidation skipped: no consolidation needed.")
            return 0

        logger.info(f"Consolidating memories to stay within limit of {self.max_memories}")

        # Consolidate category manager if available
        if self.category_manager:
            try:
                # Merge similar categories
                merged_categories = self.category_manager.consolidate_categories(
                    similarity_threshold=0.7
                )
                logger.info(f"Merged {len(merged_categories)} similar categories")
            except Exception as e:
                logger.error(f"Error consolidating categories: {e}")

        # Consolidate memory store
        removed_ids = self.memory_store.consolidate(self.max_memories)
        self.memory_store_adapter.invalidate_cache()

        # Reset counter
        self.memories_since_consolidation = 0

        logger.info(f"Consolidated {len(removed_ids)} memories.")
        return len(removed_ids)

    def get_conversation_history(self):
        """Return the conversation history."""
        return self.conversation_history

    def clear_memories(self, keep_personal_attributes: bool = True):
        """
        Clear all memories except personal attributes if specified.

        Args:
            keep_personal_attributes: Whether to keep personal attribute memories

        Returns:
            Number of memories removed
        """
        logger.info(f"Clearing memories. Keeping personal attributes: {keep_personal_attributes}")
        if keep_personal_attributes:
            # Get all memories
            memories = self.memory_store.get_all()

            # Keep personal attribute memories
            kept_ids = []
            for memory in memories:
                if (
                    memory.metadata.get("type") == "personal_attribute"
                    or "attribute_type" in memory.metadata
                ):
                    kept_ids.append(memory.id)

            # Count memories to be removed
            count_before = len(memories)
            count_kept = len(kept_ids)

            # Clear all memories
            self.memory_store.clear()
            self.memory_store_adapter.invalidate_cache()

            # Re-add the kept memories
            for memory_id in kept_ids:
                try:
                    memory = self.memory_store.get(memory_id)
                    self.memory_store.add(memory.embedding, memory.content, memory.metadata)
                except KeyError:
                    pass

            removed_count = count_before - count_kept
            logger.info(
                f"Removed {removed_count} memories, kept {count_kept} personal attribute memories."
            )
            return removed_count
        else:
            # Count memories before clearing
            count_before = len(self.memory_store.get_all())

            # Clear all memories
            self.memory_store.clear()
            self.memory_store_adapter.invalidate_cache()

            logger.info(f"Removed {count_before} memories.")
            return count_before

    def get_retrieval_stats(self):
        """Get statistics about memory retrieval."""
        return self.retrieval_stats.copy()

    def search_by_keyword(self, keyword: str, limit: int = 10):
        """
        Search for memories containing a specific keyword.

        Args:
            keyword: Keyword to search for
            limit: Maximum number of results to return

        Returns:
            List of matching memories with scores
        """
        logger.debug(f"Searching for memories with keyword: {keyword}")
        results = []

        # Get all memories
        memories = self.memory_store.get_all()

        # Filter memories containing the keyword
        for memory in memories:
            # Extract content text
            if isinstance(memory.content, dict) and "text" in memory.content:
                content = memory.content["text"]
            else:
                content = str(memory.content)

            # Check if keyword is in content
            if keyword.lower() in content.lower():
                # Calculate relevance based on keyword prominence
                occurrences = content.lower().count(keyword.lower())
                relevance = min(1.0, 0.5 + (occurrences * 0.1))

                results.append({
                    "memory_id": memory.id,
                    "content": content,
                    "metadata": memory.metadata,
                    "relevance_score": relevance,
                    "keyword_occurrences": occurrences,
                })

        # Sort by relevance score
        results.sort(key=lambda x: x["relevance_score"], reverse=True)

        logger.debug(f"Found {len(results)} memories matching keyword: {keyword}")
        return results[:limit]

    def get_similar_memories(self, memory_id: str, limit: int = 5):
        """
        Find memories similar to a given memory.

        Args:
            memory_id: ID of the reference memory
            limit: Maximum number of similar memories to return

        Returns:
            List of similar memories with similarity scores
        """
        logger.debug(f"Finding similar memories for memory ID: {memory_id}")
        try:
            # Get the memory
            memory = self.memory_store.get(memory_id)

            # Use its embedding to find similar memories
            query_embedding = memory.embedding

            # Use the retrieval orchestrator
            similar_memories = self.retrieval_orchestrator.retrieve(
                query_embedding=query_embedding,
                query="",  # No query text for direct embedding search
                top_k=limit + 1,  # +1 because we'll filter out the original memory
            )

            # Filter out the original memory
            filtered_memories = [m for m in similar_memories if m.get("memory_id") != memory_id]
            logger.debug(f"Found {len(filtered_memories)} similar memories")
            return filtered_memories

        except KeyError:
            logger.warning(f"Memory with ID {memory_id} not found")
            return []
        except Exception as e:
            logger.error(f"Error finding similar memories: {e}")
            return []

    def update_memory(self, memory_id: str, new_text: str = None, new_metadata: dict = None):
        """
        Update an existing memory with new text or metadata.

        Args:
            memory_id: ID of the memory to update
            new_text: New text content (or None to keep existing)
            new_metadata: New metadata (or None to keep existing)

        Returns:
            True if update successful, False otherwise
        """
        logger.info(f"Updating memory ID: {memory_id}")
        try:
            # Get the existing memory
            memory = self.memory_store.get(memory_id)

            # If new text is provided, update content and embedding
            if new_text is not None:
                # Generate new embedding
                new_embedding = self.embedding_model.encode(
                    new_text, show_progress_bar=self.show_progress_bar
                )

                # Create new content
                if isinstance(memory.content, dict) and "text" in memory.content:
                    new_content = memory.content.copy()
                    new_content["text"] = new_text
                else:
                    new_content = new_text

                # Remove the old memory
                self.memory_store.remove(memory_id)
                self.memory_store_adapter.invalidate_cache()

                # Create updated metadata
                updated_metadata = memory.metadata.copy() if memory.metadata else {}
                if new_metadata:
                    updated_metadata.update(new_metadata)

                # Add as a new memory with the original ID if possible
                try:
                    # Try to use the same ID
                    self.memory_store.add_with_id(
                        memory_id, new_embedding, new_content, updated_metadata
                    )
                except AttributeError:
                    # If add_with_id is not supported, just add as a new memory
                    return self.memory_store.add(new_embedding, new_content, updated_metadata)

                # Update category if enabled
                if self.category_manager:
                    self.category_manager.add_to_category(memory_id, new_embedding)

                logger.info(f"Memory ID: {memory_id} updated with new text.")
                return True

            # If only updating metadata
            elif new_metadata is not None:
                self.memory_store.update_metadata(memory_id, new_metadata)
                self.memory_store_adapter.invalidate_cache()
                logger.info(f"Memory ID: {memory_id} metadata updated.")
                return True

            logger.warning(f"No update performed for memory ID: {memory_id}")
            return False

        except KeyError:
            logger.warning(f"Memory with ID {memory_id} not found")
            return False
        except Exception as e:
            logger.error(f"Error updating memory: {e}")
            return False
