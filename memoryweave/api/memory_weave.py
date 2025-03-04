import logging
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memoryweave.api.memory_store import MemoryStoreAdapter
from memoryweave.components.activation import ActivationManager
from memoryweave.components.associative_linking import AssociativeMemoryLinker
from memoryweave.components.category_manager import CategoryManager
from memoryweave.components.dynamic_threshold_adjuster import DynamicThresholdAdjuster
from memoryweave.components.keyword_expander import KeywordExpander
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

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_TOKENIZER: AutoTokenizer | None = None
_LLM: AutoModelForCausalLM | None = None
_DEVICE: str | None = None


def _get_device(device: str | None = None) -> str:
    """Choose a device for running the model."""
    if torch.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def get_tokenizer(model_name: str = DEFAULT_MODEL, **kwargs) -> AutoTokenizer:
    """Singleton to load a tokenizer."""
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name, **kwargs)
    return _TOKENIZER


def get_llm(model_name: str = DEFAULT_MODEL, device: str = "mps", **kwargs) -> AutoModelForCausalLM:
    """Singleton to load a Hugging Face causal LM."""
    global _LLM, _DEVICE
    if _LLM is None:
        _DEVICE = _get_device(device)
        torch_dtype = torch.float16 if _DEVICE == "cuda" else torch.float32

        print(f"Loading LLM: {model_name}")
        _LLM = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=_DEVICE,
            **kwargs,
        )
    return _LLM


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
        consolidation_interval: int = 100,  # Memories added before consolidation
        show_progress_bar: bool = False,
        debug: bool = False,
        **model_kwargs,
    ):
        """Initialize MemoryWeave with an LLM, embeddings, and memory components."""
        self.device = self._get_device(device)
        self.show_progress_bar = show_progress_bar
        self.debug = debug
        self.max_memories = max_memories
        self.memories_since_consolidation = 0
        self.consolidation_interval = consolidation_interval

        # Configure logging
        if debug:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)

        # Load LLM & tokenizer
        self.tokenizer = get_tokenizer(model_name)
        if "show_progress_bar" in model_kwargs:
            model_kwargs.pop("show_progress_bar")
        self.model = get_llm(model_name, device=self.device, **model_kwargs)

        # Embedding model for memory
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = _get_embedder(model_name=embedding_model_name, device=self.device)

        # Initialize Memory Store & Components
        self.memory_store = MemoryStore()
        self.memory_store_adapter = MemoryStoreAdapter(self.memory_store)
        self.associative_linker = AssociativeMemoryLinker(self.memory_store)
        self.temporal_context = TemporalContextBuilder(self.memory_store)
        self.activation_manager = ActivationManager(self.memory_store, self.associative_linker)

        # Initialize query processing components
        self.query_analyzer = SimpleQueryAnalyzer()
        self.query_analyzer.initialize({"min_keyword_length": 3, "max_keywords": 10})

        self.query_adapter = QueryTypeAdapter()
        self.query_adapter.initialize({"apply_keyword_boost": True, "scale_params_by_length": True})

        self.keyword_expander = KeywordExpander()
        self.keyword_expander.initialize({"expansion_count": 3, "min_similarity": 0.7})

        # Initialize optional components
        self.category_manager = None
        if enable_category_management:
            self.category_manager = CategoryManager()
            # Get the actual embedding dimension from the model
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            self.category_manager.initialize(
                config=dict(
                    vigilance_threshold=0.85,
                    embedding_dim=embedding_dim,  # Use actual dimension from embedding model
                ),
            )
        self.personal_attribute_manager = None
        if enable_personal_attributes:
            self.personal_attribute_manager = PersonalAttributeManager()  # noqa: F821
            self.personal_attribute_manager.initialize()

        self.semantic_coherence_processor = None
        if enable_semantic_coherence:
            self.semantic_coherence_processor = SemanticCoherenceProcessor()
            self.semantic_coherence_processor.initialize()

        self.dynamic_threshold_adjuster = None
        if enable_dynamic_thresholds:
            self.dynamic_threshold_adjuster = DynamicThresholdAdjuster()
            self.dynamic_threshold_adjuster.initialize(
                config=dict(
                    min_threshold=0.1,
                    max_threshold=0.8,
                    learning_rate=0.05,
                ),
            )

        # Retrieval strategy
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

        # Track conversation history and statistics
        self.conversation_history = []
        self.retrieval_stats = {
            "total_queries": 0,
            "successful_retrievals": 0,
            "avg_query_time": 0,
            "avg_results_count": 0,
        }

    def _get_device(self, device: str) -> str:
        """Determine the device for running the model."""
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.mps.is_available():
            return "mps"
        return "cpu"

    def add_memory(self, text: str, metadata: dict[str, Any] = None) -> str:
        """
        Store a memory in the system with optional metadata.

        Args:
            text: The text content to store
            metadata: Optional metadata for the memory

        Returns:
            The ID of the newly created memory
        """
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

        # Add to category if category management is enabled
        if self.category_manager:
            self.category_manager.add_to_category(mem_id, embedding)

        # Track memories added since last consolidation
        self.memories_since_consolidation += 1

        # Perform consolidation if needed
        if self.memories_since_consolidation >= self.consolidation_interval:
            self._consolidate_memories()

        logger.debug(f"Added memory {mem_id}: {text}")
        return mem_id

    def add_memories(
        self, texts: list[str], metadata_list: list[dict[str, Any]] | None = None
    ) -> list[str]:
        """
        Add multiple memories efficiently.

        Args:
            texts: list of text contents to store
            metadata_list: Optional list of metadata dicts (one per text)

        Returns:
            list of memory IDs
        """
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
        """
        Process user input with advanced memory retrieval and generate a response.

        Args:
            user_message: The user's message
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            The assistant's response
        """
        # Record start time for performance tracking
        start_time = time.time()
        now = start_time

        # ✅ Step 1: Analyze query
        try:
            # Determine query type
            query_type = self.query_analyzer.analyze(user_message)

            # Extract keywords and entities
            keywords = self.query_analyzer.extract_keywords(user_message)
            entities = self.query_analyzer.extract_entities(user_message)

            # Create query object for adaptation
            query_obj = {
                "text": user_message,
                "query_type": query_type,
                "extracted_keywords": keywords,
                "extracted_entities": entities,
            }

            # Adapt parameters based on query type
            adapted_params = self.query_adapter.adapt_parameters(query_obj)

            # Expand keywords for better retrieval
            if self.keyword_expander:
                expanded_obj = self.keyword_expander.expand(query_obj)
                expanded_keywords = expanded_obj.get("extracted_keywords", keywords)
            else:
                expanded_keywords = keywords

            logger.debug(f"Query type: {query_type}")
            logger.debug(f"Keywords: {keywords}")
            logger.debug(f"Expanded keywords: {expanded_keywords}")
            logger.debug(f"Entities: {entities}")

        except Exception as analyze_err:
            logger.error(f"Error during query analysis: {analyze_err}")
            # Fall back to defaults
            query_type = QueryType.UNKNOWN
            keywords = []
            entities = []
            expanded_keywords = []
            adapted_params = {"confidence_threshold": 0.1, "max_results": 10}

        # ✅ Step 2: Compute query embedding
        try:
            query_embedding = self.embedding_model.encode(
                user_message, show_progress_bar=self.show_progress_bar
            )
        except Exception as encode_err:
            logger.error(f"Error generating query embedding: {encode_err}")
            import traceback

            traceback.print_exc()
            return "Sorry, an error occurred while processing your request."

        # ✅ Step 3: Adjust confidence threshold if dynamic adjustment is enabled
        if self.dynamic_threshold_adjuster:
            # Use historical performance to adjust threshold
            adjusted_threshold = self.dynamic_threshold_adjuster.get_adjusted_threshold(
                query_type, adapted_params.get("confidence_threshold", 0.1)
            )
            adapted_params["confidence_threshold"] = adjusted_threshold
            logger.debug(f"Adjusted confidence threshold: {adjusted_threshold}")

        # ✅ Step 4: Retrieve relevant memories with comprehensive context
        self.memory_store_adapter.invalidate_cache()
        try:
            logger.debug("Retrieving memories using strategy")
            retrieval_context = {
                "query": user_message,
                "query_type": query_type,
                "important_keywords": expanded_keywords,
                "extracted_entities": entities,
                "current_time": now,
                "memory_store": self.memory_store_adapter,
                "adapted_retrieval_params": adapted_params,
            }

            relevant_memories = self.strategy.retrieve(
                query_embedding=query_embedding,
                top_k=adapted_params.get("max_results", 10),
                context=retrieval_context,
            )
        except Exception as main_err:
            logger.error(f"Error using contextual_fabric strategy: {main_err}")
            import traceback

            traceback.print_exc()
            relevant_memories = []

        # Fallback if retrieval is empty
        if not relevant_memories:
            try:
                logger.warning("Primary retrieval returned no results, using fallback")
                fallback_results = self.memory_store_adapter.search_by_vector(
                    query_embedding, limit=10
                )
                relevant_memories = []
                for item in fallback_results:
                    relevant_memories.append({
                        "memory_id": None,  # no real ID from fallback
                        "relevance_score": item.get("score", 0.5),
                        "content": item.get("content", ""),
                        "metadata": item.get("metadata", {}),
                    })
                logger.debug(f"Fallback retrieved {len(relevant_memories)} memories")
            except Exception as fallback_err:
                logger.error(f"Even fallback retrieval failed: {fallback_err}")
                traceback.print_exc()
                relevant_memories = []

        # ✅ Step 5: Update memory activations and apply semantic coherence
        for mem_dict in relevant_memories:
            mem_id = mem_dict.get("memory_id")
            if mem_id is not None:
                # Increase activation and update timestamp
                self.activation_manager.activate_memory(mem_id, 0.2, spread=True)
                self.temporal_context.update_timestamp(mem_id, now)

        # Apply semantic coherence check if enabled
        if self.semantic_coherence_processor and len(relevant_memories) > 1:
            try:
                coherent_results = self.semantic_coherence_processor.process_results(
                    relevant_memories, query_embedding
                )
                # Only replace if we got reasonable results back
                if coherent_results:
                    relevant_memories = coherent_results
                    logger.debug(
                        f"Applied semantic coherence, now have {len(relevant_memories)} results"
                    )
            except Exception as coherence_err:
                logger.error(f"Error in semantic coherence processing: {coherence_err}")

        # ✅ Step 6: Apply temporal context to handle time-based queries
        base_context = {"current_time": now}
        memory_dicts_for_temporal = []

        for r in relevant_memories:
            mem_id = r.get("memory_id")
            base_score = r.get("relevance_score", 0.5)

            # If "created_at" is stored in metadata, retrieve it
            created_at = 0.0
            meta = r.get("metadata", {})
            if isinstance(meta, dict) and "created_at" in meta:
                created_at = meta["created_at"]

            memory_dicts_for_temporal.append({
                "memory_id": mem_id,
                "created_at": created_at,
                "relevance_score": base_score,
            })

        # Apply temporal context with the user query
        memory_dicts_for_temporal = self.temporal_context.apply_temporal_context(
            query=user_message, results=memory_dicts_for_temporal, context=base_context
        )

        # Sort by updated "relevance_score"
        memory_dicts_for_temporal.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)

        # ✅ Step 7: Process memories for prompt inclusion
        cleaned_results = []
        for md in memory_dicts_for_temporal:
            mem_id = md.get("memory_id")
            if mem_id is None:
                continue

            try:
                # Get the actual memory object
                memory_obj = None
                if isinstance(mem_id, int):
                    try:
                        memory_obj = self.memory_store.get(str(mem_id))
                    except KeyError:
                        logger.debug(f"No memory found for index {mem_id}")
                else:
                    try:
                        memory_obj = self.memory_store.get(mem_id)
                    except KeyError:
                        logger.debug(f"No memory found for ID {mem_id}")

                if memory_obj:
                    # Get content appropriately depending on type
                    if isinstance(memory_obj.content, dict) and "text" in memory_obj.content:
                        content_text = memory_obj.content["text"]
                    else:
                        content_text = str(memory_obj.content)

                    cleaned_results.append({
                        "content": content_text,
                        "metadata": memory_obj.metadata,
                        "relevance_score": md.get("relevance_score", 0.5),
                    })
            except Exception as mem_err:
                logger.error(f"Error retrieving memory {mem_id}: {mem_err}")

        # Update to use the processed results
        relevant_memories = cleaned_results
        logger.debug(f"After temporal context, have {len(relevant_memories)} memories")

        # ✅ Step 8: Extract personal attributes if enabled
        if self.personal_attribute_manager:
            try:
                attributes = self.personal_attribute_manager.extract_attributes(user_message)
                if attributes:
                    logger.debug(f"Extracted personal attributes: {attributes}")
                    # Store each attribute as a synthetic memory
                    for attr_type, attr_value in attributes.items():
                        attr_text = f"The user's {attr_type} is {attr_value}."
                        # Only add if it's substantial
                        if len(str(attr_value)) > 1:
                            self.add_memory(
                                attr_text,
                                {
                                    "type": "personal_attribute",
                                    "attribute_type": attr_type,
                                    "attribute_value": attr_value,
                                    "created_at": now,
                                    "importance": 0.9,
                                },
                            )
            except Exception as attr_err:
                logger.error(f"Error extracting personal attributes: {attr_err}")

        # ✅ Step 9: Construct system prompt with retrieved memories
        system_prompt = (
            "You are a helpful assistant. Use the entire conversation context for your answer.\n"
            "Do not disclaim that you have a memory system. If asked about user info, see if it "
            "is in your retrieved content. If yes, incorporate it naturally. If not, just say you "
            "lack that info.\n"
        )

        # Sort again by final relevance
        relevant_memories.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        # Include top memories in the prompt (more for complex queries)
        max_memories = 5 if query_type in [QueryType.PERSONAL, QueryType.TEMPORAL] else 3
        top_memories = relevant_memories[:max_memories]
        memory_text = ""

        if top_memories:
            memory_text = "MEMORY HIGHLIGHTS:\n"
            for m in top_memories:
                c = m["content"]
                # Truncate long memories
                if len(c) > 150:
                    memory_text += f"- {c[:150]}...\n"
                else:
                    memory_text += f"- {c}\n"
            memory_text += "\n"

        final_system_prompt = system_prompt + memory_text

        # ✅ Step 10: Add recent conversation history
        history_text = ""
        max_history_turns = 10  # Increase for temporal and personal queries

        recent_history = (
            self.conversation_history[-max_history_turns:] if self.conversation_history else []
        )

        for turn in recent_history:
            if turn["role"] == "user":
                history_text += f"User: {turn['content']}\n"
            else:
                history_text += f"Assistant: {turn['content']}\n"

        if history_text:
            prompt = f"{final_system_prompt}\n{history_text}User: {user_message}\nAssistant:"
        else:
            prompt = f"{final_system_prompt}\nUser: {user_message}\nAssistant:"

        # ✅ Step 11: Generate response using LLM
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_reply = full_response[len(prompt) :].strip()

        # ✅ Step 12: Store the interaction in memory
        try:
            # Create embeddings for user and assistant messages
            user_emb = self.embedding_model.encode(
                user_message, show_progress_bar=self.show_progress_bar
            )
            assistant_emb = self.embedding_model.encode(
                assistant_reply, show_progress_bar=self.show_progress_bar
            )

            # Add user message to memory store
            user_mem_id = self.memory_store.add(
                user_emb,
                user_message,
                {
                    "type": "user_message",
                    "created_at": now,
                    "conversation_id": id(self.conversation_history),
                    "importance": 0.7,
                },
            )

            # Add to category if enabled
            if self.category_manager:
                self.category_manager.add_to_category(user_mem_id, user_emb)

            # Add assistant message to memory store
            assistant_mem_id = self.memory_store.add(
                assistant_emb,
                assistant_reply,
                {
                    "type": "assistant_message",
                    "created_at": now,
                    "conversation_id": id(self.conversation_history),
                    "importance": 0.5,
                },
            )

            # Add to category if enabled
            if self.category_manager:
                self.category_manager.add_to_category(assistant_mem_id, assistant_emb)

            # Create associative link between the user message and response
            if self.associative_linker:
                self.associative_linker.create_associative_link(user_mem_id, assistant_mem_id, 0.9)

            self.memory_store_adapter.invalidate_cache()

            # Update memories since consolidation
            self.memories_since_consolidation += 2

            # Check if consolidation is needed
            if self.memories_since_consolidation >= self.consolidation_interval:
                self._consolidate_memories()

        except Exception as mem_err:
            logger.error(f"Error storing conversation in memory: {mem_err}")

        # ✅ Step 13: Update conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_reply})

        # ✅ Step 14: Update retrieval statistics
        query_time = time.time() - start_time
        self.retrieval_stats["total_queries"] += 1
        self.retrieval_stats["successful_retrievals"] += 1 if relevant_memories else 0

        # Update moving averages
        n = self.retrieval_stats["total_queries"]
        self.retrieval_stats["avg_query_time"] = (
            (n - 1) * self.retrieval_stats["avg_query_time"] + query_time
        ) / n
        self.retrieval_stats["avg_results_count"] = (
            (n - 1) * self.retrieval_stats["avg_results_count"] + len(relevant_memories)
        ) / n

        # ✅ Step 15: Update dynamic threshold if enabled
        if self.dynamic_threshold_adjuster:
            # Provide feedback on retrieval quality to improve future thresholds
            self.dynamic_threshold_adjuster.update_threshold(
                query_type, len(relevant_memories), had_good_results=len(relevant_memories) > 0
            )

        return assistant_reply

    def get_conversation_history(self) -> list[dict[str, str]]:
        """Return stored conversation history."""
        return self.conversation_history

    def retrieve(self, query: str, **kwargs) -> list[dict[str, Any]]:
        """
        Retrieve memories using the contextual fabric strategy.

        Args:
            query: The query string
            **kwargs: Additional parameters for retrieval

        Returns:
            list of retrieved memories with metadata and scores
        """
        # Create embedding
        query_embedding = self.embedding_model.encode(
            query, show_progress_bar=self.show_progress_bar
        )

        # Set up retrieval context
        context = {
            "query": query,
            "current_time": time.time(),
            "memory_store": self.memory_store_adapter,
        }

        # Add any additional context parameters
        context.update(kwargs)

        # Retrieve memories
        return self.strategy.retrieve(
            query_embedding=query_embedding, top_k=kwargs.get("top_k", 10), context=context
        )

    def search_by_keyword(self, keyword: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        Search for memories containing a specific keyword.

        Args:
            keyword: The keyword to search for
            limit: Maximum number of results to return

        Returns:
            list of matching memories with scores
        """
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
                # (Simple heuristic: more occurrences = higher relevance)
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

        return results[:limit]

    def get_similar_memories(self, memory_id: str, limit: int = 5) -> list[dict[str, Any]]:
        """
        Find memories similar to a given memory.

        Args:
            memory_id: ID of the memory to find similar ones for
            limit: Maximum number of similar memories to return

        Returns:
            list of similar memories with similarity scores
        """
        try:
            # Get the memory
            memory = self.memory_store.get(memory_id)

            # Use its embedding to find similar memories
            query_embedding = memory.embedding

            # Use the strategy to retrieve similar memories
            similar_memories = self.strategy.retrieve(
                query_embedding=query_embedding,
                top_k=limit + 1,  # +1 because we'll filter out the original memory
                context={"memory_store": self.memory_store_adapter, "current_time": time.time()},
            )

            # Filter out the original memory
            return [m for m in similar_memories if m.get("memory_id") != memory_id]

        except KeyError:
            logger.warning(f"Memory with ID {memory_id} not found")
            return []
        except Exception as e:
            logger.error(f"Error finding similar memories: {e}")
            return []

    def get_memory_categories(self) -> dict[str, list[str]]:
        """
        Get all memory categories if category management is enabled.

        Returns:
            dictionary mapping category IDs to lists of memory IDs
        """
        if not self.category_manager:
            return {}

        categories = {}

        # Get all categories
        for category_id in range(self.category_manager._next_category_id):
            try:
                # Get memories in this category
                memory_ids = self.category_manager.get_category_members(category_id)
                if memory_ids:
                    categories[str(category_id)] = memory_ids
            except KeyError:
                continue

        return categories

    def clear_memories(self, keep_personal_attributes: bool = True) -> int:
        """
        Clear all memories except personal attributes if specified.

        Args:
            keep_personal_attributes: Whether to keep personal attribute memories

        Returns:
            Number of memories removed
        """
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

            return count_before - count_kept
        else:
            # Count memories before clearing
            count_before = len(self.memory_store.get_all())

            # Clear all memories
            self.memory_store.clear()
            self.memory_store_adapter.invalidate_cache()

            return count_before

    def get_retrieval_stats(self) -> dict[str, Any]:
        """
        Get statistics about memory retrieval performance.

        Returns:
            dictionary of retrieval statistics
        """
        return self.retrieval_stats.copy()

    def update_memory(
        self, memory_id: str, new_text: str = None, new_metadata: dict = None
    ) -> bool:
        """
        Update an existing memory with new text or metadata.

        Args:
            memory_id: ID of the memory to update
            new_text: New text content (or None to keep existing)
            new_metadata: New metadata (or None to keep existing)

        Returns:
            True if update was successful, False otherwise
        """
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

                return True

            # If only updating metadata
            elif new_metadata is not None:
                self.memory_store.update_metadata(memory_id, new_metadata)
                self.memory_store_adapter.invalidate_cache()
                return True

            return False

        except KeyError:
            logger.warning(f"Memory with ID {memory_id} not found")
            return False
        except Exception as e:
            logger.error(f"Error updating memory: {e}")
            return False

    def _consolidate_memories(self) -> int:
        """
        Consolidate memories to manage memory capacity.

        Returns:
            Number of memories consolidated
        """
        # Skip if no consolidation needed
        if len(self.memory_store.get_all()) <= self.max_memories:
            self.memories_since_consolidation = 0
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

        return len(removed_ids)

    def chat_without_memory(self, user_message: str, max_new_tokens: int = 512) -> str:
        """
        A baseline method that does not use memory retrieval at all.

        Args:
            user_message: The user's message
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            The assistant's response without memory context
        """
        prompt = f"You are a helpful assistant.\n\nUser: {user_message}\nAssistant:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )

        full_response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )
        assistant_response = full_response[len(prompt) :].strip()
        return assistant_response
