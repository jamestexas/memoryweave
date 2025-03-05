"""
MemoryWeaveAPI: A unified interface for managing and retrieving memories with advanced contextual understanding.

This module provides the MemoryWeaveAPI class, which integrates a large language model (LLM),
embedding models, and various memory management components to create a sophisticated memory system.

Key Features:
- Memory Storage and Retrieval: Stores and retrieves memories using embeddings and contextual strategies.
- Contextual Understanding: Leverages query analysis, keyword expansion, and temporal context to enhance retrieval.
- Dynamic Threshold Adjustment: Adapts retrieval thresholds based on query type and performance.
- Category Management: Organizes memories into categories for efficient retrieval.
- Personal Attribute Extraction: Extracts and stores personal attributes from user interactions.
- Associative Linking: Creates links between related memories for associative retrieval.
- Semantic Coherence: Ensures retrieved memories are semantically relevant to the query.
- Conversation History: Maintains a history of interactions for contextual responses.
- Memory Consolidation: Periodically consolidates memories to optimize performance.

Components:
- MemoryStore: Stores and retrieves memories as embeddings and text.
- MemoryStoreAdapter: Adapts the MemoryStore for use with retrieval strategies.
- AssociativeMemoryLinker: Creates and manages associative links between memories.
- TemporalContextBuilder: Builds temporal context for time-sensitive queries.
- ActivationManager: Manages memory activation levels.
- SimpleQueryAnalyzer: Analyzes user queries to determine type and extract keywords.
- QueryTypeAdapter: Adapts retrieval parameters based on query type.
- KeywordExpander: Expands keywords to improve retrieval.
- CategoryManager: Manages memory categories.
- PersonalAttributeManager: Extracts and manages personal attributes.
- SemanticCoherenceProcessor: Ensures semantic coherence in retrieved memories.
- DynamicThresholdAdjuster: Dynamically adjusts retrieval thresholds.
- ContextualFabricStrategy: A retrieval strategy that combines various contextual factors.

Usage:
    Create an instance of MemoryWeaveAPI, add memories, and use the chat method to interact.

Example:
    ```python
    from memoryweave.api import MemoryWeaveAPI

    api = MemoryWeaveAPI()
    api.add_memory("The capital of France is Paris.")
    response = api.chat("What is the capital of France?")
    print(response)
    ```

Dependencies:
    - torch
    - transformers
    - typing
    - logging
    - time
    - memoryweave.api.memory_store
    - memoryweave.components.*
    - memoryweave.interfaces.retrieval
    - memoryweave.query.analyzer
    - memoryweave.storage.memory_store
"""  # noqa: W291, W505

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
        self.device = _get_device(device)
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
        """Process the user message and generate a response using various subsystems.

        Args:
            user_message: The user's message.
            max_new_tokens: Maximum tokens to generate for the response.

        Returns:
            The assistant's response as a string.
        """
        start_time = time.time()
        now = start_time

        # Step 1: Query analysis
        _query_obj, adapted_params, expanded_keywords, query_type, entities = self._analyze_query(
            user_message
        )

        # Step 2: Compute query embedding
        query_embedding = self._compute_embedding(user_message)
        if query_embedding is None:
            return "Sorry, an error occurred while processing your request."

        # Step 3: Adjust threshold if dynamic adjustment is enabled
        self._adjust_confidence_threshold(query_type, adapted_params)

        # Step 4: Retrieve memories
        relevant_memories = self._retrieve_memories(
            query_embedding,
            user_message,
            query_type,
            expanded_keywords,
            entities,
            now,
            adapted_params,
        )

        # Step 5: Update memory activations and apply semantic coherence
        self._update_memory_activation(relevant_memories, now)
        relevant_memories = self._apply_semantic_coherence(relevant_memories, query_embedding)

        # Step 6: Apply temporal context and clean results
        cleaned_results = self._apply_temporal_context_and_clean(
            user_message, relevant_memories, now
        )

        # Step 7: Extract personal attributes if enabled
        self._extract_personal_attributes(user_message, now)

        # Step 8: Construct prompt with system instructions and conversation history
        prompt = self._construct_prompt(cleaned_results, query_type, user_message)

        # Step 9: Generate assistant response using LLM
        assistant_reply = self._generate_response(prompt, max_new_tokens)

        # Step 10: Store interaction and update history/statistics
        self._store_interaction(user_message, assistant_reply, now)
        self._update_conversation_history(user_message, assistant_reply)
        self._update_retrieval_stats(start_time, len(cleaned_results))
        self._update_dynamic_threshold(query_type, len(cleaned_results))

        return assistant_reply

    def _analyze_query(self, user_message: str) -> tuple:
        """Analyze the user message and extract query details.

        Returns:
            A tuple containing:
                - query_obj (dict)
                - adapted_params (dict)
                - expanded_keywords (list)
                - query_type (QueryType)
                - entities (list)
        """
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
            if self.keyword_expander:
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

    def _compute_embedding(self, user_message: str) -> any:
        """Compute and return the embedding for the user message."""
        try:
            return self.embedding_model.encode(
                user_message, show_progress_bar=self.show_progress_bar
            )
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _adjust_confidence_threshold(self, query_type: any, adapted_params: dict) -> None:
        """Dynamically adjust the confidence threshold if enabled."""
        if self.dynamic_threshold_adjuster:
            adjusted_threshold = self.dynamic_threshold_adjuster.get_adjusted_threshold(
                query_type, adapted_params.get("confidence_threshold", 0.1)
            )
            adapted_params["confidence_threshold"] = adjusted_threshold
            logger.debug(f"Adjusted confidence threshold: {adjusted_threshold}")

    def _retrieve_memories(
        self,
        query_embedding: any,
        user_message: str,
        query_type: any,
        expanded_keywords: list,
        entities: list,
        now: float,
        adapted_params: dict,
    ) -> list:
        """Retrieve relevant memories using the primary strategy, with a fallback.

        Returns:
            A list of memory items.
        """
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
            # If the strategy doesn't support adapted parameters, remove them from the context.
            if not hasattr(self.strategy, "_apply_adapted_params"):
                retrieval_context.pop("adapted_retrieval_params", None)

            relevant_memories = self.strategy.retrieve(
                query_embedding=query_embedding,
                top_k=adapted_params.get("max_results", 10),
                context=retrieval_context,
            )
        except Exception as e:
            logger.error(f"Error using primary retrieval strategy: {e}")
            import traceback

            traceback.print_exc()
            relevant_memories = []
        # Fallback retrieval if primary returned nothing
        if not relevant_memories:
            try:
                logger.warning("Primary retrieval returned no results, using fallback")
                fallback_results = self.memory_store_adapter.search_by_vector(
                    query_embedding, limit=10
                )
                relevant_memories = [
                    {
                        "memory_id": None,
                        "relevance_score": item.get("score", 0.5),
                        "content": item.get("content", ""),
                        "metadata": item.get("metadata", {}),
                    }
                    for item in fallback_results
                ]
                logger.debug(f"Fallback retrieved {len(relevant_memories)} memories")
            except Exception as e:
                logger.error(f"Even fallback retrieval failed: {e}")
                import traceback

                traceback.print_exc()
                relevant_memories = []
        return relevant_memories

    def _update_memory_activation(self, memories: list, now: float) -> None:
        """Update activation and timestamps for retrieved memories."""
        for mem_dict in memories:
            mem_id = mem_dict.get("memory_id")
            if mem_id is not None:
                self.activation_manager.activate_memory(mem_id, 0.2, spread=True)
                self.temporal_context.update_timestamp(mem_id, now)

    def _apply_semantic_coherence(self, memories: list, query_embedding: any) -> list:
        """Apply semantic coherence check to the memories if enabled."""
        if self.semantic_coherence_processor and len(memories) > 1:
            try:
                coherent_results = self.semantic_coherence_processor.process_results(
                    memories, query_embedding
                )
                if coherent_results:
                    memories = coherent_results
                    logger.debug(f"Applied semantic coherence, now have {len(memories)} results")
            except Exception as e:
                logger.error(f"Error in semantic coherence processing: {e}")
        return memories

    def _apply_temporal_context_and_clean(
        self, user_message: str, memories: list, now: float
    ) -> list:
        """Apply temporal context and prepare cleaned memory results for prompt inclusion."""
        base_context = {"current_time": now}
        memory_dicts = []
        for r in memories:
            base_score = r.get("relevance_score", 0.5)
            meta = r.get("metadata", {})
            created_at = meta.get("created_at", 0.0) if isinstance(meta, dict) else 0.0
            memory_dicts.append({
                "memory_id": r.get("memory_id"),
                "created_at": created_at,
                "relevance_score": base_score,
                "content": r.get("content", ""),
                "metadata": meta,
            })

        memory_dicts = self.temporal_context.apply_temporal_context(
            query=user_message, results=memory_dicts, context=base_context
        )
        memory_dicts.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)

        # Clean results: retrieve full content from memory store if possible
        cleaned_results = []
        for md in memory_dicts:
            mem_id = md.get("memory_id")
            if mem_id is None:
                continue
            try:
                memory_obj = None
                try:
                    memory_obj = self.memory_store.get(str(mem_id))
                except KeyError:
                    logger.debug(f"No memory found for ID {mem_id}")
                if memory_obj:
                    content = (
                        memory_obj.content.get("text")
                        if isinstance(memory_obj.content, dict)
                        else str(memory_obj.content)
                    )
                    cleaned_results.append({
                        "content": content,
                        "metadata": memory_obj.metadata,
                        "relevance_score": md.get("relevance_score", 0.5),
                    })
            except Exception as e:
                logger.error(f"Error retrieving memory {mem_id}: {e}")
        logger.debug(f"After temporal context, have {len(cleaned_results)} memories")
        return cleaned_results

    def _extract_personal_attributes(self, user_message: str, now: float) -> None:
        """Extract personal attributes from the user message and store them as synthetic memories."""  # noqa: W505
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
                                    "created_at": now,
                                    "importance": 0.9,
                                },
                            )
            except Exception as e:
                logger.error(f"Error extracting personal attributes: {e}")

    def _construct_prompt(self, cleaned_memories: list, query_type: any, user_message: str) -> str:
        """Construct the final prompt using system instructions, memory highlights, and conversation history."""  # noqa: W505
        system_prompt = (
            "You are a helpful assistant. Use the entire conversation context for your answer.\n"
            "Do not disclose that you have a memory system. If asked about user info, incorporate "
            "it naturally if available.\n"
        )
        # Choose number of top memories based on query type
        max_memories = 5 if query_type in [QueryType.PERSONAL, QueryType.TEMPORAL] else 3
        top_memories = sorted(
            cleaned_memories, key=lambda x: x.get("relevance_score", 0), reverse=True
        )[:max_memories]

        memory_text = ""
        if top_memories:
            memory_text = "MEMORY HIGHLIGHTS:\n"
            for m in top_memories:
                content = m["content"]
                memory_text += f"- {content[:150]}...\n" if len(content) > 150 else f"- {content}\n"
            memory_text += "\n"

        final_system_prompt = system_prompt + memory_text
        # Append conversation history
        history_text = ""
        max_history_turns = 10
        recent_history = (
            self.conversation_history[-max_history_turns:] if self.conversation_history else []
        )
        for turn in recent_history:
            role = "User" if turn["role"] == "user" else "Assistant"
            history_text += f"{role}: {turn['content']}\n"
        prompt = f"{final_system_prompt}\n{history_text}User: {user_message}\nAssistant:"
        return prompt

    def _generate_response(self, prompt: str, max_new_tokens: int) -> str:
        """Generate a response from the model based on the provided prompt."""
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
        return full_response[len(prompt) :].strip()

    def _store_interaction(self, user_message: str, assistant_reply: str, now: float) -> None:
        """Store the user and assistant messages into the memory store."""
        try:
            user_emb = self.embedding_model.encode(
                user_message, show_progress_bar=self.show_progress_bar
            )
            assistant_emb = self.embedding_model.encode(
                assistant_reply, show_progress_bar=self.show_progress_bar
            )
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
            if self.category_manager:
                self.category_manager.add_to_category(user_mem_id, user_emb)
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
            if self.category_manager:
                self.category_manager.add_to_category(assistant_mem_id, assistant_emb)
            if self.associative_linker:
                self.associative_linker.create_associative_link(user_mem_id, assistant_mem_id, 0.9)
            self.memory_store_adapter.invalidate_cache()
            self.memories_since_consolidation += 2
            if self.memories_since_consolidation >= self.consolidation_interval:
                self._consolidate_memories()
        except Exception as e:
            logger.error(f"Error storing conversation in memory: {e}")

    def _update_conversation_history(self, user_message: str, assistant_reply: str) -> None:
        """Append the latest user and assistant messages to the conversation history."""
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_reply})

    def _update_retrieval_stats(self, start_time: float, results_count: int) -> None:
        """Update the statistics for retrieval performance."""
        query_time = time.time() - start_time
        stats = self.retrieval_stats
        stats["total_queries"] += 1
        stats["successful_retrievals"] += 1 if results_count > 0 else 0
        n = stats["total_queries"]
        stats["avg_query_time"] = ((n - 1) * stats["avg_query_time"] + query_time) / n
        stats["avg_results_count"] = ((n - 1) * stats["avg_results_count"] + results_count) / n

    def _update_dynamic_threshold(self, query_type: any, results_count: int) -> None:
        """Provide feedback to update the dynamic confidence threshold if enabled."""
        if self.dynamic_threshold_adjuster:
            self.dynamic_threshold_adjuster.update_threshold(
                query_type, results_count, had_good_results=results_count > 0
            )

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

        logger.debug(f"Found {len(results)} memories matching keyword: {keyword}")
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
        logger.debug(f"Finding similar memories for memory ID: {memory_id}")
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
            filtered_memories = [m for m in similar_memories if m.get("memory_id") != memory_id]
            logger.debug(
                f"Found {len(filtered_memories)} similar memories for memory ID: {memory_id}"
            )
            return filtered_memories

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
        logger.debug("Getting memory categories")
        if not self.category_manager:
            logger.debug("Category management is disabled")
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

        logger.debug(f"Found {len(categories)} memory categories")
        return categories

    def clear_memories(self, keep_personal_attributes: bool = True) -> int:
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

    def get_retrieval_stats(self) -> dict[str, Any]:
        """
        Get statistics about memory retrieval performance.

        Returns:
            dictionary of retrieval statistics
        """
        logger.debug("Getting retrieval statistics")
        stats = self.retrieval_stats.copy()
        logger.debug(f"Retrieval statistics: {stats}")
        return stats

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

    def _consolidate_memories(self) -> int:
        """
        Consolidate memories to manage memory capacity.

        Returns:
            Number of memories consolidated
        """
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

    def chat_without_memory(self, user_message: str, max_new_tokens: int = 512) -> str:
        """
        A baseline method that does not use memory retrieval at all.

        Args:
            user_message: The user's message
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            The assistant's response without memory context
        """
        logger.debug("Generating response without memory context.")
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
        logger.debug("Response generated without memory context.")
        return assistant_response
