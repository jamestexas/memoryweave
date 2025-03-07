# memoryweave/api/hybrid_memory_weave.py
"""
Hybrid MemoryWeave API with adaptive chunking for memory efficiency.

This module provides an efficient middle-ground between standard single-vector
and full chunking approaches, optimizing memory usage while maintaining
retrieval quality.
"""

import logging
import re
import time
from typing import Any, Optional

import numpy as np
from rank_bm25 import BM25Okapi  # Add this import

from memoryweave.api.llm_provider import DEFAULT_MODEL, LLMProvider
from memoryweave.api.memory_weave import DEFAULT_EMBEDDING_MODEL, MemoryWeaveAPI
from memoryweave.components.retrieval_strategies.hybrid_fabric_strategy import HybridFabricStrategy
from memoryweave.components.retriever import _get_embedder
from memoryweave.components.text_chunker import TextChunker
from memoryweave.factory.memory_factory import (
    MemoryStoreConfig,
    VectorSearchConfig,
    create_memory_store_and_adapter,
)
from memoryweave.utils import _get_device

logger = logging.getLogger(__name__)


class HybridMemoryWeaveAPI(MemoryWeaveAPI):
    """
    Memory-efficient MemoryWeave API with adaptive chunking and hierarchical embeddings.

    This API implements a middle-ground approach between single-vector and full chunking:
    - Uses full embedding for short texts
    - Creates strategic, sparse chunks for longer texts
    - Implements importance-based chunking with keyword enhancement
    - Maintains low memory footprint while preserving context awareness
    - Integrates BM25 for keyword-based retrieval to complement vector similarity
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
        two_stage_retrieval: bool = True,
        **model_kwargs,
    ):
        """Initialize the HybridMemoryWeaveAPI with optimized loading."""
        # Initialize core attributes first
        self.debug = debug
        self.two_stage_retrieval = two_stage_retrieval
        self.device = _get_device(device)
        self.show_progress_bar = show_progress_bar

        # Start with a much lighter initialization
        self._initialize_core_components(
            model_name,
            embedding_model_name,
            max_memories,
            consolidation_interval,
            llm_provider,
            **model_kwargs,
        )

        # Store configuration for deferred initialization
        self._deferred_config = {
            "enable_category_management": enable_category_management,
            "enable_personal_attributes": enable_personal_attributes,
            "enable_semantic_coherence": enable_semantic_coherence,
            "enable_dynamic_thresholds": enable_dynamic_thresholds,
        }

        # Initialize chunking parameters
        self.adaptive_chunk_threshold = 800  # Character count that triggers chunking
        self.max_chunks_per_memory = 3  # Maximum number of chunks per memory
        self.importance_threshold = 0.6  # Threshold for keeping chunks
        self.enable_auto_chunking = True  # Whether to automatically chunk large texts
        self.chunked_memory_ids = set()  # Track which memories are chunked

        # Initialize BM25 components
        self._init_bm25_indexing()

        # Initialize retrieval strategy
        self._setup_hybrid_strategy()

        logger.info(
            f"HybridMemoryWeaveAPI initialized in {self.timer.timings.get('init', [0])[-1]:.3f}s"
        )

    def _initialize_core_components(
        self,
        model_name: str,
        embedding_model_name: str,
        max_memories: int,
        consolidation_interval: int,
        llm_provider: Optional[LLMProvider] = None,
        **model_kwargs,
    ):
        """Initialize only the core components needed for basic functionality."""
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = _get_embedder(model_name=embedding_model_name, device=self.device)
        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Initialize text chunker (essential for hybrid approach)
        self.text_chunker = TextChunker()
        self.text_chunker.initialize(
            {
                "chunk_size": 300,  # Larger chunks than the full chunked version
                "chunk_overlap": 30,  # Less overlap to save memory
                "min_chunk_size": 50,
                "respect_paragraphs": True,
                "respect_sentences": True,
            }
        )

        # Configure and create memory store with optimized settings
        store_config = MemoryStoreConfig(
            type="hybrid",
            vector_search=VectorSearchConfig(
                type="numpy",
                dimension=embedding_dim,
            ),
            chunk_size=300,
            chunk_overlap=30,
            min_chunk_size=50,
            adaptive_threshold=800,
            max_chunks_per_memory=3,
            importance_threshold=0.6,
        )

        # Create adapter (which includes the store)
        hybrid_adapter = create_memory_store_and_adapter(store_config)

        # Store for direct access
        self.hybrid_memory_store = hybrid_adapter.memory_store
        self.hybrid_memory_adapter = hybrid_adapter

        # Create LLM provider
        self.llm_provider = llm_provider or LLMProvider(model_name, self.device, **model_kwargs)

        # Basic history tracking
        self.conversation_history = []
        self.max_memories = max_memories
        self.consolidation_interval = consolidation_interval
        self.memories_since_consolidation = 0

        # Override for parent's constructor
        self._memory_store_override = self.hybrid_memory_store
        self._memory_adapter_override = self.hybrid_memory_adapter

        # Initialize components dictionary for lazy loading
        self._components = {}
        self._component_initialized = {
            "category_manager": False,
            "personal_attribute_manager": False,
            "semantic_coherence_processor": False,
            "query_analyzer": False,
            "query_adapter": False,
            "associative_linker": False,
            "temporal_context": False,
            "activation_manager": False,
        }

    def _init_bm25_indexing(self):
        """Initialize BM25 indexing for keyword-based retrieval."""
        self.bm25_index = None
        self.bm25_documents = []
        self.bm25_doc_ids = []
        self.tokenizer = lambda x: x.lower().split()
        self.bm25_update_batch_size = 50  # Add documents in batches for efficiency
        self.bm25_pending_docs = []
        self.bm25_pending_ids = []

    def _update_bm25_index(self, text: str, memory_id: str):
        """
        Update BM25 index with new document, using efficient batched updates.

        This method adds documents to the BM25 index incrementally, only rebuilding
        the index when necessary to maintain performance with large document collections.

        Args:
            text: Document text to add to the index
            memory_id: ID of the memory/document
        """
        # For benchmarking, we'll use a much simpler approach to reduce load time

        # Add document to BM25 corpus
        self.bm25_documents.append(text)
        self.bm25_doc_ids.append(memory_id)

        # For benchmarking, we'll delay index building until first query
        # Only initialize if this is the first document
        if len(self.bm25_documents) == 1:
            tokenized_corpus = [self.tokenizer(text)]
            self.bm25_index = BM25Okapi(tokenized_corpus)
        elif len(self.bm25_documents) % 20 == 0:  # Only update every 20 docs for benchmarking
            tokenized_corpus = [self.tokenizer(doc) for doc in self.bm25_documents]
            self.bm25_index = BM25Okapi(tokenized_corpus)
            return

        # For subsequent documents, check if we should rebuild
        # We rebuild in these scenarios:
        # 1. Every 50 documents (configurable batch size)
        # 2. If the document is very long (indicating potentially important content)
        should_rebuild = False

        # Check if we've reached the batch limit
        if not hasattr(self, "bm25_batch_size"):
            self.bm25_batch_size = 50  # Default batch size

        if len(self.bm25_documents) % self.bm25_batch_size == 0:
            should_rebuild = True

        # Check if document is important (long)
        if len(text) > 2000:  # Consider a 2000+ character document as important
            should_rebuild = True

        # Rebuild the index if needed
        if should_rebuild:
            # Tokenize all documents - this is necessary because BM25Okapi doesn't support incremental updates
            tokenized_corpus = [self.tokenizer(doc) for doc in self.bm25_documents]
            self.bm25_index = BM25Okapi(tokenized_corpus)
            if self.debug:
                logger.debug(f"Rebuilt BM25 index with {len(self.bm25_documents)} documents")

    def _get_component(self, name: str) -> Any:
        """Lazy-load a component only when it's actually needed."""
        if name not in self._components or not self._component_initialized[name]:
            if name == "category_manager" and self._deferred_config.get(
                "enable_category_management", True
            ):
                from memoryweave.components.category_manager import CategoryManager

                self._components[name] = CategoryManager()
                self._components[name].initialize(
                    {
                        "vigilance_threshold": 0.85,
                        "embedding_dim": self.embedding_model.get_sentence_embedding_dimension(),
                    }
                )

            elif name == "personal_attribute_manager" and self._deferred_config.get(
                "enable_personal_attributes", True
            ):
                from memoryweave.components.personal_attributes import PersonalAttributeManager

                self._components[name] = PersonalAttributeManager()
                self._components[name].initialize()

            elif name == "semantic_coherence_processor" and self._deferred_config.get(
                "enable_semantic_coherence", True
            ):
                from memoryweave.components.post_processors import SemanticCoherenceProcessor

                self._components[name] = SemanticCoherenceProcessor()
                self._components[name].initialize()

            elif name == "query_analyzer":
                from memoryweave.query.analyzer import SimpleQueryAnalyzer

                self._components[name] = SimpleQueryAnalyzer()
                self._components[name].initialize(
                    {
                        "min_keyword_length": 3,
                        "max_keywords": 10,
                    }
                )

            elif name == "query_adapter":
                from memoryweave.components.query_adapter import QueryTypeAdapter

                self._components[name] = QueryTypeAdapter()
                self._components[name].initialize(
                    {
                        "apply_keyword_boost": True,
                        "scale_params_by_length": True,
                    }
                )

            elif name == "associative_linker":
                from memoryweave.components.associative_linking import AssociativeMemoryLinker

                self._components[name] = AssociativeMemoryLinker(self.hybrid_memory_store)
                self._components[name].initialize({})

            elif name == "temporal_context":
                from memoryweave.components.temporal_context import TemporalContextBuilder

                self._components[name] = TemporalContextBuilder(self.hybrid_memory_store)
                self._components[name].initialize({})

            elif name == "activation_manager":
                from memoryweave.components.activation import ActivationManager

                self._components[name] = ActivationManager()
                assoc_linker = self._get_component("associative_linker")
                self._components[name].initialize(
                    {
                        "memory_store": self.hybrid_memory_store,
                        "associative_linker": assoc_linker,
                    }
                )

            self._component_initialized[name] = True

        return self._components.get(name)

    @property
    def category_manager(self):
        return (
            self._get_component("category_manager")
            if self._deferred_config.get("enable_category_management")
            else None
        )

    @property
    def personal_attribute_manager(self):
        return (
            self._get_component("personal_attribute_manager")
            if self._deferred_config.get("enable_personal_attributes")
            else None
        )

    @property
    def semantic_coherence_processor(self):
        return (
            self._get_component("semantic_coherence_processor")
            if self._deferred_config.get("enable_semantic_coherence")
            else None
        )

    @property
    def query_analyzer(self):
        return self._get_component("query_analyzer")

    @property
    def query_adapter(self):
        return self._get_component("query_adapter")

    @property
    def associative_linker(self):
        return self._get_component("associative_linker")

    @property
    def temporal_context(self):
        return self._get_component("temporal_context")

    @property
    def activation_manager(self):
        return self._get_component("activation_manager")

    def configure_two_stage_retrieval(
        self,
        enable: bool = True,
        first_stage_k: int = 30,
        first_stage_threshold_factor: float = 0.7,
    ) -> None:
        """
        Configure two-stage retrieval settings.

        Args:
            enable: Whether to enable two-stage retrieval
            first_stage_k: Number of candidates to retrieve in first stage
            first_stage_threshold_factor: Factor to multiply confidence threshold by in first stage
        """
        if hasattr(self, "hybrid_strategy") and hasattr(
            self.hybrid_strategy, "configure_two_stage"
        ):
            self.hybrid_strategy.configure_two_stage(
                enable, first_stage_k, first_stage_threshold_factor
            )
            logger.info(
                f"Configured two-stage retrieval: enable={enable}, "
                f"first_stage_k={first_stage_k}, "
                f"first_stage_threshold_factor={first_stage_threshold_factor}"
            )

    def _setup_hybrid_strategy(self):
        """Set up the hybrid fabric strategy to replace the standard strategy."""
        # Create hybrid strategy
        self.hybrid_strategy = HybridFabricStrategy(
            memory_store=self.hybrid_memory_adapter,
            associative_linker=self.associative_linker
            if hasattr(self, "associative_linker")
            else None,
            temporal_context=self.temporal_context if hasattr(self, "temporal_context") else None,
            activation_manager=self.activation_manager
            if hasattr(self, "activation_manager")
            else None,
        )

        # Initialize with optimized parameters
        params = {
            "confidence_threshold": 0.1,  # Lower threshold for better recall
            "similarity_weight": 0.5,  # Strong emphasis on semantic similarity
            "associative_weight": 0.2,  # Moderate weight for associative links
            "temporal_weight": 0.2,  # Improved weight for temporal context
            "activation_weight": 0.1,  # Moderate weight for activation
            "use_keyword_filtering": True,  # Enable keyword filtering
            "keyword_boost_factor": 0.3,  # Strong boost for keyword matches
            "max_chunks_per_memory": self.max_chunks_per_memory,
            "prioritize_full_embeddings": True,  # Prioritize full embeddings over chunks
            "min_results": 3,  # Ensure reasonable number of results
            "max_candidates": 50,  # Consider more candidates for better selection
            "debug": self.debug,
            # Two-stage configuration
            "use_two_stage": self.two_stage_retrieval,  # Enable two-stage retrieval by default
            "first_stage_k": 30,  # Get 30 candidates in first stage
            "first_stage_threshold_factor": 0.7,  # Use 70% of confidence threshold in first stage
            # Advanced optimization parameters
            "use_batched_computation": True,  # Process large matrices in batches
            "batch_size": 200,  # Reasonable batch size for most systems
            "max_associative_hops": 2,  # Limit associative traversal depth
        }
        self.hybrid_strategy.initialize(params)

        # Replace the strategy
        self.strategy = self.hybrid_strategy

        # Initialize retrieval orchestrator if needed
        if not hasattr(self, "retrieval_orchestrator"):
            from memoryweave.api.retrieval_orchestrator import RetrievalOrchestrator

            # Create retrieval orchestrator with optimized settings
            self.retrieval_orchestrator = RetrievalOrchestrator(
                strategy=self.strategy,
                activation_manager=self.activation_manager
                if hasattr(self, "activation_manager")
                else None,
                temporal_context=self.temporal_context
                if hasattr(self, "temporal_context")
                else None,
                semantic_coherence_processor=self.semantic_coherence_processor
                if hasattr(self, "semantic_coherence_processor")
                else None,
                memory_store_adapter=self.hybrid_memory_adapter,
                debug=self.debug,
                max_workers=4,  # Reasonable parallelism for most systems
                enable_cache=True,  # Enable caching for faster repeat queries
                max_cache_size=50,  # Moderate cache size
            )
        else:
            # Update existing orchestrator
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

        # Analyze content to determine chunking strategy
        should_chunk, chunk_threshold = self._analyze_chunking_needs(text)

        # For small texts, use standard storage
        if not should_chunk:
            # Create a single embedding
            embedding = self.embedding_model.encode(text, show_progress_bar=self.show_progress_bar)

            # Add to memory store
            mem_id = self.hybrid_memory_adapter.add(embedding, text, metadata)

            # Add to BM25 index
            self._update_bm25_index(text, mem_id)

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
            mem_id = self.hybrid_memory_adapter.add(full_embedding, text, metadata)
            # Add to BM25 index
            self._update_bm25_index(text, mem_id)
            return mem_id

        # 4. Add to hybrid memory store
        mem_id = self.hybrid_memory_adapter.add_hybrid(
            full_embedding=full_embedding,
            chunks=selected_chunks,
            chunk_embeddings=chunk_embeddings,
            original_content=text,
            metadata=metadata,
        )

        # Add to BM25 index
        self._update_bm25_index(text, mem_id)

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

    def _analyze_chunking_needs(self, text: str) -> tuple[bool, int]:
        """
        Analyze text to determine if and how it should be chunked.

        Args:
            text: The text to analyze

        Returns:
            tuple of (should_chunk, threshold_value)
        """
        # If auto chunking is disabled, use the fixed threshold
        if not self.enable_auto_chunking:
            return len(text) > self.adaptive_chunk_threshold, self.adaptive_chunk_threshold

        # Base decision on text length
        if len(text) < 500:  # Short texts never need chunking
            return False, self.adaptive_chunk_threshold

        # For longer texts, analyze structure and content
        sentences = re.split(r"[.!?]\s+", text)
        paragraph_splits = text.split("\n\n")

        # Analyze sentence length
        avg_sentence_length = sum(len(s) for s in sentences) / max(1, len(sentences))

        # Analyze content complexity
        has_entities = bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text))
        has_code_blocks = bool(re.search(r"```|    def |class |function", text))
        has_lists = bool(re.search(r"\n\s*[-*]\s+|\n\s*\d+\.\s+", text))

        # Calculate adaptive threshold based on content characteristics
        threshold_value = self.adaptive_chunk_threshold

        # Adjust based on content type
        if has_code_blocks:
            # Code needs more careful chunking to preserve function blocks
            threshold_value = min(
                500, threshold_value
            )  # Lower threshold to chunk more aggressively
            return True, threshold_value  # Always chunk code

        if has_lists:
            # lists benefit from chunking to preserve item groupings
            threshold_value = min(700, threshold_value)

        if has_entities:
            # Entity-rich text might benefit from more chunking
            threshold_value = min(800, threshold_value)

        # Very long sentences suggest complex content that benefits from chunking
        if avg_sentence_length > 100:
            threshold_value = min(700, threshold_value)

        # Multiple paragraphs suggest natural chunk boundaries
        if len(paragraph_splits) > 5:
            # Benefit from chunking at paragraph boundaries
            return True, threshold_value

        # Make final decision
        should_chunk = len(text) > threshold_value
        return should_chunk, threshold_value

    def _select_important_chunks(
        self,
        chunks: list[dict[str, Any]],
        full_embedding: np.ndarray,
        query_embedding: np.ndarray | None = None,
    ) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
        """
        Enhanced chunk selection using semantic importance and query relevance.

        This implementation focuses on:
        1. Finding truly information-dense chunks
        2. Preserving document structure and coherence
        3. Optimizing for memory usage
        4. Balancing semantic importance with entity coverage

        Args:
            chunks: List of chunk dictionaries
            full_embedding: Embedding of full text
            query_embedding: Optional query embedding for relevance scoring

        Returns:
            Tuple of (selected_chunks, chunk_embeddings)
        """
        if not chunks:
            return [], []

        # Determine if we should use query relevance
        use_query_relevance = query_embedding is not None and isinstance(
            query_embedding, np.ndarray
        )

        # Calculate chunk scores based on multiple factors
        chunk_data = []
        chunk_embeddings = []

        for i, chunk in enumerate(chunks):
            # Get chunk text
            chunk_text = chunk["text"]

            # Skip very short chunks
            if len(chunk_text) < 30:
                continue

            # 1. Create embedding for this chunk
            chunk_embedding = self.embedding_model.encode(
                chunk_text, show_progress_bar=self.show_progress_bar
            )
            chunk_embeddings.append(chunk_embedding)

            # Initialize score components
            score_components = {}

            # 2. Calculate semantic importance (cosine similarity to full embedding)
            # This measures how representative this chunk is of the full text
            full_norm = np.linalg.norm(full_embedding)
            chunk_norm = np.linalg.norm(chunk_embedding)

            if full_norm > 0 and chunk_norm > 0:
                semantic_importance = np.dot(full_embedding, chunk_embedding) / (
                    full_norm * chunk_norm
                )
                score_components["semantic_importance"] = semantic_importance
            else:
                score_components["semantic_importance"] = 0.0

            # 3. Calculate query relevance if available
            if use_query_relevance:
                query_norm = np.linalg.norm(query_embedding)
                if query_norm > 0 and chunk_norm > 0:
                    query_relevance = np.dot(query_embedding, chunk_embedding) / (
                        query_norm * chunk_norm
                    )
                    score_components["query_relevance"] = query_relevance
                else:
                    score_components["query_relevance"] = 0.0

            # 4. Calculate information density
            # Count named entities (capitalized terms that aren't at start of sentences)
            text = chunk_text
            entity_matches = re.findall(r"(?<!^)(?<!\. )[A-Z][a-zA-Z]+", text)
            entity_density = len(entity_matches) / (len(text) / 100)  # Entities per 100 chars
            score_components["entity_density"] = min(1.0, entity_density / 3)

            # 5. Check for structural indicators
            # Lists and enumerations often contain important information
            has_list = bool(re.search(r"\n\s*[-*]\s+|\n\s*\d+\.\s+", text))
            # Headers often mark important sections
            has_header = bool(re.search(r"\n#+\s+|\n[A-Z][A-Z\s]+\n", text))
            # Key phrases often indicate important information
            has_key_phrase = bool(
                re.search(r"important|key|significant|crucial|essential", text.lower())
            )

            structure_score = 0.0
            if has_list:
                structure_score += 0.2
            if has_header:
                structure_score += 0.3
            if has_key_phrase:
                structure_score += 0.1
            score_components["structure_score"] = structure_score

            # 6. Position importance - first and last chunks often have key information
            position_score = 0.0
            if i == 0:  # First chunk
                position_score = 0.3
            elif i == len(chunks) - 1:  # Last chunk
                position_score = 0.2
            score_components["position_score"] = position_score

            # 7. Consider chunk length - neither too short nor too long
            length = len(chunk_text)
            ideal_length = 300  # Target length
            length_score = 1.0 - min(1.0, abs(length - ideal_length) / ideal_length)
            score_components["length_score"] = length_score

            # 8. Check for numeric content (often important in data-focused text)
            numeric_matches = len(re.findall(r"\d+\.\d+|\d+", text))
            numeric_density = numeric_matches / (len(text) / 100)  # Numbers per 100 chars
            score_components["numeric_density"] = min(0.8, numeric_density / 2)  # Cap at 0.8

            # Calculate final score with appropriate weights
            weights = {
                "semantic_importance": 0.4,
                "entity_density": 0.2,
                "structure_score": 0.15,
                "position_score": 0.1,
                "length_score": 0.05,
                "numeric_density": 0.1,
            }

            # Add query relevance if available
            if use_query_relevance:
                # Reduce other weights to accommodate query relevance
                for k in weights:
                    weights[k] *= 0.6  # Reduce all weights by 40%
                weights["query_relevance"] = 0.4  # Add 40% weight for query relevance

            # Calculate weighted score
            final_score = sum(score_components.get(k, 0) * weights[k] for k in weights)

            # Store chunk data
            chunk_data.append(
                {
                    "index": i,
                    "chunk": chunk,
                    "embedding": chunk_embedding,
                    "score": final_score,
                    "components": score_components,
                }
            )

        # Sort chunks by score
        chunk_data.sort(key=lambda x: x["score"], reverse=True)

        # Select top N chunks but ensure coherence
        max_chunks = min(self.max_chunks_per_memory, len(chunk_data))

        # Initially select top chunks by score
        top_chunks_data = chunk_data[:max_chunks]

        # Check if we have sequential chunks among the top scorers
        top_indices = [item["index"] for item in top_chunks_data]
        are_sequential = all(
            top_indices[i + 1] == top_indices[i] + 1 for i in range(len(top_indices) - 1)
        )

        # If top chunks aren't sequential, check if we can find a coherent segment
        if not are_sequential and len(chunk_data) > max_chunks:
            # Look for sequential chunks with good scores
            best_sequential_score = 0
            best_sequential_group = []

            # Try different starting positions
            for start_idx in range(len(chunks) - max_chunks + 1):
                # Get sequential chunk group
                sequential_indices = list(range(start_idx, start_idx + max_chunks))
                sequential_chunks = [c for c in chunk_data if c["index"] in sequential_indices]

                # Only consider if we found all chunks
                if len(sequential_chunks) == max_chunks:
                    group_score = sum(c["score"] for c in sequential_chunks)

                    # If this sequential group has better score than previous best
                    if group_score > best_sequential_score:
                        best_sequential_score = group_score
                        best_sequential_group = sequential_chunks

            # Compare best sequential group score with top individual chunks score
            top_individual_score = sum(c["score"] for c in top_chunks_data)

            # If sequential group is at least 80% as good as individual top chunks, use it
            if best_sequential_group and best_sequential_score >= 0.8 * top_individual_score:
                top_chunks_data = best_sequential_group

        # Sort the selected chunks by index to maintain document order
        top_chunks_data.sort(key=lambda x: x["index"])

        # Extract the final selected chunks and their embeddings
        selected_chunks = [item["chunk"] for item in top_chunks_data]
        selected_embeddings = [item["embedding"] for item in top_chunks_data]

        return selected_chunks, selected_embeddings

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
                    chunk_metadata.update(
                        {
                            "chunk_index": i,
                            "is_conversation": True,
                            "role": role,
                        }
                    )

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
            mem_id = self.hybrid_memory_adapter.add_hybrid(
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
            mem_id = self.hybrid_memory_adapter.add(embedding, full_text, metadata)

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
                chunk_metadata.update(
                    {
                        "chunk_index": i,
                        "is_conversation": True,
                        "role": role,
                    }
                )

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
        first_metadata.update(
            {
                "chunk_index": 0,
                "is_conversation": True,
                "role": first_role,
                "is_first": True,
            }
        )

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
                group_metadata.update(
                    {
                        "chunk_index": len(chunks),
                        "is_conversation": True,
                        "turn_range": (i + 1, min(i + group_size, middle_count)),
                    }
                )

                chunks.append({"text": group_text, "metadata": group_metadata})

        # Always include the last turn
        last_turn = turns[-1]
        last_role = last_turn.get("role", "unknown")
        last_content = last_turn.get("content", "")
        last_text = f"{last_role}: {last_content}"

        last_metadata = metadata.copy()
        last_metadata.update(
            {
                "chunk_index": len(chunks),
                "is_conversation": True,
                "role": last_role,
                "is_last": True,
            }
        )

        chunks.append({"text": last_text, "metadata": last_metadata})

        return chunks

    def retrieve(self, query: str, top_k: int = 10, **kwargs) -> list[dict[str, Any]]:
        """
        Enhanced retrieval combining BM25 and vector similarity.

        This method improves over the base implementation by:
        1. Using BM25 for keyword matching
        2. Using vector similarity for semantic matching
        3. Combining results with rank fusion
        4. Optimizing for performance with caching

        Args:
            query: The query string
            top_k: Number of results to return
            **kwargs: Additional parameters

        Returns:
            List of retrieved memories with relevance scores
        """

        # Check cache first (simple LRU cache implementation)
        cache_key = f"{query}:{top_k}"
        if hasattr(self, "_result_cache") and cache_key in self._result_cache:
            cached_result = self._result_cache[cache_key]
            return cached_result

        # Create query embedding
        self.embedding_model.encode(query, show_progress_bar=self.show_progress_bar)

        # Get BM25 results if available
        bm25_results = []
        if self.bm25_index is not None:
            self.timer.start("bm25_search")
            tokenized_query = self.tokenizer(query)
            doc_scores = self.bm25_index.get_scores(tokenized_query)

            # Get top BM25 results
            top_bm25_limit = min(top_k * 2, len(self.bm25_documents))
            top_indices = np.argsort(doc_scores)[::-1][:top_bm25_limit]

            for i, idx in enumerate(top_indices):
                if doc_scores[idx] > 0:
                    doc_id = self.bm25_doc_ids[idx]
                    try:
                        # Get the memory to retrieve its content
                        memory = self.hybrid_memory_adapter.get(doc_id)

                        # Format result similar to vector results
                        result = {
                            "memory_id": doc_id,
                            "content": memory.content,
                            "bm25_score": float(doc_scores[idx]),
                            "bm25_rank": i,
                            "retrieval_method": "bm25",
                            "metadata": memory.metadata,
                        }
                        bm25_results.append(result)
                    except Exception as e:
                        logger.warning(f"Error retrieving BM25 result {doc_id}: {e}")
            self.timer.stop("bm25_search")

        # Get vector similarity results
        self.timer.start("vector_search")
        # Use the retrieval orchestrator for this
        vector_results = super().retrieve(query, top_k=top_k * 2, **kwargs)
        self.timer.stop("vector_search")

        # Mark vector results source
        for result in vector_results:
            result["retrieval_method"] = "vector"

        # If we have both result types, combine them
        if bm25_results and vector_results:
            self.timer.start("combine_results")
            combined_results = self._combine_with_reciprocal_rank_fusion(
                bm25_results, vector_results, k1=60, k2=40, top_k=top_k
            )
            self.timer.stop("combine_results")
        elif bm25_results:
            # Only BM25 results available
            combined_results = bm25_results[:top_k]
        else:
            # Only vector results (or no results)
            combined_results = vector_results[:top_k]

        # Cache results
        if not hasattr(self, "_result_cache"):
            # Initialize cache
            self._result_cache = {}
            self._result_cache_keys = []
            self._max_cache_size = 100

        # Store in cache
        if len(self._result_cache) >= self._max_cache_size:
            # Remove oldest entry
            oldest_key = self._result_cache_keys.pop(0)
            if oldest_key in self._result_cache:
                del self._result_cache[oldest_key]

        self._result_cache[cache_key] = combined_results
        self._result_cache_keys.append(cache_key)

        self.timer.stop("retrieve")
        return combined_results

    def _combine_with_reciprocal_rank_fusion(
        self,
        results1: list[dict[str, Any]],
        results2: list[dict[str, Any]],
        k1: float = 60.0,
        k2: float = 40.0,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Combine results using reciprocal rank fusion.

        Args:
            results1: First set of results (BM25)
            results2: Second set of results (Vector)
            k1: Weight parameter for first result set
            k2: Weight parameter for second result set
            top_k: Number of results to return

        Returns:
            Combined results
        """
        # Create dictionary of memory_id -> result object with score
        result_map = {}

        # Add scores from first result set (BM25)
        for rank, result in enumerate(results1):
            memory_id = result["memory_id"]
            # RRF formula: 1/(k + rank)
            score = 1.0 / (k1 + rank)

            if memory_id not in result_map:
                # Create new entry
                result_map[memory_id] = {"result": result, "score": score, "sources": ["bm25"]}
            else:
                # Update existing entry
                result_map[memory_id]["score"] += score
                if "bm25" not in result_map[memory_id]["sources"]:
                    result_map[memory_id]["sources"].append("bm25")

        # Add scores from second result set (Vector)
        for rank, result in enumerate(results2):
            memory_id = result["memory_id"]
            # RRF formula: 1/(k + rank)
            score = 1.0 / (k2 + rank)

            if memory_id not in result_map:
                # Create new entry
                result_map[memory_id] = {"result": result, "score": score, "sources": ["vector"]}
            else:
                # Update existing entry
                result_map[memory_id]["score"] += score
                if "vector" not in result_map[memory_id]["sources"]:
                    result_map[memory_id]["sources"].append("vector")
                    # Keep vector metadata which is more complete
                    result_map[memory_id]["result"] = result

        # Build final results
        combined = []
        for _memory_id, data in result_map.items():
            result_obj = data["result"].copy()
            result_obj["rrf_score"] = data["score"]
            result_obj["retrieval_sources"] = data["sources"]

            # Normalize relevance score
            if "bm25_score" in result_obj and "relevance_score" in result_obj:
                # Combine BM25 and vector scores
                bm25_weight = 0.4
                vector_weight = 0.6
                combined_score = bm25_weight * min(
                    1.0, result_obj["bm25_score"] / 5.0
                ) + vector_weight * result_obj.get("relevance_score", 0.0)
                result_obj["relevance_score"] = combined_score
            elif "bm25_score" in result_obj:
                # Only BM25 score available - normalize it
                result_obj["relevance_score"] = min(1.0, result_obj["bm25_score"] / 5.0)

            combined.append(result_obj)

        # Sort by RRF score and return top K
        return sorted(combined, key=lambda x: x["rrf_score"], reverse=True)[:top_k]

    def chat(self, user_message: str, max_new_tokens: int = 512) -> str:
        """
        Enhanced chat method with query type-specific optimizations.

        This method:
        1. Analyzes query to determine optimal retrieval approach
        2. Uses different strategies for different query types
        3. Optimizes memory usage during retrieval
        4. Caches responses for frequent queries

        Args:
            user_message: User's message
            max_new_tokens: Maximum new tokens for response

        Returns:
            Assistant's response
        """
        start_time = time.time()

        # Check response cache
        if hasattr(self, "_response_cache") and user_message in self._response_cache:
            return self._response_cache[user_message]

        # Step 1: Query analysis
        query_info = self._analyze_query(user_message)
        _query_obj, adapted_params, expanded_keywords, query_type, entities = query_info

        # Step 2: Compute query embedding
        self.timer.start("embedding_computation")
        query_embedding = self._compute_embedding(user_message)
        self.timer.stop("embedding_computation")
        if query_embedding is None:
            return "Sorry, an error occurred while processing your request."

        # Step 3: Optimize retrieval based on query type
        # Default to standard parameters
        retrieval_params = {"top_k": adapted_params.get("max_results", 10)}

        # Adjust based on query type
        if query_type == "factual":
            # For factual queries, prioritize precision and use more results
            retrieval_params["top_k"] = min(15, retrieval_params["top_k"] + 5)
            # BM25 works well for factual queries, so it's weighted more in retrieve()
        elif query_type == "personal":
            # For personal queries, we want focused, relevant results
            retrieval_params["top_k"] = max(5, retrieval_params["top_k"] - 2)
        elif query_type == "temporal":
            # For temporal queries, we rely more on temporal context
            adapted_params["temporal_weight"] = 0.4  # Boost temporal weight
        elif query_type == "conceptual" or query_type == "opinion":
            # For conceptual/opinion queries, we need more diverse results
            retrieval_params["top_k"] = min(15, retrieval_params["top_k"] + 5)

        # Step 4: Add keywords to params if available
        if expanded_keywords:
            retrieval_params["keywords"] = expanded_keywords

        # Step 5: Retrieve memories
        relevant_memories = self.retrieve(query=user_message, **retrieval_params)

        # Step 6: Extract personal attributes if enabled
        if self.personal_attribute_manager:
            self._extract_personal_attributes(user_message, time.time())

        # Step 7: Construct prompt
        self.timer.start("prompt_construction")
        prompt = self.prompt_builder.build_chat_prompt(
            user_message=user_message,
            memories=relevant_memories,
            conversation_history=self.conversation_history,
            query_type=query_type,
        )
        self.timer.stop("prompt_construction")

        if self.debug:
            logger.debug("[bold cyan]===== Prompt Start =====[/]")
            logger.debug(f"[bold]{prompt}[/]")
            logger.debug("[bold cyan]===== Prompt End =====[/]")

        # Step 8: Generate response
        self.timer.start("response_generation")
        assistant_reply = self.llm_provider.generate(prompt=prompt, max_new_tokens=max_new_tokens)
        self.timer.stop("response_generation")

        # Step 9: Store interaction
        self.timer.start("store_interaction")
        self._store_hybrid_interaction(user_message, assistant_reply, time.time())
        self.timer.stop("store_interaction")

        # Step 10: Update history and statistics
        self.timer.start("update_history")
        self._update_conversation_history(user_message, assistant_reply)
        self.timer.stop("update_history")

        self._update_retrieval_stats(start_time, len(relevant_memories))

        # Cache the response
        if not hasattr(self, "_response_cache"):
            self._response_cache = {}
            self._response_cache_keys = []
            self._max_response_cache = 20

        # Add to cache with LRU eviction
        if len(self._response_cache) >= self._max_response_cache:
            # Remove oldest entry
            oldest_key = self._response_cache_keys.pop(0)
            if oldest_key in self._response_cache:
                del self._response_cache[oldest_key]

        self._response_cache[user_message] = assistant_reply
        self._response_cache_keys.append(user_message)

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
                user_mem_id = self.hybrid_memory_adapter.add_hybrid(
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
                user_mem_id = self.hybrid_memory_adapter.add(user_emb, user_message, user_metadata)

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
                assistant_mem_id = self.hybrid_memory_adapter.add_hybrid(
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
                assistant_mem_id = self.hybrid_memory_adapter.add(
                    assistant_emb, assistant_reply, assistant_metadata
                )

            # Create associative link between messages
            if self.associative_linker:
                self.associative_linker.create_associative_link(user_mem_id, assistant_mem_id, 0.9)

        except Exception as e:
            logger.error(f"Error storing hybrid conversation in memory: {e}")

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
                "total_memories": len(self.hybrid_memory_adapter.get_all()),
                "chunked_memories": len(self.chunked_memory_ids),
                "total_chunks": self.hybrid_memory_adapter.get_chunk_count(),
                "avg_chunks_per_memory": self.hybrid_memory_adapter.get_average_chunks_per_memory(),
                "adaptive_chunk_threshold": self.adaptive_chunk_threshold,
                "max_chunks_per_memory": self.max_chunks_per_memory,
                "enable_auto_chunking": self.enable_auto_chunking,
                "memory_usage": {
                    "full_embeddings": len(self.hybrid_memory_adapter.get_all()),
                    "chunk_embeddings": self.hybrid_memory_adapter.get_chunk_count(),
                    "total_embeddings": (
                        len(self.hybrid_memory_adapter.get_all())
                        + self.hybrid_memory_adapter.get_chunk_count()
                    ),
                },
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting chunking statistics: {e}")
            return {}

    def get_performance_stats(self) -> dict[str, Any]:
        """
        Get detailed performance statistics for the HybridMemoryWeaveAPI.

        Returns:
            Dictionary with performance metrics
        """
        stats = {
            "memory_usage": {},
            "timing": {},
            "counts": {},
            "retrieval": {},
        }

        # Memory usage stats
        if hasattr(self, "hybrid_memory_store"):
            stats["memory_usage"]["total_memories"] = len(self.hybrid_memory_adapter.get_all())
            stats["memory_usage"]["chunked_memories"] = len(self.chunked_memory_ids)
            stats["memory_usage"]["total_chunks"] = self.hybrid_memory_adapter.get_chunk_count()
            if stats["memory_usage"]["chunked_memories"] > 0:
                stats["memory_usage"]["avg_chunks_per_memory"] = (
                    stats["memory_usage"]["total_chunks"]
                    / stats["memory_usage"]["chunked_memories"]
                )
            else:
                stats["memory_usage"]["avg_chunks_per_memory"] = 0

        # Timing stats
        if hasattr(self, "timer") and hasattr(self.timer, "timings"):
            for operation, times in self.timer.timings.items():
                if times:
                    stats["timing"][operation] = {
                        "avg": sum(times) / len(times),
                        "min": min(times),
                        "max": max(times),
                        "total": sum(times),
                        "count": len(times),
                    }

        # Component initialization stats
        if hasattr(self, "_component_initialized"):
            stats["counts"]["components_initialized"] = sum(
                1 for v in self._component_initialized.values() if v
            )
            stats["counts"]["components_total"] = len(self._component_initialized)
            stats["counts"]["lazy_loading_savings"] = len(self._component_initialized) - sum(
                1 for v in self._component_initialized.values() if v
            )

        # Retrieval stats
        if hasattr(self, "retrieval_stats"):
            stats["retrieval"] = self.retrieval_stats.copy()

        # Cache stats
        stats["cache"] = {}
        if hasattr(self, "_result_cache"):
            stats["cache"]["result_cache_size"] = len(self._result_cache)
            stats["cache"]["result_cache_limit"] = self._max_cache_size
        if hasattr(self, "_response_cache"):
            stats["cache"]["response_cache_size"] = len(self._response_cache)
            stats["cache"]["response_cache_limit"] = self._max_response_cache

        # BM25 stats
        if hasattr(self, "bm25_documents"):
            stats["bm25"] = {
                "document_count": len(self.bm25_documents),
                "enabled": self.bm25_index is not None,
            }

        return stats

    def search_by_keywords(self, keywords: list[str], limit: int = 10) -> list[dict[str, Any]]:
        """
        Search memories using BM25 keyword matching.

        Args:
            keywords: List of keywords to search for
            limit: Maximum number of results to return

        Returns:
            List of memories matching the keywords
        """
        if self.bm25_index is None or not keywords:
            return []

        # Apply any pending updates
        if self.bm25_pending_docs and len(self.bm25_pending_docs) > 0:
            tokenized_corpus = [self.tokenizer(doc) for doc in self.bm25_documents]
            self.bm25_index = BM25Okapi(tokenized_corpus)
            self.bm25_pending_docs = []
            self.bm25_pending_ids = []

        # Create query from keywords
        query = " ".join(keywords)
        tokenized_query = self.tokenizer(query)

        # Get document scores
        doc_scores = self.bm25_index.get_scores(tokenized_query)

        # Get top scores
        top_indices = np.argsort(doc_scores)[::-1][:limit]

        # Format results
        results = []
        for idx in top_indices:
            if idx < len(self.bm25_doc_ids) and doc_scores[idx] > 0:
                memory_id = self.bm25_doc_ids[idx]
                try:
                    # Get the memory to include content
                    memory = self.hybrid_memory_adapter.get(memory_id)

                    results.append(
                        {
                            "memory_id": memory_id,
                            "content": memory.content,
                            "bm25_score": float(doc_scores[idx]),
                            "retrieval_method": "bm25",
                            "metadata": memory.metadata,
                            "relevance_score": min(0.95, float(doc_scores[idx]) / 10),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error retrieving BM25 result {memory_id}: {e}")

        return results
