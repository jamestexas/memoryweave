import logging
from typing import Any

import numpy as np
import torch
from rich.logging import RichHandler
from transformers import AutoModelForCausalLM, AutoTokenizer

from memoryweave.storage.memory_store import MemoryStore

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_TOKENIZER: AutoTokenizer | None = None
_LLM: AutoModelForCausalLM | None = None
_DEVICE: str | None = None


def get_device(device: str | None = None) -> str:
    """Choose a device for running the model."""
    if device is None or device == "auto":
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
        _DEVICE = get_device(device)
        torch_dtype = torch.float16 if _DEVICE == "cuda" else torch.float32

        print(f"Loading LLM: {model_name}")
        _LLM = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=_DEVICE,
            **kwargs,
        )
    return _LLM


# Add to memory_store.py
class UnifiedMemoryAdapter:
    """Base interface for all memory adapters"""

    def invalidate_cache(self):
        """Clear any cached data"""
        pass

    def get_memory_embeddings(self):
        """Get all memory embeddings as a matrix"""
        pass

    def search_by_vector(self, query_vector, limit=10, threshold=None):
        """Search memories by vector similarity"""
        pass

    # Add specialized methods
    def search_chunks(self, query_vector, limit=10, threshold=None):
        """Search for matching chunks (for chunked memories)"""
        return []

    def search_hybrid(self, query_vector, limit=10, threshold=None, keywords=None):
        """Search using hybrid approach (for hybrid memories)"""
        return []


class MemoryStoreAdapter:
    """
    Adapter class to make MemoryStore compatible with retrieval strategies.
    This helps the strategy do similarity computations and retrieve embeddings.
    """

    def __init__(self, memory_store: MemoryStore):
        """Initialize the adapter with a MemoryStore instance."""
        self.memory_store = memory_store
        self._embeddings_matrix = None
        self._metadata_dict = None
        self._ids_list = None
        self._index_to_id_map = {}

    @property
    def memory_embeddings(self) -> np.ndarray:
        """Return all embeddings as a NumPy matrix for similarity computations."""
        if self._embeddings_matrix is None:
            self._build_cache()
        return self._embeddings_matrix

    @property
    def memory_metadata(self) -> list[dict[str, Any]]:
        """Return metadata for each memory, in the same order as memory_embeddings."""
        if self._metadata_dict is None:
            self._build_cache()
        return self._metadata_dict

    def _build_cache(self):
        """Build an internal cache of embeddings and metadata for fast retrieval."""
        try:
            logger.debug("Building memory adapter cache")
            memories = self.memory_store.get_all()

            if not memories:
                self._embeddings_matrix = np.zeros((0, 384))
                self._metadata_dict = []
                self._ids_list = []
                self._index_to_id_map = {}
                logger.debug("No memories found")
                return

            embeddings = []
            metadata_list = []
            ids_list = []
            self._index_to_id_map = {}

            for idx, memory in enumerate(memories):
                try:
                    embeddings.append(memory.embedding)

                    # Ensure metadata is a dictionary
                    if memory.metadata is not None and isinstance(memory.metadata, dict):
                        mdata = memory.metadata.copy()
                    else:
                        mdata = {}

                    mdata["memory_id"] = idx
                    mdata["original_id"] = memory.id
                    metadata_list.append(mdata)

                    ids_list.append(memory.id)
                    self._index_to_id_map[idx] = memory.id
                except Exception as e:
                    logger.debug(f"Error processing memory {memory.id}: {e}")

            self._embeddings_matrix = np.stack(embeddings) if embeddings else np.zeros((0, 384))
            self._metadata_dict = metadata_list
            self._ids_list = ids_list

            logger.debug(f"Built cache with {len(embeddings)} memories")
            logger.debug(f"ID map has {len(self._index_to_id_map)} entries")

        except Exception as e:
            logger.error(f"Error in _build_cache: {e}")
            import traceback

            traceback.print_exc()
            self._embeddings_matrix = np.zeros((0, 384))
            self._metadata_dict = []
            self._ids_list = []
            self._index_to_id_map = {}

    def invalidate_cache(self):
        """Invalidate the cache after new memories are added."""
        self._embeddings_matrix = None
        self._metadata_dict = None
        self._ids_list = None
        self._index_to_id_map = {}

    def get(self, memory_id: str | int):
        """
        Retrieve a single memory by ID, translating integer indexes to real memory IDs if needed.
        """
        if isinstance(memory_id, int) or (isinstance(memory_id, str) and memory_id.isdigit()):
            idx = int(memory_id)
            if idx in self._index_to_id_map:
                actual_id = self._index_to_id_map[idx]
                logger.debug(f"Translated index {idx} to ID {actual_id}")
                return self.memory_store.get(actual_id)
            # fallback
            return self.memory_store.get(str(memory_id))
        else:
            return self.memory_store.get(memory_id)

    def add(self, embedding: np.ndarray, content: Any, metadata: dict[str, Any] | None = None):
        """Add a new memory, then invalidate our cache."""
        mem_id = self.memory_store.add(embedding, content, metadata)
        self.invalidate_cache()
        return mem_id

    def get_all(self):
        """Forward to the underlying memory store."""
        return self.memory_store.get_all()

    def search_by_vector(
        self, query_vector: np.ndarray, limit: int = 10, threshold: float | None = None
    ) -> list[dict]:
        """
        Vector similarity search with optional threshold filtering.
        """
        if self._embeddings_matrix is None or len(self._embeddings_matrix) == 0:
            self._build_cache()
            if len(self._embeddings_matrix) == 0:
                return []

        similarities = np.dot(self._embeddings_matrix, query_vector)
        top_indices = np.argsort(-similarities)[:limit]

        results = []
        for idx in top_indices:
            memory_id = self._ids_list[idx]
            similarity_score = float(similarities[idx])

            # Apply threshold if provided
            if threshold is not None and similarity_score < threshold:
                continue  # Skip results below the threshold

            try:
                memory = self.memory_store.get(memory_id)
                results.append(
                    {
                        "id": memory.id,
                        "content": str(memory.content),
                        "metadata": memory.metadata,
                        "score": similarity_score,
                    }
                )
            except Exception as e:
                logger.error(f"Error retrieving memory {memory_id}: {e}")

        return results

    # Add these methods to support the specialized memory types
    def search_chunks(
        self, query_vector: np.ndarray, limit: int = 10, threshold: float | None = None
    ) -> list[dict]:
        """
        Default implementation for chunk search - will be overridden by specialized adapters.
        """
        # Default implementation just returns []
        return []

    def search_hybrid(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        threshold: float | None = None,
        keywords: list[str] | None = None,
    ) -> list[dict]:
        """
        Default implementation for hybrid search - will be overridden by specialized adapters.
        """
        # Default implementation just calls search_by_vector
        return self.search_by_vector(query_vector, limit, threshold)
