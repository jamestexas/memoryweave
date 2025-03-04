import logging
import time
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memoryweave.components.activation import ActivationManager
from memoryweave.components.associative_linking import AssociativeMemoryLinker
from memoryweave.components.retrieval_strategies.contextual_fabric_strategy import (
    ContextualFabricStrategy,
)
from memoryweave.components.retriever import _get_embedder
from memoryweave.components.temporal_context import TemporalContextBuilder
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


class MemoryStoreAdapter:
    """
    Adapter class to make MemoryStore compatible with ContextualFabricStrategy.
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
            print("DEBUG: Building memory adapter cache")
            memories = self.memory_store.get_all()

            if not memories:
                self._embeddings_matrix = np.zeros((0, 384))
                self._metadata_dict = []
                self._ids_list = []
                self._index_to_id_map = {}
                print("DEBUG: No memories found")
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
                    print(f"DEBUG: Error processing memory {memory.id}: {e}")

            self._embeddings_matrix = np.stack(embeddings) if embeddings else np.zeros((0, 384))
            self._metadata_dict = metadata_list
            self._ids_list = ids_list

            print(f"DEBUG: Built cache with {len(embeddings)} memories")
            print(f"DEBUG: ID map has {len(self._index_to_id_map)} entries")

        except Exception as e:
            print(f"ERROR in _build_cache: {e}")
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
                print(f"DEBUG: Translated index {idx} to ID {actual_id}")
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

    def search_by_vector(self, query_vector: np.ndarray, limit: int = 10) -> list[dict]:
        """Simple direct vector similarity search, returning top results."""
        if self._embeddings_matrix is None or len(self._embeddings_matrix) == 0:
            self._build_cache()
            if len(self._embeddings_matrix) == 0:
                return []

        similarities = np.dot(self._embeddings_matrix, query_vector)
        top_indices = np.argsort(-similarities)[:limit]

        results = []
        for idx in top_indices:
            memory_id = self._ids_list[idx]
            try:
                memory = self.memory_store.get(memory_id)
                results.append({
                    "id": memory.id,
                    "content": str(memory.content),
                    "metadata": memory.metadata,
                    "score": float(similarities[idx]),
                })
            except Exception as e:
                print(f"Error retrieving memory {memory_id}: {e}")

        return results


class MemoryWeaveAPI:
    """
    A unified API for MemoryWeave, managing memory and retrieval in a simple interface.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str = "auto",
        show_progress_bar: bool = False,
        **model_kwargs,
    ):
        """Initialize MemoryWeave with an LLM, embeddings, and memory components."""
        self.device = self._get_device(device)
        self.show_progress_bar = show_progress_bar
        # Load LLM & tokenizer
        self.tokenizer = get_tokenizer(model_name)
        if "show_progress_bar" in model_kwargs:
            model_kwargs.pop("show_progress_bar")
        self.model = get_llm(model_name, device=self.device, **model_kwargs)

        # 2) Embedding model for memory
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = _get_embedder(model_name=embedding_model_name, device=self.device)

        # Initialize Memory Store & Components
        self.memory_store = MemoryStore()
        self.memory_store_adapter = MemoryStoreAdapter(self.memory_store)
        self.associative_linker = AssociativeMemoryLinker(self.memory_store)
        self.temporal_context = TemporalContextBuilder(self.memory_store)
        self.activation_manager = ActivationManager(self.memory_store, self.associative_linker)

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
            "debug": False,
        })

        # Track conversation history
        self.conversation_history = []

    def _get_device(self, device: str) -> str:
        """Determine the device for running the model."""
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.mps.is_available():
            return "mps"
        return "cpu"

    def add_memory(self, text: str, metadata: dict[str, Any] = None):
        """Store a memory in the system."""
        logger.debug(f"Adding memory: {text}")
        embedding = self.embedding_model.encode(text, show_progress_bar=self.show_progress_bar)
        mem_id = self.memory_store.add(embedding, text, metadata or {"type": "manual"})
        self.memory_store_adapter.invalidate_cache()
        logger.debug(f"DEBUG: Added memory {mem_id}: {text}")
        return mem_id

    def chat(self, user_message: str, max_new_tokens: int = 512) -> str:
        """
        Process user input, retrieve memory context, and generate a response efficiently.
        """

        now = time.time()

        # ✅ Step 1: Compute query embedding once
        try:
            query_embedding = self.embedding_model.encode(
                user_message, show_progress_bar=self.show_progress_bar
            )
        except Exception as encode_err:
            print(f"Error generating query embedding: {encode_err}")
            import traceback

            traceback.print_exc()
            return "Sorry, an error occurred while processing your request."

        # ✅ Step 2: Retrieve relevant memories efficiently
        self.memory_store_adapter.invalidate_cache()
        try:
            print("DEBUG: Calling strategy.retrieve")
            relevant_memories = self.strategy.retrieve(
                query_embedding=query_embedding,
                top_k=10,
                context={
                    "query": user_message,
                    "current_time": now,
                    "memory_store": self.memory_store_adapter,
                },
            )
        except Exception as main_err:
            print(f"Error using contextual_fabric strategy: {main_err}")
            traceback.print_exc()
            relevant_memories = []

        # Fallback if retrieval is empty
        if not relevant_memories:
            try:
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
                print(f"DEBUG: Fallback retrieved {len(relevant_memories)} memories")
            except Exception as fallback_err:
                print(f"Even fallback retrieval failed: {fallback_err}")
                traceback.print_exc()
                relevant_memories = []

        # ✅ Step 3: Update memory activations immediately for retrieved memories
        for mem_dict in relevant_memories:
            mem_id = mem_dict.get("memory_id")
            if mem_id is not None:
                self.activation_manager.activate_memory(mem_id, 0.2, spread=True)
                self.temporal_context.update_timestamp(mem_id, now)

        # 4) Build minimal dictionaries for apply_temporal_context, so it can do recency or date-based boosts
        base_context = {"current_time": now}
        memory_dicts_for_temporal = []
        for r in relevant_memories:
            mem_id = r.get("memory_id")
            base_score = r.get("relevance_score", 0.5)

            # If "created_at" is stored in metadata, retrieve it:
            created_at = 0.0
            meta = r.get("metadata", {})
            if isinstance(meta, dict) and "created_at" in meta:
                created_at = meta["created_at"]

            memory_dicts_for_temporal.append({
                "memory_id": mem_id,
                "created_at": created_at,
                "relevance_score": base_score,
            })

        # 5) Call apply_temporal_context(...) with the user query
        memory_dicts_for_temporal = self.temporal_context.apply_temporal_context(
            query=user_message, results=memory_dicts_for_temporal, context=base_context
        )

        # Sort by updated "relevance_score"
        memory_dicts_for_temporal.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)

        # 6) Convert them back to "cleaned_results" for the prompt
        cleaned_results = []
        for md in memory_dicts_for_temporal:
            mem_id = md.get("memory_id")
            if mem_id is None:
                continue

            if isinstance(mem_id, int):
                try:
                    memory_obj = self.memory_store.get(str(mem_id))
                    if memory_obj:
                        cleaned_results.append({
                            "content": str(memory_obj.content),
                            "metadata": memory_obj.metadata,
                            "relevance_score": md.get("relevance_score", 0.5),
                        })
                    else:
                        logger.info(f"No real ID found for index {mem_id}, skipping")

                except KeyError:
                    logger.info(f"No real ID found for index {mem_id}, skipping")

            else:  # It's presumably the real ID
                try:
                    memory_obj = self.memory_store.get(mem_id)
                    if memory_obj:
                        cleaned_results.append({
                            "content": str(memory_obj.content),
                            "metadata": memory_obj.metadata,
                            "relevance_score": md.get("relevance_score", 0.5),
                        })
                    else:
                        logger.info(f"No memory with ID {mem_id}, skipping")
                except KeyError:
                    logger.info(f"No memory with ID {mem_id}, skipping")

        relevant_memories = cleaned_results
        print(f"DEBUG: After temporal context, we have {len(relevant_memories)} memories")

        # ✅ Step 4: Construct system prompt with top retrieved memories
        system_prompt = (
            "You are a helpful assistant. Use the entire conversation context for your answer.\n"
            "Do not disclaim that you have a memory system. If asked about user info, see if it "
            "is in your retrieved content. If yes, incorporate it naturally. If not, just say you "
            "lack that info.\n"
        )

        # Sort again by final relevance
        relevant_memories.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        # Include top 3 in the prompt
        top_memories = relevant_memories[:3]
        memory_text = ""
        if top_memories:
            memory_text = "MEMORY HIGHLIGHTS:\n"
            for m in top_memories:
                c = m["content"]
                memory_text += f"- {c[:150]}...\n"
            memory_text += "\n"

        final_system_prompt = system_prompt + memory_text

        # 8) Add the last 5 conversation turns
        history_text = ""
        for turn in self.conversation_history[-5:]:
            if turn["role"] == "user":
                history_text += f"User: {turn['content']}\n"
            else:
                history_text += f"Assistant: {turn['content']}\n"

        if history_text:
            prompt = f"{final_system_prompt}\n{history_text}User: {user_message}\nAssistant:"
        else:
            prompt = f"{final_system_prompt}\nUser: {user_message}\nAssistant:"

        # ✅ Step 5: Generate response using LLM
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

        # ✅ Step 6: Store the interaction in memory
        user_emb = self.embedding_model.encode(
            user_message, show_progress_bar=self.show_progress_bar
        )
        assistant_emb = self.embedding_model.encode(
            assistant_reply, show_progress_bar=self.show_progress_bar
        )

        self.memory_store.add(
            user_emb,
            user_message,
            {
                "type": "user_message",
                "timestamp": now,
                "conversation_id": id(self.conversation_history),
                "importance": 0.7,
            },
        )
        self.memory_store_adapter.invalidate_cache()
        self.memory_store.add(
            assistant_emb,
            assistant_reply,
            {
                "type": "assistant_message",
                "timestamp": now,
                "conversation_id": id(self.conversation_history),
                "importance": 0.5,
            },
        )
        self.memory_store_adapter.invalidate_cache()

        # ✅ Step 7: Maintain conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_reply})

        return assistant_reply

    def get_conversation_history(self) -> list[dict[str, str]]:
        """Return stored conversation history."""
        return self.conversation_history

    def retrieve(self, query, **kwargs):
        """Retrieve memories using the strategy."""
        query_embedding = self.embedding_model.encode(
            query, show_progress_bar=self.show_progress_bar
        )
        return self.strategy.retrieve(query_embedding, **kwargs)

    def chat_without_memory(self, user_message: str, max_new_tokens: int = 512) -> str:
        """
        A baseline method that does not use memory retrieval at all.
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
            show_progress_bar=False,
        )
        assistant_response = full_response[len(prompt) :].strip()
        return assistant_response
