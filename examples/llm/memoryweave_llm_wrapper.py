"""
MemoryWeave integration with Hugging Face transformers for local LLM usage.

This module provides a simple wrapper to use MemoryWeave with locally running Hugging Face models.
"""

import time
import traceback
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from memoryweave.components.retriever import _get_embedder

DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_TOKENIZER: AutoTokenizer | None = None
_LLM: AutoModelForCausalLM | None = None
_DEVICE: str | None = None

from memoryweave.components.retrieval_strategies.contextual_fabric_strategy import ContextualFabricStrategy
from memoryweave.storage.memory_store import MemoryStore
from memoryweave.components.associative_linking import AssociativeMemoryLinker
from memoryweave.components.temporal_context import TemporalContextBuilder
from memoryweave.components.activation import ActivationManager

# Import the adapter we created
# from memory_store_adapter import MemoryStoreAdapter

# Define the MemoryStoreAdapter class inline if you can't import it
class MemoryStoreAdapter:
    """
    Adapter class to make MemoryStore compatible with ContextualFabricStrategy.
    """
    
    def __init__(self, memory_store):
        """Initialize the adapter with a MemoryStore instance."""
        self.memory_store = memory_store
        # Cache for embeddings matrix and metadata
        self._embeddings_matrix = None
        self._metadata_dict = None
        self._ids_list = None
    
    @property
    def memory_embeddings(self):
        """
        Property that returns all embeddings as a numpy matrix.
        """
        if self._embeddings_matrix is None:
            self._build_cache()
        return self._embeddings_matrix
    
    @property
    def memory_metadata(self):
        """
        Property that returns all metadata as a list.
        """
        if self._metadata_dict is None:
            self._build_cache()
        return self._metadata_dict
    
    def _build_cache(self):
        """Build cache of embeddings and metadata for efficient access."""
        # Get all memories
        memories = self.memory_store.get_all()
        
        # If there are no memories, set empty values
        if not memories:
            self._embeddings_matrix = np.zeros((0, 384))  # Empty array with embedding dimension
            self._metadata_dict = []
            self._ids_list = []
            return
        
        # Extract embeddings into a matrix
        embeddings = [memory.embedding for memory in memories]
        self._embeddings_matrix = np.stack(embeddings) if embeddings else np.array([])
        
        # Build metadata list in the same order
        self._metadata_dict = [memory.metadata for memory in memories]
        
        # Store IDs in the same order
        self._ids_list = [memory.id for memory in memories]
    
    def invalidate_cache(self):
        """Invalidate the cache when memories change."""
        self._embeddings_matrix = None
        self._metadata_dict = None
        self._ids_list = None
    
    def get(self, memory_id):
        """Get a memory by ID, same as the original memory store."""
        return self.memory_store.get(memory_id)
    
    def add(self, embedding, content, metadata=None):
        """Add a memory, same as the original memory store."""
        memory_id = self.memory_store.add(embedding, content, metadata)
        # Invalidate cache after adding
        self.invalidate_cache()
        return memory_id
    
    def get_all(self):
        """Get all memories, same as the original memory store."""
        return self.memory_store.get_all()
    
    def search_by_vector(self, query_vector, limit=10):
        """
        Search memories by vector similarity.
        """
        # Make sure cache is built
        if self._embeddings_matrix is None or len(self._embeddings_matrix) == 0:
            self._build_cache()
            # If still empty, return empty list
            if len(self._embeddings_matrix) == 0:
                return []
        
        # Calculate similarities
        similarities = np.dot(self._embeddings_matrix, query_vector)
        
        # Get indices of top results
        top_indices = np.argsort(-similarities)[:limit]
        
        # Create result objects
        results = []
        for idx in top_indices:
            memory_id = self._ids_list[idx]
            try:
                memory = self.memory_store.get(memory_id)
                # Add similarity score to result
                memory_dict = {
                    "id": memory.id,
                    "content": memory.content["text"] if isinstance(memory.content, dict) and "text" in memory.content else str(memory.content),
                    "metadata": memory.metadata,
                    "score": float(similarities[idx])
                }
                results.append(memory_dict)
            except Exception as e:
                print(f"Error retrieving memory {memory_id}: {e}")
        
        return results


def get_tokenizer(
    model_name: str = DEFAULT_MODEL,
    **kwargs,
) -> AutoTokenizer:
    """Gets the tokenizer as a singleton."""
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name, **kwargs)
    return _TOKENIZER


def get_llm(
    model_name: str = DEFAULT_MODEL,
    device: str = "mps",
    **kwargs,
) -> AutoModelForCausalLM:
    """Gets the LLM model as a singleton."""
    global _LLM
    if _LLM is None:
        if _DEVICE is None:
            _get_device(device=device)
        torch_dtype = torch.float16 if _DEVICE == "cuda" else torch.float32

        print(f"Loading LLM: {model_name}")
        _LLM = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            **kwargs,
        )
    return _LLM


def _get_device(device: str | None = _DEVICE) -> str:
    """Gets the device to use for the model."""
    device: str
    if torch.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device


class MemoryWeaveLLM:
    """A simple wrapper for using MemoryWeave with local Hugging Face models."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str = "mps",
        **model_kwargs,
    ):
        """
        Initialize the MemoryWeaveLLM wrapper.

        Args:
            model_name: Name of the Hugging Face model to use
            embedding_model_name: Name of the sentence transformer model for embeddings
            device: Device to run the model on ("cpu", "cuda", "auto")
        """
        # 1) Determine device
        self.device = _get_device(device=device)
        self.tokenizer = get_tokenizer(model_name)
        self.model = get_llm(model_name, device=self.device, **model_kwargs)
        # 2) Load or reuse the LLM + tokenizer singletons
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 3) Load the embedding model
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = _get_embedder(
            model_name=embedding_model_name,
            device=self.device,
        )

        # Setup MemoryWeave
        # 4) Initialize the memory store
        self.memory_store = MemoryStore()
        
        # Create adapter for compatibility with ContextualFabricStrategy
        self.memory_store_adapter = MemoryStoreAdapter(self.memory_store)

        # Initialize required components for the contextual fabric strategy
        self.associative_linker = AssociativeMemoryLinker(self.memory_store)
        self.temporal_context = TemporalContextBuilder(self.memory_store)
        self.activation_manager = ActivationManager(self.memory_store, self.associative_linker)

        # Initialize the contextual fabric strategy with the adapter
        self.strategy = ContextualFabricStrategy(
            memory_store=self.memory_store_adapter,  # Use the adapter here
            associative_linker=self.associative_linker,
            temporal_context=self.temporal_context,
            activation_manager=self.activation_manager
        )

        # Configure the strategy with weights for different aspects
        self.strategy.initialize({
            "confidence_threshold": 0.1,
            "similarity_weight": 0.5,
            "associative_weight": 0.3,
            "temporal_weight": 0.1,
            "activation_weight": 0.1,
            "max_associative_hops": 2,
            "debug": True  # Enable debug logging
        })

        # For backwards compatibility, keep memory_manager reference
        self.memory_manager = self.memory_store
        
        # 7) Keep local conversation state
        self.conversation_history = []

    def chat(self, user_message: str, max_new_tokens: int = 512) -> str:
        """
        Process a user message through MemoryWeave and the LLM.

        Args:
            user_message: The user's message
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            The assistant's response
        """
        # 1. Retrieve relevant memories using the ContextualFabricStrategy
        try:
            # Create embedding for the query
            query_embedding = self.embedding_model.encode(user_message, show_progress_bar=False)
            
            # Refresh the adapter's cache to ensure it has the latest data
            self.memory_store_adapter.invalidate_cache()
            
            # Use the retrieve method instead of process_query
            # Make sure we're passing the right parameters
            print("DEBUG: Calling strategy.retrieve")
            relevant_memories = self.strategy.retrieve(
                query_embedding=query_embedding,
                top_k=10,  # Retrieve more memories to improve recall
                context={
                    "query": user_message,
                    "current_time": time.time(),
                    "memory_store": self.memory_store_adapter
                }
            )
            
            # Convert strategy results to the format expected by the rest of the code
            formatted_memories = []
            for result in relevant_memories:
                try:
                    # Extract content and metadata
                    memory_id = result.get("memory_id")
                    if memory_id is not None:
                        # Get the full memory from the store
                        memory = self.memory_store.get(memory_id)
                        # Format it for the rest of the code
                        formatted_memory = {
                            "content": memory.content["text"] if isinstance(memory.content, dict) and "text" in memory.content else str(memory.content),
                            "metadata": memory.metadata,
                            "relevance_score": result.get("relevance_score", 0.5)
                        }
                        formatted_memories.append(formatted_memory)
                except Exception as memory_err:
                    print(f"Error retrieving memory {memory_id}: {memory_err}")
            
            relevant_memories = formatted_memories
            print(f"DEBUG: Retrieved {len(relevant_memories)} memories")
            
        except Exception as e:
            print(f"Error using contextual_fabric strategy: {e}")
            traceback.print_exc()
            # Fall back to simpler retrieval if needed
            relevant_memories = []
            
            # Try a direct memory search as fallback
            try:
                # Create embedding for the query
                query_embedding = self.embedding_model.encode(user_message, show_progress_bar=False)
                # Direct vector search using the adapter
                simple_results = self.memory_store_adapter.search_by_vector(query_embedding, limit=10)
                
                # Format results for consistency
                for item in simple_results:
                    memory = {
                        "content": item["content"] if "content" in item else str(item),
                        "metadata": item["metadata"] if "metadata" in item else {},
                        "relevance_score": item["score"] if "score" in item else 0.5
                    }
                    relevant_memories.append(memory)
                
                print(f"DEBUG: Retrieved {len(relevant_memories)} memories using fallback")
            except Exception as fallback_error:
                print(f"Even fallback retrieval failed: {fallback_error}")
                traceback.print_exc()

        # 2. Format memories for the prompt in a more structured way
        memory_text = ""
        facts_about_user = []
        preferences = []

        # Sort memories by relevance score to prioritize the most relevant ones
        relevant_memories.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        # Categorize and extract structured information from memories
        for memory in relevant_memories:
            if not memory or not isinstance(memory, dict):
                continue

            memory_type = memory.get("metadata", {}).get("type", "")
            content = memory.get("content", "")
            score = memory.get("relevance_score", 0)

            # Skip lower quality memories (only use if score is reasonable)
            if score < 0.1:  # Skip very low relevance memories
                continue

            # Skip conversation memories to reduce noise
            if memory_type in ["user_message", "assistant_message"]:
                continue

            # Categorize memories by type
            if (
                memory_type == "personal_info"
                or memory_type == "location"
                or memory_type == "pet_name"
            ):
                # Avoid duplicates
                if content not in facts_about_user:
                    facts_about_user.append(content)
            elif memory_type == "preference":
                # Avoid duplicates
                if content not in preferences:
                    preferences.append(content)

        # Format user information in a clean, structured way
        if facts_about_user or preferences:
            memory_text = "USER INFORMATION:\n"

        for fact in facts_about_user:
            # Clean up fact text if it starts with "User..."
            if fact.startswith("User "):
                parts = fact.split(": ", 1)
                if len(parts) > 1:
                    fact = parts[1]
            memory_text += f"- {fact}\n"

        if preferences:
            if facts_about_user:  # Add a separator if we had facts
                memory_text += "\n"
            memory_text += "USER PREFERENCES:\n"
            for pref in preferences:
                # Clean up preference text
                if pref.startswith("User preference:"):
                    pref = pref.replace("User preference:", "").strip()
                memory_text += f"- {pref}\n"

        # Add stronger instruction to use this information
        if memory_text:
            memory_text += "\nIMPORTANT: Use ALL of this information to personalize your responses. Don't explicitly state you're using stored information, but DON'T claim you don't know this information. Incorporate it naturally in your answers.\n\n"

        # 3. Prepare prompt with conversation history and memories
        system_prompt = "You are a helpful assistant. Provide accurate, relevant responses based on the conversation context."

        if memory_text:
            system_prompt += f"\n\n{memory_text}"
            system_prompt += "\nCRITICAL INSTRUCTION: When asked ANY questions about the user (their name, location, preferences, pets, hobbies, food likes, etc.), you MUST use ONLY the information provided above to answer. NEVER claim you don't know this information or that you can't access it. If the information IS provided above, use it with confidence. If it's NOT provided above, only then should you say you don't have that specific information."

        # Format history for the model
        history_text = ""
        for msg in self.conversation_history[-5:]:  # Last 5 exchanges only
            if msg["role"] == "user":
                history_text += f"User: {msg['content']}\n"
            else:
                history_text += f"Assistant: {msg['content']}\n"

        # Create the final prompt
        if history_text:
            prompt = f"{system_prompt}\n\n{history_text}User: {user_message}\nAssistant:"
        else:
            prompt = f"{system_prompt}\n\nUser: {user_message}\nAssistant:"

        # 4. Generate response from the LLM
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )

        # Decode the generated text, skipping the input prompt
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_message = full_response[len(prompt):]

        # Clean up the response to only include the assistant's part
        assistant_message = assistant_message.strip()

        # 5. Store conversation in history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_message})

        # 6. Store the interaction in MemoryWeave for future retrieval
        self._store_interaction(user_message, assistant_message)

        return assistant_message

    def _store_interaction(self, user_message: str, assistant_message: str):
        """Store the conversation turn in MemoryWeave."""
        # Store the user's message with proper analysis for better retrieval later

        # First, try to extract personal information from user messages
        # Simple pattern matching for common personal information patterns
        import re

        # Patterns for common personal info - could be expanded with more robust NLP
        location_pattern = (
            r"(?:I live|I'm from|my home|my city|my location|I reside) (?:is |in )?([\w\s,]+)"
        )
        pet_pattern = r"(?:my pet|my dog|my cat|my animal)(?:'s| is)? (?:called |named )?([\w\s]+)"
        preference_pattern = r"(?:I (?:like|love|enjoy|prefer)|my favorite) ([\w\s,]+)"

        # Store the user's message
        user_embedding = self.embedding_model.encode(user_message, show_progress_bar=False)
        self.memory_store.add(
            user_embedding,
            user_message,
            {
                "type": "user_message",
                "timestamp": time.time(),
                "conversation_id": id(self.conversation_history),
                "importance": 0.7,  # Higher importance for user messages
            },
        )
        
        # Invalidate the adapter's cache
        self.memory_store_adapter.invalidate_cache()

        # Extract potential personal information from the message
        # Location detection
        location_match = re.search(location_pattern, user_message, re.IGNORECASE)
        if location_match:
            location = location_match.group(1).strip()
            location_text = f"User lives in {location}."
            location_embedding = self.embedding_model.encode(location_text)
            self.memory_store.add(
                location_embedding,
                location_text,
                {
                    "type": "personal_info",
                    "subtype": "location",
                    "timestamp": time.time(),
                    "importance": 0.9,  # Personal info is very important
                },
            )
            print(f"DEBUG: Extracted location: {location}")
            self.memory_store_adapter.invalidate_cache()

        # Pet detection
        pet_match = re.search(pet_pattern, user_message, re.IGNORECASE)
        if pet_match:
            pet_name = pet_match.group(1).strip()
            pet_text = f"User has a pet named {pet_name}."
            pet_embedding = self.embedding_model.encode(pet_text)
            self.memory_store.add(
                pet_embedding,
                pet_text,
                {
                    "type": "pet_name",
                    "timestamp": time.time(),
                    "importance": 0.8,
                },
            )
            print(f"DEBUG: Extracted pet name: {pet_name}")
            self.memory_store_adapter.invalidate_cache()

        # Preference detection
        preference_match = re.search(preference_pattern, user_message, re.IGNORECASE)
        if preference_match:
            preference = preference_match.group(1).strip()
            preference_text = f"User preference: {preference}"
            preference_embedding = self.embedding_model.encode(preference_text)
            self.memory_store.add(
                preference_embedding,
                preference_text,
                {
                    "type": "preference",
                    "timestamp": time.time(),
                    "importance": 0.75,
                },
            )
            print(f"DEBUG: Extracted preference: {preference}")
            self.memory_store_adapter.invalidate_cache()

        # Store the assistant's response
        assistant_embedding = self.embedding_model.encode(assistant_message)
        self.memory_store.add(
            assistant_embedding,
            assistant_message,
            {
                "type": "assistant_message",
                "timestamp": time.time(),
                "conversation_id": id(self.conversation_history),
                "importance": 0.5,  # Lower importance than user messages
            },
        )
        self.memory_store_adapter.invalidate_cache()

    def add_memory(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ):
        """Add a memory directly to the memory store."""
        if metadata is None:
            metadata = dict(type="fact", importance=0.7)

        # Get embedding for the text
        embedding = self.embedding_model.encode(text)

        # Use the memory_store.add method
        memory_id = self.memory_store.add(embedding, text, metadata)
        # Invalidate the adapter's cache
        self.memory_store_adapter.invalidate_cache()
        
        print(f"DEBUG: Added memory with ID {memory_id}: {text}")

        return memory_id

    def get_conversation_history(self):
        """Get the current conversation history."""
        return self.conversation_history

    def chat_without_memory(self, user_message: str, max_new_tokens: int = 512) -> str:
        """
        Process a user message through the LLM without using MemoryWeave.
        This provides a baseline for comparison.

        Args:
            user_message: The user's message
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            The assistant's response
        """
        # Create a simple prompt without memory augmentation
        prompt = f"You are a helpful assistant.\n\nUser: {user_message}\nAssistant:"

        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )

        # Decode the generated text, skipping the input prompt
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_message = full_response[len(prompt):]

        # Clean up the response
        assistant_message = assistant_message.strip()

        return assistant_message