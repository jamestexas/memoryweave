"""
MemoryWeave integration with Hugging Face transformers for local LLM usage.

This module provides a simple wrapper to use MemoryWeave with locally running Hugging Face models.
"""

import time
from typing import Any

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from memoryweave.components import Retriever
from memoryweave.components.memory_manager import MemoryManager

DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class MemoryWeaveLLM:
    """A simple wrapper for using MemoryWeave with local Hugging Face models."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str = "mps",
    ):
        """
        Initialize the MemoryWeaveLLM wrapper.

        Args:
            model_name: Name of the Hugging Face model to use
            embedding_model_name: Name of the sentence transformer model for embeddings
            device: Device to run the model on ("cpu", "cuda", "auto")
        """
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "mps"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Initialize LLM
        print(f"Loading LLM: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
        )

        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name, device=self.device)

        # Setup MemoryWeave
        self.memory_manager = MemoryManager()
        self.retriever = Retriever(memory=self.memory_manager, embedding_model=self.embedding_model)

        # Configure the retriever for balanced performance
        self.retriever.configure_query_type_adaptation(enable=True)
        self.retriever.configure_semantic_coherence(enable=True)
        self.retriever.configure_two_stage_retrieval(enable=True)
        self.retriever.initialize_components()

        # Track conversation for context
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
        # 1. Retrieve relevant memories
        try:
            relevant_memories = self.retriever.retrieve(user_message, top_k=3)
        except Exception as e:
            print(f"Retrieval error: {e}")
            relevant_memories = []

        # 2. Format memories for the prompt
        memory_text = ""
        if relevant_memories:
            memory_text = "Previous information that might be relevant:\n"
            for memory in relevant_memories:
                if isinstance(memory, dict) and "content" in memory:
                    content = memory.get("content", "")
                    if isinstance(content, dict) and "text" in content:
                        memory_text += f"- {content['text']}\n"
                    else:
                        memory_text += f"- {content}\n"
                else:
                    memory_text += f"- {str(memory)}\n"

        # 3. Prepare prompt with conversation history and memories
        system_prompt = "You are a helpful assistant with memory capabilities."

        if memory_text:
            system_prompt += f"\n\n{memory_text}"

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
        assistant_message = full_response[len(prompt) :]

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
        # Store the user's message
        user_embedding = self.embedding_model.encode(user_message)
        self.memory_manager.memory_store.add(
            user_embedding,
            user_message,
            {
                "type": "user_message",
                "timestamp": time.time(),
                "conversation_id": id(self.conversation_history),
            },
        )

        # Store the assistant's response
        assistant_embedding = self.embedding_model.encode(assistant_message)
        self.memory_manager.memory_store.add(
            assistant_embedding,
            assistant_message,
            {
                "type": "assistant_message",
                "timestamp": time.time(),
                "conversation_id": id(self.conversation_history),
            },
        )

        # Extract and store any personal information or preferences
        # This could be enhanced with a more sophisticated extraction mechanism
        if "my favorite" in user_message.lower() or "i like" in user_message.lower():
            preference_embedding = self.embedding_model.encode(f"User preference: {user_message}")
            self.memory_manager.memory_store.add(
                embedding=preference_embedding,
                content=f"User preference: {user_message}",
                metadata=dict(
                    type="preference",
                    timestamp=time.time(),
                    importance=0.8,
                ),
            )

    def add_memory(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ):
        """Add a memory directly to the memory store."""
        if metadata is None:
            metadata = {"type": "fact", "importance": 0.7}

        # Get embedding for the text
        embedding = self.embedding_model.encode(text)

        # Use the memory_store.add method from MemoryStore
        self.memory_manager.memory_store.add(embedding, text, metadata)

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
        assistant_message = full_response[len(prompt) :]

        # Clean up the response
        assistant_message = assistant_message.strip()

        return assistant_message
