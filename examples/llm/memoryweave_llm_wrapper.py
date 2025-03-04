"""
MemoryWeave integration with Hugging Face transformers for local LLM usage.

This module provides a simple wrapper to use MemoryWeave with locally running Hugging Face models.
"""

import time
from typing import Any

import torch
from memoryweave.components import Retriever
from memoryweave.components.memory_manager import MemoryManager
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

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

        # First initialize components
        self.retriever.initialize_components()

        # Then explicitly configure each feature for better control
        self.retriever.configure_query_type_adaptation(enable=True, adaptation_strength=1.0)
        self.retriever.configure_semantic_coherence(enable=True)
        self.retriever.configure_two_stage_retrieval(
            enable=True, first_stage_k=30, first_stage_threshold_factor=0.5
        )
        self.retriever.enable_dynamic_threshold_adjustment(enable=True, window_size=3)

        # Set the default retrieval strategy to TwoStage
        self.retriever.retrieval_strategy = self.retriever.memory_manager.components.get(
            "two_stage_retrieval"
        )

        # Force rebuild the pipeline to ensure all settings take effect
        self.retriever._build_default_pipeline()

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

        # Use the ContextualFabricStrategy for advanced memory retrieval
        try:
            relevant_memories = self.retriever.retrieve(
                user_message,
                top_k=10,  # Retrieve more memories to improve recall
                strategy="contextual_fabric",  # Use the contextual fabric strategy
                conversation_history=self.conversation_history,
            )
        except Exception as e:
            print(f"Error using contextual_fabric strategy: {e}")
            # Fall back to two_stage if contextual_fabric fails
            relevant_memories = self.retriever.retrieve(
                user_message,
                top_k=10,
                strategy="two_stage",  # Fallback to two-stage retrieval strategy
                conversation_history=self.conversation_history,
            )
        print(f"DEBUG: Retrieved {len(relevant_memories)} memories")

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
        user_embedding = self.embedding_model.encode(user_message)
        self.memory_manager.memory_store.add(
            user_embedding,
            user_message,
            {
                "type": "user_message",
                "timestamp": time.time(),
                "conversation_id": id(self.conversation_history),
                "importance": 0.7,  # Higher importance for user messages
            },
        )

        # Extract potential personal information from the message
        # Location detection
        location_match = re.search(location_pattern, user_message, re.IGNORECASE)
        if location_match:
            location = location_match.group(1).strip()
            location_text = f"User lives in {location}."
            location_embedding = self.embedding_model.encode(location_text)
            self.memory_manager.memory_store.add(
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

        # Pet detection
        pet_match = re.search(pet_pattern, user_message, re.IGNORECASE)
        if pet_match:
            pet_name = pet_match.group(1).strip()
            pet_text = f"User has a pet named {pet_name}."
            pet_embedding = self.embedding_model.encode(pet_text)
            self.memory_manager.memory_store.add(
                pet_embedding,
                pet_text,
                {
                    "type": "pet_name",
                    "timestamp": time.time(),
                    "importance": 0.8,
                },
            )
            print(f"DEBUG: Extracted pet name: {pet_name}")

        # Preference detection
        preference_match = re.search(preference_pattern, user_message, re.IGNORECASE)
        if preference_match:
            preference = preference_match.group(1).strip()
            preference_text = f"User preference: {preference}"
            preference_embedding = self.embedding_model.encode(preference_text)
            self.memory_manager.memory_store.add(
                preference_embedding,
                preference_text,
                {
                    "type": "preference",
                    "timestamp": time.time(),
                    "importance": 0.75,
                },
            )
            print(f"DEBUG: Extracted preference: {preference}")

        # Store the assistant's response
        assistant_embedding = self.embedding_model.encode(assistant_message)
        self.memory_manager.memory_store.add(
            assistant_embedding,
            assistant_message,
            {
                "type": "assistant_message",
                "timestamp": time.time(),
                "conversation_id": id(self.conversation_history),
                "importance": 0.5,  # Lower importance than user messages
            },
        )

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

        # Use the memory_store.add method from MemoryStore
        memory_id = self.memory_manager.memory_store.add(embedding, text, metadata)
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
        assistant_message = full_response[len(prompt) :]

        # Clean up the response
        assistant_message = assistant_message.strip()

        return assistant_message
