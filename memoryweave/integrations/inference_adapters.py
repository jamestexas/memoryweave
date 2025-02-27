"""
Adapters for integrating MemoryWeave with popular LLM frameworks.
"""

from typing import Any, Callable, Optional


class BaseAdapter:
    """Base adapter class for LLM integration."""

    def __init__(
        self, memory_system: dict[str, Any], format_memories_fn: Optional[Callable] = None
    ):
        """
        Initialize the adapter.

        Args:
            memory_system: dictionary containing memory components
            format_memories_fn: Function to format memories for prompt inclusion
        """
        self.memory = memory_system.get("memory")
        self.encoder = memory_system.get("encoder")
        self.retriever = memory_system.get("retriever")

        if not all([self.memory, self.encoder, self.retriever]):
            raise ValueError("Memory system must contain memory, encoder, and retriever")

        self.format_memories_fn = format_memories_fn or self._default_format_memories

    def _default_format_memories(self, memories: list[dict]) -> str:
        """Default memory formatting for prompt augmentation."""
        if not memories:
            return ""

        formatted = []
        for mem in memories:
            if mem.get("type") == "interaction":
                formatted.append(
                    f"Previous conversation: {mem.get('speaker', 'User')}: {mem.get('content', '')}"
                )
                if mem.get("response"):
                    formatted.append(f"Response: {mem.get('response')}")
            elif mem.get("type") == "concept":
                formatted.append(f"Concept '{mem.get('name', '')}': {mem.get('description', '')}")

        return "\n".join(formatted)

    def store_interaction(
        self, user_input: str, response: str, metadata: Optional[dict] = None
    ) -> None:
        """
        Store an interaction in memory.

        Args:
            user_input: User's input text
            response: Model's response
            metadata: Additional metadata
        """
        if not metadata:
            metadata = {}

        embedding, interaction_metadata = self.encoder.encode_interaction(
            message=user_input, speaker="user", response=response, additional_context=metadata
        )

        self.memory.add_memory(embedding, user_input, interaction_metadata)

    def retrieve_memories(
        self, user_input: str, conversation_history: list[dict], top_k: int = 5
    ) -> list[dict]:
        """
        Retrieve relevant memories for the current context.

        Args:
            user_input: Current user input
            conversation_history: list of previous conversation turns
            top_k: Number of memories to retrieve

        Returns:
            list of relevant memory entries
        """
        return self.retriever.retrieve_for_context(user_input, conversation_history, top_k=top_k)


class HuggingFaceAdapter(BaseAdapter):
    """Adapter for Hugging Face Transformers."""

    def __init__(
        self,
        memory_system: dict[str, Any],
        model,
        tokenizer,
        format_memories_fn: Optional[Callable] = None,
        include_memories_as_prefix: bool = True,
    ):
        """
        Initialize the Hugging Face adapter.

        Args:
            memory_system: dictionary containing memory components
            model: Hugging Face model
            tokenizer: Hugging Face tokenizer
            format_memories_fn: Function to format memories for prompt inclusion
            include_memories_as_prefix: Whether to include memories as prefix
        """
        super().__init__(memory_system, format_memories_fn)
        self.model = model
        self.tokenizer = tokenizer
        self.include_memories_as_prefix = include_memories_as_prefix

    def generate(
        self,
        user_input: str,
        conversation_history: list[dict],
        generation_kwargs: Optional[dict] = None,
        store_in_memory: bool = True,
    ) -> str:
        """
        Generate a response using the model with memory augmentation.

        Args:
            user_input: User's input text
            conversation_history: Previous conversation turns
            generation_kwargs: Additional kwargs for model.generate()
            store_in_memory: Whether to store the interaction in memory

        Returns:
            Model's generated response
        """
        if generation_kwargs is None:
            generation_kwargs = {}

        # Retrieve relevant memories
        memories = self.retrieve_memories(user_input, conversation_history)

        # Format memories for inclusion in prompt
        memory_text = self.format_memories_fn(memories)

        # Create augmented prompt
        if memory_text and self.include_memories_as_prefix:
            augmented_input = f"{memory_text}\n\nUser: {user_input}"
        else:
            augmented_input = f"User: {user_input}"

        # Tokenize and generate
        inputs = self.tokenizer(augmented_input, return_tensors="pt")
        outputs = self.model.generate(
            inputs.input_ids, attention_mask=inputs.attention_mask, **generation_kwargs
        )

        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Store in memory if requested
        if store_in_memory:
            self.store_interaction(user_input, response)

        return response


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI API."""

    def __init__(
        self,
        memory_system: dict[str, Any],
        api_client: Any,
        model: str = "gpt-3.5-turbo",
        format_memories_fn: Optional[Callable] = None,
        system_prompt: str = "You are a helpful assistant with access to conversation memory.",
    ):
        """
        Initialize the OpenAI adapter.

        Args:
            memory_system: dictionary containing memory components
            api_client: OpenAI API client
            model: Model identifier
            format_memories_fn: Function to format memories for prompt inclusion
            system_prompt: System prompt for the conversation
        """
        super().__init__(memory_system, format_memories_fn)
        self.api_client = api_client
        self.model = model
        self.system_prompt = system_prompt

    def generate(
        self,
        user_input: str,
        conversation_history: Optional[list[dict]] = None,
        generation_kwargs: Optional[dict] = None,
        store_in_memory: bool = True,
    ) -> str:
        """
        Generate a response using the OpenAI API with memory augmentation.

        Args:
            user_input: User's input text
            conversation_history: Previous conversation turns
            generation_kwargs: Additional kwargs for API call
            store_in_memory: Whether to store the interaction in memory

        Returns:
            Model's generated response
        """
        if generation_kwargs is None:
            generation_kwargs = {}

        if conversation_history is None:
            conversation_history = []

        # Retrieve relevant memories
        memories = self.retrieve_memories(user_input, conversation_history)

        # Format memories
        memory_text = self.format_memories_fn(memories)

        # Create messages for API
        messages = [{"role": "system", "content": self.system_prompt}]

        # Include memory context if available
        if memory_text:
            messages.append({
                "role": "system",
                "content": f"Relevant context from memory: {memory_text}",
            })

        # Add conversation history
        for turn in conversation_history[-5:]:  # Include last 5 turns
            speaker = turn.get("speaker", "").lower()
            if "user" in speaker:
                messages.append({"role": "user", "content": turn.get("message", "")})
            else:
                messages.append({"role": "assistant", "content": turn.get("message", "")})

        # Add current user input
        messages.append({"role": "user", "content": user_input})

        # Call API
        response = self.api_client.chat.completions.create(
            model=self.model, messages=messages, **generation_kwargs
        )

        # Extract response text
        response_text = response.choices[0].message.content

        # Store in memory if requested
        if store_in_memory:
            self.store_interaction(user_input, response_text)

        return response_text


class LangChainAdapter(BaseAdapter):
    """Adapter for LangChain framework."""

    def __init__(
        self, memory_system: dict[str, Any], llm: Any, format_memories_fn: Optional[Callable] = None
    ):
        """
        Initialize the LangChain adapter.

        Args:
            memory_system: dictionary containing memory components
            llm: LangChain LLM instance
            format_memories_fn: Function to format memories for prompt inclusion
        """
        super().__init__(memory_system, format_memories_fn)
        self.llm = llm

    def generate(
        self, user_input: str, conversation_history: list[dict], store_in_memory: bool = True
    ) -> str:
        """
        Generate a response using LangChain with memory augmentation.

        Args:
            user_input: User's input text
            conversation_history: Previous conversation turns
            store_in_memory: Whether to store the interaction in memory

        Returns:
            Model's generated response
        """
        # Retrieve relevant memories
        memories = self.retrieve_memories(user_input, conversation_history)

        # Format memories
        memory_text = self.format_memories_fn(memories)

        # Create prompt for LangChain
        if memory_text:
            prompt = f"""Relevant memory context:
{memory_text}

User query: {user_input}

Please respond based on both the query and the relevant memory context:"""
        else:
            prompt = user_input

        # Generate response
        response = self.llm.invoke(prompt)

        # Store in memory if requested
        if store_in_memory:
            self.store_interaction(user_input, response)

        return response
