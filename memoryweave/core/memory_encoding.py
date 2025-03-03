"""
DEPRECATED: Handles encoding of different content types into the contextual fabric.

This module is deprecated. Please use the new component-based architecture instead.
"""

import warnings
from typing import Any, Optional

import numpy as np

warnings.warn(
    "memoryweave.core.memory_encoding is deprecated. "
    "Use memoryweave.components.memory_adapter instead.",
    DeprecationWarning,
    stacklevel=2,
)

class MemoryEncoder:
    """
    DEPRECATED: Encodes different types of content into memory representations.

    This class is deprecated and will be removed in a future version.
    Please use the component-based architecture instead.
    """

    def __init__(
        self,
        embedding_model: Any,
        use_episodic_markers: bool = True,
        context_window_size: int = 3,
    ):
        """
        Initialize the memory encoder.

        Args:
            embedding_model: Model to use for creating embeddings
            use_episodic_markers: Whether to use episodic markers in encoding
            context_window_size: Size of context window for enriching embeddings
        """
        warnings.warn(
            "MemoryEncoder is deprecated and will be removed in a future version. "
            "Use memoryweave.components architecture instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.embedding_model = embedding_model
        self.use_episodic_markers = use_episodic_markers
        self.context_window_size = context_window_size
        self.conversation_history = []

    def encode_interaction(
        self,
        message: str,
        speaker: str,
        response: Optional[str] = None,
        additional_context: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Encode a conversation interaction into a contextual memory.

        Args:
            message: The message content
            speaker: Identifier for who spoke the message
            response: Optional response to the message
            additional_context: Additional contextual information

        Returns:
            tuple of (embedding, metadata)
        """
        # Append to conversation history
        interaction = {
            "message": message,
            "speaker": speaker,
            "response": response,
            "timestamp": len(self.conversation_history),
        }
        if additional_context:
            interaction.update(additional_context)

        self.conversation_history.append(interaction)

        # Create contextually enriched content for embedding
        context_window = self._get_context_window()
        contextual_content = self._create_contextual_content(interaction, context_window)

        # Generate embedding
        embedding = self._generate_embedding(contextual_content)

        # Create rich metadata
        metadata = {
            "type": "interaction",
            "content": message,
            "speaker": speaker,
            "response": response,
            "position": len(self.conversation_history) - 1,
        }
        if additional_context:
            metadata["context"] = additional_context

        return embedding, metadata

    def encode_concept(
        self,
        concept: str,
        description: str,
        related_concepts: Optional[list[str]] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Encode a concept into the contextual fabric.

        Args:
            concept: The concept name
            description: Description of the concept
            related_concepts: list of related concepts

        Returns:
            tuple of (embedding, metadata)
        """
        # Create enriched content for embedding
        content = f"Concept: {concept}\nDescription: {description}"
        if related_concepts:
            content += f"\nRelated concepts: {', '.join(related_concepts)}"

        # Generate embedding
        embedding = self._generate_embedding(content)

        # Create metadata
        metadata = {
            "type": "concept",
            "name": concept,
            "description": description,
        }
        if related_concepts:
            metadata["related_concepts"] = related_concepts

        return embedding, metadata

    def _get_context_window(self) -> list[dict]:
        """Get a window of recent interactions for context."""
        history_len = len(self.conversation_history)
        start_idx = max(0, history_len - self.context_window_size - 1)
        return self.conversation_history[start_idx : history_len - 1]

    def _create_contextual_content(
        self, current_interaction: dict, context_window: list[dict]
    ) -> str:
        """Create contextually enriched content for embedding."""
        # Start with the current interaction
        content = f"{current_interaction['speaker']}: {current_interaction['message']}"

        # Add response if available
        if current_interaction.get("response"):
            content += f"\nResponse: {current_interaction['response']}"

        # Add context from previous interactions
        if context_window:
            content += "\nContext:"
            for ctx in context_window:
                content += f"\n- {ctx['speaker']}: {ctx['message']}"
                if ctx.get("response"):
                    content += f" → Response: {ctx['response']}"

        return content

    def _generate_embedding(self, content: str) -> np.ndarray:
        """Generate an embedding for the content."""
        # This is a placeholder - in practice, you would use your embedding model
        return self.embedding_model.encode(content)
