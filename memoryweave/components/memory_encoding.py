"""
Memory encoding component for MemoryWeave.

This module provides functionality for encoding different types of content
into memory representations with rich contextual information.
"""

from typing import Any, Optional

import numpy as np
from pydantic import Field

from memoryweave.components.base import Component
from memoryweave.components.context_enhancement import ContextualEmbeddingEnhancer


class MemoryEncoder(Component):
    """
    Encodes different types of content into memory representations.

    This component handles the encoding of various content types (text, interactions,
    concepts) into memory embeddings with rich contextual information.
    """

    embedding_model: Any = Field(
        default=...,
        description="Model to use for creating embeddings",
    )
    context_enhancer: ContextualEmbeddingEnhancer = Field(
        default_factory=ContextualEmbeddingEnhancer,
        description="Contextual embedding enhancer",
    )
    conversation_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="History of interactions in the conversation",
    )
    context_window_size: int = Field(
        default=3,
        description="Size of context window for enriching embeddings",
    )
    use_episodic_markers: bool = Field(
        default=True,
        description="Whether to use episodic markers in encoding",
    )

    def initialize(self, config: dict[str, Any]) -> None:
        """
        Initialize the component with configuration.

        Args:
            config: Configuration dictionary with parameters:
                - context_window_size: Size of context window for enriching embeddings (default: 3)
                - use_episodic_markers: Whether to use episodic markers in encoding (default: True)
        """
        self.context_window_size = config.get("context_window_size", 3)
        self.use_episodic_markers = config.get("use_episodic_markers", True)

        # Initialize context enhancer with configuration
        context_enhancer_config = config.get("context_enhancer", {})
        self.context_enhancer.initialize(context_enhancer_config)

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text into an embedding vector.

        Args:
            text: Text to encode

        Returns:
            Embedding vector
        """
        return self.embedding_model.encode(text)

    def encode_interaction(
        self,
        query: str,
        response: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Encode a query-response interaction into an embedding vector.

        Args:
            query: The query text
            response: The response text
            metadata: Additional metadata for the interaction

        Returns:
            Embedding vector
        """
        # Append to conversation history
        interaction = {
            "message": query,
            "response": response,
            "timestamp": len(self.conversation_history),
        }
        if metadata:
            interaction.update(metadata)

        self.conversation_history.append(interaction)

        # Create contextually enriched content for embedding
        context_window = self._get_context_window()
        contextual_content = self._create_contextual_content(interaction, context_window)

        # Generate embedding
        embedding = self._generate_embedding(contextual_content)

        # Apply context enhancement if available
        if hasattr(self.context_enhancer, "process"):
            # Create context for enhancement
            enhancement_context = {
                "conversation_history": self.conversation_history[-self.context_window_size :],
                "current_time": interaction.get("timestamp", 0),
                "topics": metadata.get("topics", set()) if metadata else set(),
            }

            # Process the embedding with context enhancement
            enhanced_data = self.context_enhancer.process(
                {"embedding": embedding, "content": contextual_content},
                enhancement_context,
            )

            # Use enhanced embedding if available
            if "embedding" in enhanced_data:
                embedding = enhanced_data["embedding"]

        return embedding

    def encode_concept(
        self,
        concept: str,
        definition: str,
        examples: Optional[list[str]] = None,
    ) -> np.ndarray:
        """
        Encode a concept into an embedding vector.

        Args:
            concept: The concept name
            definition: Definition of the concept
            examples: Optional list of examples

        Returns:
            Embedding vector
        """
        # Create enriched content for embedding
        content = f"Concept: {concept}\nDefinition: {definition}"
        if examples:
            content += f"\nExamples: {', '.join(examples)}"

        # Generate embedding
        embedding = self._generate_embedding(content)

        return embedding

    def _get_context_window(self) -> list[dict[str, Any]]:
        """Get a window of recent interactions for context."""
        history_len = len(self.conversation_history)
        start_idx = max(0, history_len - self.context_window_size - 1)
        return self.conversation_history[start_idx : history_len - 1]

    def _create_contextual_content(
        self, current_interaction: dict[str, Any], context_window: list[dict[str, Any]]
    ) -> str:
        """Create contextually enriched content for embedding."""
        # Start with the current interaction
        content = f"Query: {current_interaction['message']}"

        # Add response if available
        if current_interaction.get("response"):
            content += f"\nResponse: {current_interaction['response']}"

        # Add context from previous interactions
        if context_window and self.use_episodic_markers:
            content += "\nContext:"
            for ctx in context_window:
                content += f"\n- Query: {ctx['message']}"
                if ctx.get("response"):
                    content += f"\n  Response: {ctx['response']}"

        return content

    def _generate_embedding(self, content: str) -> np.ndarray:
        """Generate an embedding for the content."""
        return self.embedding_model.encode(content)

    def process(self, data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """
        Process data to create memory embeddings.

        Args:
            data: Data to process, containing content to encode
            context: Context information

        Returns:
            Dictionary with embedding and metadata
        """
        content_type = data.get("type", "text")
        result = {}

        if content_type == "text":
            text = data.get("text", "")
            embedding = self.encode_text(text)
            result["embedding"] = embedding
            result["metadata"] = {"type": "text", "content": text}

        elif content_type == "interaction":
            query = data.get("query", "")
            response = data.get("response", "")
            metadata = data.get("metadata", {})
            embedding = self.encode_interaction(query, response, metadata)
            result["embedding"] = embedding
            result["metadata"] = {
                "type": "interaction",
                "query": query,
                "response": response,
                **metadata,
            }

        elif content_type == "concept":
            concept = data.get("concept", "")
            definition = data.get("definition", "")
            examples = data.get("examples", [])
            embedding = self.encode_concept(concept, definition, examples)
            result["embedding"] = embedding
            result["metadata"] = {
                "type": "concept",
                "concept": concept,
                "definition": definition,
                "examples": examples,
            }

        return result

    def process_query(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Process a query for embedding in the retrieval pipeline.

        Args:
            query: Query string to process
            context: Context information including working context

        Returns:
            Updated context with query embedding
        """
        # Create basic embedding for the query text
        query_embedding = self.encode_text(query)

        # Include query parameters from context if available
        working_context = context.get("working_context", {})

        # Update context with query embedding for downstream components
        result = {"query_embedding": query_embedding, "query_processed": True}

        # Apply context enhancement if available
        if hasattr(self.context_enhancer, "process") and "conversation_history" in working_context:
            # Prepare enhancement context
            enhancement_context = {
                "conversation_history": working_context.get("conversation_history", []),
                "current_time": working_context.get("current_time", 0),
                "topics": working_context.get("topics", set()),
            }

            # Process with contextual enhancement
            enhanced_data = self.context_enhancer.process(
                {"embedding": query_embedding, "content": query},
                enhancement_context,
            )

            # Use enhanced embedding if available
            if "embedding" in enhanced_data:
                result["query_embedding"] = enhanced_data["embedding"]
                result["query_enhanced"] = True

        return result
