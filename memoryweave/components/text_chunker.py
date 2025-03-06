"""
Text chunking component for MemoryWeave.

This module provides functionality for breaking down large texts into semantically
meaningful chunks to improve embedding quality and retrieval accuracy.
"""

import re
from typing import Any

from memoryweave.benchmarks.utils.perf_timer import timer
from memoryweave.components.base import Component
from memoryweave.components.component_names import ComponentName


@timer
class TextChunker(Component):
    """
    Component for chunking text into smaller segments for better embedding representation.

    This component breaks down large texts into semantically meaningful chunks,
    which improves embedding quality and retrieval accuracy for large contexts.
    """

    def __init__(self):
        """Initialize the text chunker."""
        self.component_id = ComponentName.TEXT_CHUNKER

        # Default configuration
        self.chunk_size = 200  # Target size in characters
        self.chunk_overlap = 50  # Characters of overlap between chunks
        self.min_chunk_size = 50  # Minimum chunk size to process
        self.respect_paragraphs = True  # Try to keep paragraphs intact
        self.respect_sentences = True  # Try not to break sentences
        self.include_metadata = True  # Include position metadata in chunks

    def get_id(self) -> str:
        """Get the unique identifier for this component."""
        return self.component_id

    def get_type(self):
        """Get the type of this component."""
        from memoryweave.interfaces.pipeline import ComponentType

        return ComponentType.PROCESSOR

    def initialize(self, config: dict[str, Any] = None) -> None:
        """Initialize the chunker with the provided configuration."""
        if config is None:
            config = {}

        self.chunk_size = config.get("chunk_size", self.chunk_size)
        self.chunk_overlap = config.get("chunk_overlap", self.chunk_overlap)
        self.min_chunk_size = config.get("min_chunk_size", self.min_chunk_size)
        self.respect_paragraphs = config.get("respect_paragraphs", self.respect_paragraphs)
        self.respect_sentences = config.get("respect_sentences", self.respect_sentences)
        self.include_metadata = config.get("include_metadata", self.include_metadata)

    @timer()
    def create_chunks(self, text: str, metadata: dict[str, Any] = None) -> list[dict[str, Any]]:
        """
        Create chunks from the input text.

        Args:
            text: The text to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of dictionaries containing chunk text and metadata
        """
        if not text:
            return []

        # Use appropriate chunking strategy based on configuration
        if self.respect_paragraphs:
            chunks = self._chunk_by_paragraphs(text)
        else:
            chunks = self._chunk_by_size(text)

        # Ensure consistent chunk size if needed
        chunks = self._normalize_chunks(chunks)

        # Add metadata to chunks
        result = []
        base_metadata = metadata or {}

        for i, chunk in enumerate(chunks):
            chunk_metadata = base_metadata.copy()

            # Add position metadata if requested
            if self.include_metadata:
                chunk_metadata.update({
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                    "is_first_chunk": i == 0,
                    "is_last_chunk": i == len(chunks) - 1,
                    "text_start_position": text.find(chunk[:50]),  # Approximate position
                    "chunk_size": len(chunk),
                })

            result.append({"text": chunk, "metadata": chunk_metadata})

        return result

    def process_conversation(self, turns: list[dict[str, str]]) -> list[dict[str, Any]]:
        """
        Process a conversation into appropriate chunks.

        This method is specialized for conversation data, preserving turn context
        while creating appropriately sized chunks.

        Args:
            turns: List of conversation turns with "role" and "content" keys

        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []
        current_chunk = []
        current_length = 0

        for i, turn in enumerate(turns):
            role = turn.get("role", "unknown")
            content = turn.get("content", "")

            turn_text = f"{role}: {content}"

            # If this turn would make the chunk too large, finalize current chunk
            if current_length > 0 and current_length + len(turn_text) > self.chunk_size:
                chunk_text = "\n".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "chunk_index": len(chunks),
                        "is_conversation": True,
                        "turn_range": (i - len(current_chunk), i - 1),
                    },
                })
                current_chunk = []
                current_length = 0

            # Add this turn
            current_chunk.append(turn_text)
            current_length += len(turn_text)

        # Add the final chunk if there's anything left
        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "chunk_index": len(chunks),
                    "is_conversation": True,
                    "turn_range": (len(turns) - len(current_chunk), len(turns) - 1),
                },
            })

        return chunks

    def _chunk_by_paragraphs(self, text: str) -> list[str]:
        """
        Chunk text by paragraphs, respecting natural text boundaries.

        This method tries to keep paragraphs intact while maintaining
        the target chunk size.
        """
        # Split text into paragraphs
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return [text]

        chunks = []
        current_chunk = []
        current_length = 0

        for paragraph in paragraphs:
            # If adding this paragraph would exceed the chunk size,
            # finalize the current chunk (if we have content)
            if current_length > 0 and current_length + len(paragraph) > self.chunk_size:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_length = 0

            # If paragraph itself exceeds chunk size, we need to split it
            if len(paragraph) > self.chunk_size:
                # First add any accumulated paragraphs
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # Then split the large paragraph
                if self.respect_sentences:
                    para_chunks = self._split_paragraph_by_sentences(paragraph)
                else:
                    para_chunks = self._split_text_by_size(paragraph)

                chunks.extend(para_chunks)
            else:
                # Add paragraph to current chunk
                current_chunk.append(paragraph)
                current_length += len(paragraph)

        # Add the final chunk if there's anything left
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _chunk_by_size(self, text: str) -> list[str]:
        """
        Chunk text by size, optionally respecting sentence boundaries.
        """
        if self.respect_sentences:
            return self._split_text_by_sentences(text)
        else:
            return self._split_text_by_size(text)

    def _split_text_by_sentences(self, text: str) -> list[str]:
        """
        Split text into chunks while trying to keep sentences intact.
        """
        # Simple sentence splitting pattern
        sentence_pattern = r"(?<=[.!?])\s+"
        sentences = re.split(sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            # If sentence itself is too large, split it further
            if len(sentence) > self.chunk_size:
                # First add any accumulated sentences
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # Split the sentence by size
                sentence_chunks = self._split_text_by_size(sentence)
                chunks.extend(sentence_chunks)
                continue

            # If adding this sentence would exceed the chunk size,
            # finalize the current chunk
            if current_length > 0 and current_length + len(sentence) + 1 > self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += len(sentence) + 1  # +1 for space

        # Add the final chunk if there's anything left
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _split_paragraph_by_sentences(self, paragraph: str) -> list[str]:
        """
        Split a large paragraph into chunks by sentences.
        """
        return self._split_text_by_sentences(paragraph)

    def _split_text_by_size(self, text: str) -> list[str]:
        """
        Split text into chunks of approximately target_size characters.
        This method doesn't try to respect sentence boundaries.
        """
        chunks = []

        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk_end = min(i + self.chunk_size, len(text))

            # Skip small final chunks
            if chunk_end - i < self.min_chunk_size:
                # Either combine with previous chunk or skip
                if chunks:
                    # Get the last chunk and append this text
                    last_chunk = chunks.pop()
                    combined = last_chunk + text[i:chunk_end]
                    chunks.append(combined)
                continue

            chunk = text[i:chunk_end]
            chunks.append(chunk)

            # Break if we've reached the end
            if chunk_end == len(text):
                break

        return chunks

    def _normalize_chunks(self, chunks: list[str]) -> list[str]:
        """
        Ensure chunks meet minimum size requirements.

        This combines or filters very small chunks.
        """
        if not chunks:
            return []

        # Filter out empty chunks
        filtered_chunks = [c for c in chunks if c.strip()]

        # Check if any chunks are too small
        too_small = [i for i, c in enumerate(filtered_chunks) if len(c) < self.min_chunk_size]

        if not too_small:
            return filtered_chunks

        # Combine small chunks with neighbors where possible
        result = []
        skip_next = False

        for i in range(len(filtered_chunks)):
            if skip_next:
                skip_next = False
                continue

            current = filtered_chunks[i]

            # If current chunk is too small and not the last one
            if len(current) < self.min_chunk_size and i < len(filtered_chunks) - 1:
                next_chunk = filtered_chunks[i + 1]
                combined = current + "\n" + next_chunk
                result.append(combined)
                skip_next = True
            # If current chunk is too small and is the last one
            elif len(current) < self.min_chunk_size and i > 0 and result:
                # Combine with previous chunk
                prev_chunk = result.pop()
                combined = prev_chunk + "\n" + current
                result.append(combined)
            else:
                # Normal sized chunk
                result.append(current)

        return result
