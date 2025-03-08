# memoryweave/models/types.py
from typing import Any, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator

# Define MemoryID as a custom type that can be either int or str but is normalized internally
MemoryID = Union[int, str]


class MemoryContent(BaseModel):
    """Model for memory content with flexible text representation."""

    text: Optional[str] = None
    raw_content: Optional[Any] = None

    def __str__(self) -> str:
        if self.text is not None:
            return self.text
        if self.raw_content is not None:
            return str(self.raw_content)
        return ""


class Memory(BaseModel):
    """Model for a memory entry."""

    id: MemoryID
    embedding: Any  # For numpy array support
    content: Union[str, dict[str, Any], MemoryContent]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def validate_id(cls, v):
        """Ensure ID is either an int or str."""
        if isinstance(v, (int, str)):
            return v
        return str(v)

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v):
        """Ensure embedding is a numpy array."""
        if isinstance(v, np.ndarray):
            return v
        elif isinstance(v, list):
            return np.array(v, dtype=np.float32)
        return v  # Return as is if we can't convert


class AssociativeLink(BaseModel):
    """Model for a link between memories."""

    source_id: MemoryID
    target_id: MemoryID
    strength: float = Field(ge=0.0, le=1.0)

    @field_validator("source_id", "target_id")
    @classmethod
    def validate_id(cls, v):
        """Ensure IDs are either int or str."""
        if isinstance(v, (int, str)):
            return v
        return str(v)


class RetrievalResult(BaseModel):
    """Model for a memory retrieval result."""

    memory_id: MemoryID
    relevance_score: float = Field(ge=0.0, le=1.0)
    content: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


# Helper functions for ID normalization
def normalize_memory_id_to_int(memory_id: MemoryID) -> int:
    """Convert any valid memory ID to integer."""
    if isinstance(memory_id, str):
        try:
            return int(memory_id)
        except ValueError:
            # For non-numeric string IDs, use hash
            return hash(memory_id) % (2**31)
    return memory_id


def normalize_memory_id_to_str(memory_id: MemoryID) -> str:
    """Convert any valid memory ID to string."""
    return str(memory_id)


def memory_ids_equal(id1: MemoryID, id2: MemoryID) -> bool:
    """Check if two memory IDs are equal regardless of type."""
    if id1 == id2:
        return True

    try:
        if isinstance(id1, str) and not isinstance(id2, str):
            return int(id1) == id2
        elif isinstance(id2, str) and not isinstance(id1, str):
            return id1 == int(id2)
    except (ValueError, TypeError):
        # If conversion fails, fall back to string comparison
        return str(id1) == str(id2)

    return False
