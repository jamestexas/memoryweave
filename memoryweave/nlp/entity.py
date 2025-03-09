# memoryweave/nlp/entity.py
import re

from pydantic import BaseModel, Field

from memoryweave.nlp.interfaces import EntityExtractor


class Entity(BaseModel):
    """A named entity extracted from text."""

    text: str
    label: str
    start: int
    end: int
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    class Config:
        frozen = True


class SpacyEntityExtractor(EntityExtractor):
    """Entity extractor using spaCy."""

    def __init__(self, model: str = "en_core_web_sm"):
        self._model_name = model
        self._nlp = None
        self._initialized = False

        # Try to initialize SpaCy
        self._initialize()

    def _initialize(self) -> None:
        """Initialize spaCy if not already initialized."""
        if self._initialized:
            return

        try:
            import spacy

            self._nlp = spacy.load(self._model_name)
            self._initialized = True
        except (ImportError, OSError):
            self._nlp = None
            self._initialized = False

    @property
    def available(self) -> bool:
        """Check if spaCy is available."""
        return self._initialized and self._nlp is not None

    def extract(self, text: str) -> list[Entity]:
        """Extract entities from text."""
        return self.extract_entities(text)

    def extract_entities(self, text: str) -> list[Entity]:
        """Extract entities using spaCy."""
        if not self.available or not text:
            return []

        # Process the text with spaCy
        doc = self._nlp(text)

        entities = []
        for ent in doc.ents:
            entity = Entity(text=ent.text, label=ent.label_, start=ent.start_char, end=ent.end_char)
            entities.append(entity)

        return entities


class RegexEntityExtractor(EntityExtractor):
    """Entity extractor using regular expressions."""

    def __init__(self):
        # Common regex patterns for entities
        self._patterns = {
            "PERSON": r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
            "LOCATION": r"\b([A-Z][a-z]+(?:,\s+[A-Z][a-z]+)*)\b",
            "ORGANIZATION": r"\b([A-Z][a-z]*(?:\s+[A-Z][a-z]*)+)\b",
            "DATE": r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b|\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:[a-z]*)?(?:,\s+\d{4})?)\b",
            "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "PHONE": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        }

        # Compile patterns
        self._compiled_patterns = {
            entity_type: re.compile(pattern) for entity_type, pattern in self._patterns.items()
        }

    @property
    def available(self) -> bool:
        """Always available since only using builtin modules."""
        return True

    def extract(self, text: str) -> list[Entity]:
        """Extract entities from text."""
        return self.extract_entities(text)

    def extract_entities(self, text: str) -> list[Entity]:
        """Extract entities using regex patterns."""
        if not text:
            return []

        entities = []

        # Apply each pattern
        for entity_type, pattern in self._compiled_patterns.items():
            for match in pattern.finditer(text):
                start, end = match.span()
                entity_text = match.group(0)

                # Create entity
                entity = Entity(
                    text=entity_text,
                    label=entity_type,
                    start=start,
                    end=end,
                    # Regex is less reliable than ML-based methods
                    confidence=0.7,
                )
                entities.append(entity)

        # Sort by position in text
        entities.sort(key=lambda e: e.start)

        # Remove overlapping entities, preferring longer ones
        return self._remove_overlapping_entities(entities)

    def _remove_overlapping_entities(self, entities: list[Entity]) -> list[Entity]:
        """Remove overlapping entities, keeping the longest ones."""
        if not entities:
            return []

        # Sort by length (descending) to prefer longer entities
        sorted_entities = sorted(entities, key=lambda e: e.end - e.start, reverse=True)

        # Keep track of used character positions
        used_positions: set[int] = set()
        filtered_entities = []

        for entity in sorted_entities:
            # Check if this entity overlaps with already selected entities
            entity_positions = set(range(entity.start, entity.end))
            if not entity_positions.intersection(used_positions):
                # No overlap, add to filtered list
                filtered_entities.append(entity)
                used_positions.update(entity_positions)

        # Sort by position in text
        filtered_entities.sort(key=lambda e: e.start)

        return filtered_entities
