# MemoryWeave Knowledge File

## Project Structure

- `memoryweave/`: Main package directory
  - `core/`: Core functionality
  - `utils/`: Utility modules
    - `nlp_extraction.py`: NLP-based extraction utilities
  - `models/`: Data models
  - `api/`: API interfaces
- `tests/`: Test suite
- `notebooks/`: Jupyter notebooks (LARGE directory - avoid analyzing unless explicitly requested)

## Extraction Approaches

### NLP Extraction (Current Approach)

The project now uses a pure NLP-based approach for extracting personal attributes and identifying query types. This is implemented in `memoryweave/utils/nlp_extraction.py`.

Key features:
- Uses spaCy for advanced NLP capabilities when available
- Falls back to basic NLP techniques when spaCy is not available
- Extracts personal attributes, preferences, relationships, and traits
- Identifies query types (factual, personal, opinion, instruction)
- More adaptable to variations in language compared to regex

### Regex Extraction (Legacy Approach)

Previously, the project used regex patterns for extraction. This approach was:
- More rigid and prone to missing variations in language
- Faster but less accurate
- Required constant pattern maintenance

The regex approach is being phased out in favor of the more flexible NLP approach.

## Testing Approach

The NLP extraction capabilities are tested in `test_nlp_extraction.py`, which evaluates:
- Personal attribute extraction from various text samples
- Query type identification
- Performance with and without spaCy

## Performance Considerations

- spaCy is used when available for best results
- Basic NLP techniques serve as fallbacks
- Extraction performance depends on the quality of input text
- Future improvements will focus on enhancing extraction accuracy rather than performance optimizations
