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
- Uses spaCy for advanced NLP capabilities
- Extracts personal attributes, preferences, relationships, and traits
- Identifies query types (factual, personal, opinion, instruction)
- More adaptable to variations in language compared to regex
- Uses rule-based matchers, dependency parsing, and named entity recognition

### spaCy Integration

The NLP extraction system requires spaCy and uses several advanced features:
- Rule-based matchers for pattern recognition
- Dependency parsing for relationship extraction
- Named Entity Recognition (NER) for identifying people, locations, etc.
- Custom pipeline components for attribute extraction
- Part-of-speech tagging for identifying key elements

## Testing Approach

The NLP extraction capabilities are tested in `test_nlp_extraction.py`, which evaluates:
- Personal attribute extraction from various text samples
- Query type identification
- Performance with spaCy

## Performance Considerations

- Extraction performance depends on the quality of input text
- spaCy model size affects extraction quality (larger models = better extraction)
- Future improvements will focus on enhancing extraction accuracy

## Project Roadmap

The project is following a phased development approach:

### 1. Diagnostic Analysis Phase
- Identify core issues with memory retrieval (high recall but poor precision)
- Analyze performance differences between personal vs. factual queries
- Conduct targeted analysis tests with separate test sets
- Analyze embedding space to understand retrieval failures
- Examine similarity distributions and threshold optimization

### 2. Retrieval Strategy Overhaul
- Implement query type detection and specialized retrieval pipelines
- Enhance base similarity computation with hybrid approaches
- Apply advanced semantic matching for factual queries

### 3. Memory Organization Enhancement
- Refine category formation algorithms
- Implement memory metadata enrichment
- Apply hierarchical memory organization

### 4. Confidence and Relevance Calculation Improvements
- Develop calibrated confidence scoring
- Implement relevance diversity optimization
- Apply adaptive result count based on query type

### 5. Integration and Context Management
- Enhance context window management
- Develop entity and reference tracking
- Apply temporal coherence enhancement

### 6. Evaluation and Optimization Framework
- Create comprehensive evaluation suite
- Build parameter optimization system
- Establish continuous improvement pipeline

### 7. Integration with LLM Frameworks
- Enhance prompt construction
- Implement bidirectional memory-LLM communication
- Develop framework-specific optimizations

## Current Focus: Diagnostic Analysis Phase

We are currently focused on the Diagnostic Analysis Phase to identify and address core issues with memory retrieval performance.
