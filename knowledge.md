# MemoryWeave Project Knowledge

## Project Overview
MemoryWeave is a novel approach to memory management for language models that uses a "contextual fabric" approach inspired by biological memory systems. Instead of traditional knowledge graph approaches with discrete nodes and edges, MemoryWeave focuses on capturing rich contextual signatures of information for improved long-context coherence in language model conversations.

## Key Features
- **Contextual Fabric**: Memory traces capture rich contextual signatures rather than isolated facts
- **Activation-Based Retrieval**: Memory retrieval uses dynamic activation patterns similar to biological systems
- **Episodic Structure**: Memories maintain temporal relationships and episodic anchoring
- **Non-Structured Memory**: Works with raw LLM outputs without requiring structured formats
- **Modular Architecture**: Easily integrates with existing LLM inference frameworks
- **ART-Inspired Clustering**: Optional Adaptive Resonance Theory inspired memory categorization

## Project Structure
- `memoryweave/core/`: Core components of the memory system
- `memoryweave/integrations/`: Adapters for various LLM frameworks
- `memoryweave/evaluation/`: Metrics and tools for evaluating performance
- `memoryweave/examples/`: Example usage patterns
- `memoryweave/utils/`: Utility functions

## Dependencies
- Python 3.12+
- Key packages: faiss-cpu, huggingface-hub, numpy, pydantic, torch, transformers

## Development Guidelines
- Follow PEP 8 style guidelines
- Include docstrings for all public functions and classes
- Write unit tests for new functionality

## Working with this Repository
- Use `uv` for package management and running Python scripts
- Install the package in development mode with: `uv pip install -e .`
- Run Python scripts with: `uv run python script_name.py`
- Run tests with: `uv run python -m pytest`
