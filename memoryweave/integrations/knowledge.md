# Integrations Module Knowledge

## Module Purpose
The integrations module provides adapters to easily integrate MemoryWeave with popular LLM frameworks and APIs.

## Key Components
- `BaseAdapter`: Common functionality for all adapters
- `HuggingFaceAdapter`: Integration with Hugging Face Transformers
- `OpenAIAdapter`: Integration with OpenAI API
- `LangChainAdapter`: Integration with LangChain framework

## Usage Patterns
Each adapter follows a similar pattern:
1. Initialize with a memory system and model-specific components
2. Use the `generate()` method to get responses with memory augmentation
3. Memories are automatically stored and retrieved as needed

## Implementation Details
- Adapters handle formatting memories for inclusion in prompts
- They manage the interaction between the memory system and the LLM
- Each adapter is tailored to the specific API of its target framework

## Extension
To add support for a new framework:
1. Create a new adapter class inheriting from `BaseAdapter`
2. Implement the framework-specific `generate()` method
3. Add the new adapter to the `__init__.py` exports
