# MemoryWeave LLM Integration Examples

This directory contains examples of integrating MemoryWeave with Language Models to enhance conversation capabilities through advanced memory management.

## Files

- `memoryweave_llm_wrapper.py` - A wrapper class that integrates MemoryWeave with Hugging Face models
- `conversation_simulation.py` - A script that runs simulated conversations to demonstrate MemoryWeave's capabilities

## Requirements

To run these examples, you'll need the following additional packages:

```bash
pip install transformers torch sentence-transformers accelerate
```

For better performance, consider installing `bitsandbytes` and `unsloth` (for Llama models):

```bash
pip install bitsandbytes unsloth
```

## Running the Examples

### Basic simulation

Run the conversation simulation with default settings:

```bash
# Navigate to the root directory of the project
cd /path/to/memoryweave

# Run the simulation
uv run python examples/llm/conversation_simulation.py
```

### Using a specific model

You can specify a different Hugging Face model:

```bash
uv run python examples/llm/conversation_simulation.py --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

### Comparing with and without memory

To see the difference MemoryWeave makes, run a comparison:

```bash
uv run python examples/llm/conversation_simulation.py --compare
```

This will run the simulation twice - once with memory features enabled and once without - so you can directly compare the results.

## Evaluating Results

When evaluating the effectiveness of MemoryWeave, look for:

1. **Memory Recall** - Does the assistant correctly remember personal details and preferences?
1. **Conversation Coherence** - Does the conversation flow naturally, with appropriate references to past information?
1. **Response Quality** - Are responses more personalized and relevant with memory enabled?
1. **Retrieval Accuracy** - Are the right memories being selected for each query?

## Example Conversation

Here's a sample of what to expect in the output:

```
=== Conversation 1: Small Talk ===
User: Hello! How are you today?
Assistant (1.25s): Hello! I'm doing quite well today, thank you for asking. How about yourself? Is there anything I can assist you with today?

User: I'm doing well. What can you help me with?
Assistant (1.18s): I'm glad to hear you're doing well! I can help you with a wide range of tasks, including:

- Answering questions on various topics
- Providing information and explanations
- Offering suggestions or recommendations
- Helping with creative writing or brainstorming
- Discussing concepts or ideas
- Assisting with planning or organization

Feel free to ask me about anything you're curious about, and I'll do my best to assist you!

=== Conversation 3: Memory Recall Test ===
User: Where do I live?
Assistant (1.31s): Based on the information I have, you live in Boston.

User: What's my pet's name?
Assistant (1.22s): Based on what you've shared previously, your dog's name is Max.
```

## Implementation Notes

The MemoryWeaveLLM class in `memoryweave_llm_wrapper.py` interacts with the memory store as follows:

1. Adding memories:

   ```python
   # Create an embedding for the text
   embedding = embedding_model.encode(text)

   # Add the memory to the store with optional metadata
   memory_manager.memory_store.add(embedding, text, metadata)
   ```

1. Retrieving memories:

   ```python
   # Retrieve relevant memories for a query
   relevant_memories = retriever.retrieve(query, top_k=3)
   ```

1. Storing conversation interactions:

   ```python
   # Store a user message
   user_embedding = embedding_model.encode(user_message)
   memory_manager.memory_store.add(user_embedding, user_message, metadata)

   # Store the assistant's response
   assistant_embedding = embedding_model.encode(assistant_message)
   memory_manager.memory_store.add(assistant_embedding, assistant_message, metadata)
   ```

For detailed implementation, see the `memoryweave_llm_wrapper.py` file.

## Customizing the Integration

You can modify the `MemoryWeaveLLM` class in `memoryweave_llm_wrapper.py` to:

- Use different embedding models
- Change MemoryWeave configuration for different use cases
- Implement more sophisticated personal information extraction
- Add specialized memory types for different domains
