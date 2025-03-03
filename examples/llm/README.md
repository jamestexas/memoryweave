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
# Navigate to the examples/llm directory
cd examples/llm

# Run the simulation
python conversation_simulation.py
```

### Using a specific model

You can specify a different Hugging Face model:

```bash
python conversation_simulation.py --model "meta-llama/Llama-2-7b-chat-hf"
```

### Comparing with and without memory

To see the difference MemoryWeave makes, run a comparison:

```bash
python conversation_simulation.py --compare
```

This will run the simulation twice - once with memory features enabled and once without - so you can directly compare the results.

## Evaluating Results

When evaluating the effectiveness of MemoryWeave, look for:

1. **Memory Recall** - Does the assistant correctly remember personal details and preferences?
2. **Conversation Coherence** - Does the conversation flow naturally, with appropriate references to past information?
3. **Response Quality** - Are responses more personalized and relevant with memory enabled?
4. **Retrieval Accuracy** - Are the right memories being selected for each query?

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

## Customizing the Integration

You can modify the `MemoryWeaveLLM` class in `memoryweave_llm_wrapper.py` to:

- Use different embedding models
- Change MemoryWeave configuration for different use cases
- Implement more sophisticated personal information extraction
- Add specialized memory types for different domains