"""
Conversation simulation to test MemoryWeave with Hugging Face models.

This script demonstrates how MemoryWeave improves conversation quality
by comparing conversations with and without memory capabilities.
"""

import argparse
import os
import sys
import time

from rich import print

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from memoryweave_llm_wrapper import MemoryWeaveLLM

# Default model to use - choose a very small model to avoid timeouts
DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Smaller model to avoid timeouts


def run_simulation(model_name: str, with_memory: bool = True):
    """
    Run a simulated conversation to demonstrate MemoryWeave's capabilities.

    Args:
        model_name: Name of the Hugging Face model to use
        with_memory: Whether to use memory features
    """
    # Initialize the LLM system
    llm = MemoryWeaveLLM(model_name=model_name)

    # Add some pre-existing memories if memory is enabled
    if with_memory:
        llm.add_memory(
            text="The user's name is James and he lives in Boston.",
            metadata={"type": "personal_info", "importance": 0.9},
        )

        llm.add_memory(
            text="James mentioned he has a dog named Max.",
            metadata={"type": "personal_info", "importance": 0.7},
        )

    # Simulation 1: Basic conversation without memory references
    print("\n=== Conversation 1: Small Talk ===")
    print("User: Hello! How are you today?")
    start_time = time.time()
    try:
        response = (
            llm.chat("Hello! How are you today?", max_new_tokens=100)
            if with_memory
            else llm.chat_without_memory("Hello! How are you today?", max_new_tokens=100)
        )
        elapsed = time.time() - start_time
        print(f"Assistant ({elapsed:.2f}s): {response}\n")
    except Exception as e:
        print(f"Error: {e}")
        return llm.get_conversation_history() if with_memory else []

    print("User: I'm doing well. What can you help me with?")
    start_time = time.time()
    try:
        response = (
            llm.chat("I'm doing well. What can you help me with?", max_new_tokens=100)
            if with_memory
            else llm.chat_without_memory("I'm doing well. What can you help me with?", max_new_tokens=100)
        )
        elapsed = time.time() - start_time
        print(f"Assistant ({elapsed:.2f}s): {response}\n")
    except Exception as e:
        print(f"Error: {e}")
        return llm.get_conversation_history() if with_memory else []

    # Simulation 2: Personal preference being stored
    print("\n=== Conversation 2: Sharing Preferences ===")
    print("User: I really like pizza with mushrooms. It's my favorite food.")
    start_time = time.time()
    try:
        response = (
            llm.chat("I really like pizza with mushrooms. It's my favorite food.", max_new_tokens=100)
            if with_memory
            else llm.chat_without_memory("I really like pizza with mushrooms. It's my favorite food.", max_new_tokens=100)
        )
        elapsed = time.time() - start_time
        print(f"Assistant ({elapsed:.2f}s): {response}\n")
    except Exception as e:
        print(f"Error: {e}")
        return llm.get_conversation_history() if with_memory else []

    print("User: I also enjoy hiking on the weekends when the weather is nice.")
    start_time = time.time()
    try:
        response = (
            llm.chat("I also enjoy hiking on the weekends when the weather is nice.", max_new_tokens=100)
            if with_memory
            else llm.chat_without_memory(
                "I also enjoy hiking on the weekends when the weather is nice.", max_new_tokens=100
            )
        )
        elapsed = time.time() - start_time
        print(f"Assistant ({elapsed:.2f}s): {response}\n")
    except Exception as e:
        print(f"Error: {e}")
        return llm.get_conversation_history() if with_memory else []

    # Simulation 3: Testing memory recall of personal information
    print("\n=== Conversation 3: Memory Recall Test ===")
    print("User: Where do I live?")
    start_time = time.time()
    try:
        response = (
            llm.chat("Where do I live?", max_new_tokens=100) 
            if with_memory 
            else llm.chat_without_memory("Where do I live?", max_new_tokens=100)
        )
        elapsed = time.time() - start_time
        print(f"Assistant ({elapsed:.2f}s): {response}\n")
    except Exception as e:
        print(f"Error: {e}")
        return llm.get_conversation_history() if with_memory else []

    print("User: What's my pet's name?")
    start_time = time.time()
    try:
        response = (
            llm.chat("What's my pet's name?", max_new_tokens=100)
            if with_memory
            else llm.chat_without_memory("What's my pet's name?", max_new_tokens=100)
        )
        elapsed = time.time() - start_time
        print(f"Assistant ({elapsed:.2f}s): {response}\n")
    except Exception as e:
        print(f"Error: {e}")
        return llm.get_conversation_history() if with_memory else []

    # Simulation 4: Testing memory of stated preferences
    print("\n=== Conversation 4: Preference Recall ===")
    print("User: What food do I like?")
    start_time = time.time()
    try:
        response = (
            llm.chat("What food do I like?", max_new_tokens=100)
            if with_memory
            else llm.chat_without_memory("What food do I like?", max_new_tokens=100)
        )
        elapsed = time.time() - start_time
        print(f"Assistant ({elapsed:.2f}s): {response}\n")
    except Exception as e:
        print(f"Error: {e}")
        return llm.get_conversation_history() if with_memory else []

    print("User: What activities do I enjoy?")
    start_time = time.time()
    try:
        response = (
            llm.chat("What activities do I enjoy?", max_new_tokens=100)
            if with_memory
            else llm.chat_without_memory("What activities do I enjoy?", max_new_tokens=100)
        )
        elapsed = time.time() - start_time
        print(f"Assistant ({elapsed:.2f}s): {response}\n")
    except Exception as e:
        print(f"Error: {e}")
        return llm.get_conversation_history() if with_memory else []

    # Final summary question
    print("\n=== Final Summary Question ===")
    print("User: Tell me about myself and what I enjoy.")
    start_time = time.time()
    try:
        response = (
            llm.chat("Tell me about myself and what I enjoy.", max_new_tokens=100)
            if with_memory
            else llm.chat_without_memory("Tell me about myself and what I enjoy.", max_new_tokens=100)
        )
        elapsed = time.time() - start_time
        print(f"Assistant ({elapsed:.2f}s): {response}\n")
    except Exception as e:
        print(f"Error: {e}")
        return llm.get_conversation_history() if with_memory else []

    return llm.get_conversation_history() if with_memory else []


def main():
    parser = argparse.ArgumentParser(description="Run a simulated conversation with MemoryWeave")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Hugging Face model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--no-memory", action="store_true", help="Disable MemoryWeave features for comparison"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run two simulations with and without memory for comparison",
    )

    args = parser.parse_args()

    if args.compare:
        print("\n" + "=" * 80)
        print("RUNNING SIMULATION WITH MEMORY ENABLED")
        print("=" * 80)
        with_memory_history = run_simulation(args.model, with_memory=True)

        print("\n" + "=" * 80)
        print("RUNNING SIMULATION WITH MEMORY DISABLED")
        print("=" * 80)
        no_memory_history = run_simulation(args.model, with_memory=False)
        print(f"WITHOUT MEMORY HISTORY: {no_memory_history}")
        print(f"WITH MEMORY HISTORY: {with_memory_history}")
        # Compare results
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        print(
            "The simulation has now been run both with and without memory. Review the outputs above to see how MemoryWeave impacts conversation quality, especially for questions like:"
        )
        print("- 'Where do I live?'")
        print("- 'What's my pet's name?'")
        print("- 'What food do I like?'")
        print("- 'Tell me about myself and what I enjoy.'")
    else:
        run_simulation(args.model, with_memory=not args.no_memory)

    print(
        "\nSimulation complete! Review the conversations above to see how MemoryWeave impacts conversation quality."
    )


if __name__ == "__main__":
    main()
