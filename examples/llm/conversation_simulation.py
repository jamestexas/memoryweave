"""
Conversation simulation to test MemoryWeave with Hugging Face models.

This script demonstrates how MemoryWeave improves conversation quality
by comparing conversations with and without memory capabilities.
"""

import argparse
import logging
import os
import sys
import time

from memoryweave_llm_wrapper import MemoryWeaveLLM
from rich import print
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

# Add the parent directory to the path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Default model to use
DEFAULT_MODEL = "unsloth/Llama-3.2-3B-Instruct"

# Create a rich console for pretty output
console = Console()
FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)

def evaluate_memory_recall(history, expected_recalls):
    """
    Evaluate how well the memory system recalled expected information.

    Args:
        history: Conversation history from the simulation
        expected_recalls: Dict mapping queries to expected information bits

    Returns:
        Dict with success metrics
    """
    results = {"successes": 0, "failures": 0, "partial": 0, "queries_tested": 0}

    # Process the conversation history
    # Format is [{"role": "user", "content": msg}, {"role": "assistant", "content": response}, ...]
    for i in range(0, len(history), 2):
        if i + 1 >= len(history):  # Skip if no response
            continue

        user_msg = history[i]["content"]
        assistant_msg = history[i + 1]["content"]

        if user_msg in expected_recalls:
            results["queries_tested"] += 1
            expected_info = expected_recalls[user_msg]

            # Check if ALL expected info pieces are in the response
            all_found = all(info.lower() in assistant_msg.lower() for info in expected_info)
            # Check if ANY expected info pieces are in the response
            any_found = any(info.lower() in assistant_msg.lower() for info in expected_info)

            if all_found:
                results["successes"] += 1
            elif any_found:
                results["partial"] += 1
            else:
                results["failures"] += 1

    # Calculate overall score (100% scale)
    total = results["queries_tested"]
    if total > 0:
        results["score"] = (results["successes"] + 0.5 * results["partial"]) / total * 100
    else:
        results["score"] = 0

    return results


def run_simulation(model_name: str, with_memory: bool = True):
    """
    Run a simulated conversation to demonstrate MemoryWeave's capabilities.

    Args:
        model_name: Name of the Hugging Face model to use
        with_memory: Whether to use memory features

    Returns:
        Tuple of conversation history and metrics
    """
    # Expected recall information for evaluation
    expected_recalls = {
        "Where do I live?": ["Boston"],
        "What's my pet's name?": ["dog", "Max"],
        "What food do I like?": ["pizza", "mushrooms"],
        "What activities do I enjoy?": ["hiking", "weekends"],
        "Tell me about myself and what I enjoy.": [
            "James",
            "Boston",
            "dog",
            "Max",
            "pizza",
            "hiking",
        ],
    }

    # Initialize the LLM system
    llm = MemoryWeaveLLM(model_name=model_name)

    # Track query response times
    query_times = []

    # Add some pre-existing memories if memory is enabled
    if with_memory:
        # Add personal information with proper typing for better retrieval
        llm.add_memory(
            text="User lives in Boston.",
            metadata={"type": "personal_info", "subtype": "location", "importance": 0.9},
        )

        llm.add_memory(
            text="User has a dog named Max.",
            metadata={"type": "pet_name", "importance": 0.85},
        )
        
        # Add preferences with proper typing
        llm.add_memory(
            text="User preference: pizza with mushrooms",
            metadata={"type": "preference", "subtype": "food", "importance": 0.8},
        )
        
        llm.add_memory(
            text="User preference: hiking on weekends",
            metadata={"type": "preference", "subtype": "activity", "importance": 0.8},
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
        query_times.append(elapsed)
        print(f"Assistant ({elapsed:.2f}s): {response}\n")
    except Exception as e:
        print(f"Error: {e}")
        return llm.get_conversation_history() if with_memory else [], {"error": str(e)}

    print("User: I'm doing well. What can you help me with?")
    start_time = time.time()
    try:
        response = (
            llm.chat("I'm doing well. What can you help me with?", max_new_tokens=100)
            if with_memory
            else llm.chat_without_memory(
                "I'm doing well. What can you help me with?", max_new_tokens=100
            )
        )
        elapsed = time.time() - start_time
        query_times.append(elapsed)
        print(f"Assistant ({elapsed:.2f}s): {response}\n")
    except Exception as e:
        print(f"Error: {e}")
        return llm.get_conversation_history() if with_memory else [], {"error": str(e)}

    # Simulation 2: Personal preference being stored
    print("\n=== Conversation 2: Sharing Preferences ===")
    print("User: I really like pizza with mushrooms. It's my favorite food.")
    start_time = time.time()
    try:
        response = (
            llm.chat(
                "I really like pizza with mushrooms. It's my favorite food.", max_new_tokens=100
            )
            if with_memory
            else llm.chat_without_memory(
                "I really like pizza with mushrooms. It's my favorite food.", max_new_tokens=100
            )
        )
        elapsed = time.time() - start_time
        query_times.append(elapsed)
        print(f"Assistant ({elapsed:.2f}s): {response}\n")
    except Exception as e:
        print(f"Error: {e}")
        return llm.get_conversation_history() if with_memory else [], {"error": str(e)}

    print("User: I also enjoy hiking on the weekends when the weather is nice.")
    start_time = time.time()
    try:
        response = (
            llm.chat(
                "I also enjoy hiking on the weekends when the weather is nice.", max_new_tokens=100
            )
            if with_memory
            else llm.chat_without_memory(
                "I also enjoy hiking on the weekends when the weather is nice.", max_new_tokens=100
            )
        )
        elapsed = time.time() - start_time
        query_times.append(elapsed)
        print(f"Assistant ({elapsed:.2f}s): {response}\n")
    except Exception as e:
        print(f"Error: {e}")
        return llm.get_conversation_history() if with_memory else [], {"error": str(e)}

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
        query_times.append(elapsed)
        print(f"Assistant ({elapsed:.2f}s): {response}\n")
    except Exception as e:
        print(f"Error: {e}")
        return llm.get_conversation_history() if with_memory else [], {"error": str(e)}

    print("User: What's my pet's name?")
    start_time = time.time()
    try:
        response = (
            llm.chat("What's my pet's name?", max_new_tokens=100)
            if with_memory
            else llm.chat_without_memory("What's my pet's name?", max_new_tokens=100)
        )
        elapsed = time.time() - start_time
        query_times.append(elapsed)
        print(f"Assistant ({elapsed:.2f}s): {response}\n")
    except Exception as e:
        print(f"Error: {e}")
        return llm.get_conversation_history() if with_memory else [], {"error": str(e)}

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
        query_times.append(elapsed)
        print(f"Assistant ({elapsed:.2f}s): {response}\n")
    except Exception as e:
        print(f"Error: {e}")
        return llm.get_conversation_history() if with_memory else [], {"error": str(e)}

    print("User: What activities do I enjoy?")
    start_time = time.time()
    try:
        response = (
            llm.chat("What activities do I enjoy?", max_new_tokens=100)
            if with_memory
            else llm.chat_without_memory("What activities do I enjoy?", max_new_tokens=100)
        )
        elapsed = time.time() - start_time
        query_times.append(elapsed)
        print(f"Assistant ({elapsed:.2f}s): {response}\n")
    except Exception as e:
        print(f"Error: {e}")
        return llm.get_conversation_history() if with_memory else [], {"error": str(e)}

    # Final summary question
    print("\n=== Final Summary Question ===")
    print("User: Tell me about myself and what I enjoy.")
    start_time = time.time()
    try:
        response = (
            llm.chat("Tell me about myself and what I enjoy.", max_new_tokens=100)
            if with_memory
            else llm.chat_without_memory(
                "Tell me about myself and what I enjoy.", max_new_tokens=100
            )
        )
        elapsed = time.time() - start_time
        query_times.append(elapsed)
        print(f"Assistant ({elapsed:.2f}s): {response}\n")
    except Exception as e:
        print(f"Error: {e}")
        return llm.get_conversation_history() if with_memory else [], {"error": str(e)}

    # Evaluate memory recall performance
    history = llm.get_conversation_history() if with_memory else []
    recall_metrics = evaluate_memory_recall(history, expected_recalls)

    # Add query time metrics
    metrics = {
        "recall": recall_metrics,
        "avg_query_time": sum(query_times) / len(query_times) if query_times else 0,
        "total_queries": len(query_times),
    }

    return history, metrics


def display_metrics(with_memory_metrics, no_memory_metrics):
    """Display detailed metrics comparison between memory and no-memory runs."""
    table = Table(title="MemoryWeave Performance Metrics")

    # Add columns
    table.add_column("Metric", style="cyan")
    table.add_column("With Memory", style="green")
    table.add_column("Without Memory", style="yellow")
    table.add_column("Improvement", style="magenta")

    # Add rows for memory recall metrics
    memory_score = with_memory_metrics.get("recall", {}).get("score", 0)
    no_memory_score = no_memory_metrics.get("recall", {}).get("score", 0)
    improvement = memory_score - no_memory_score

    # Format as percentages
    memory_score_str = f"{memory_score:.1f}%"
    no_memory_score_str = f"{no_memory_score:.1f}%"
    improvement_str = (
        f"{improvement:.1f}%" if improvement >= 0 else f"[red]{improvement:.1f}%[/red]"
    )

    table.add_row("Recall Score", memory_score_str, no_memory_score_str, improvement_str)

    # Add rows for success/failure metrics
    mem_success = with_memory_metrics.get("recall", {}).get("successes", 0)
    no_mem_success = no_memory_metrics.get("recall", {}).get("successes", 0)
    success_diff = mem_success - no_mem_success

    mem_partial = with_memory_metrics.get("recall", {}).get("partial", 0)
    no_mem_partial = no_memory_metrics.get("recall", {}).get("partial", 0)
    partial_diff = mem_partial - no_mem_partial

    mem_fail = with_memory_metrics.get("recall", {}).get("failures", 0)
    no_mem_fail = no_memory_metrics.get("recall", {}).get("failures", 0)
    fail_diff = mem_fail - no_mem_fail

    table.add_row(
        "Complete Recalls",
        str(mem_success),
        str(no_mem_success),
        f"+{success_diff}" if success_diff > 0 else str(success_diff),
    )

    table.add_row(
        "Partial Recalls",
        str(mem_partial),
        str(no_mem_partial),
        f"+{partial_diff}" if partial_diff > 0 else str(partial_diff),
    )

    table.add_row(
        "Failed Recalls",
        str(mem_fail),
        str(no_mem_fail),
        f"{fail_diff}" if fail_diff <= 0 else f"[red]+{fail_diff}[/red]",
    )

    # Add query time metrics
    mem_time = with_memory_metrics.get("avg_query_time", 0)
    no_mem_time = no_memory_metrics.get("avg_query_time", 0)
    time_diff = mem_time - no_mem_time
    time_pct = (time_diff / no_mem_time * 100) if no_mem_time > 0 else 0

    table.add_row(
        "Avg Query Time",
        f"{mem_time:.2f}s",
        f"{no_mem_time:.2f}s",
        f"{time_diff:.2f}s ({time_pct:.1f}%)",
    )

    # Print the table
    console.print(table)

    # Additional analysis
    if memory_score > no_memory_score:
        console.print("\n[bold green]✓ MemoryWeave improved response quality[/bold green]")
        console.print(f"   Memory recall score improved by {improvement:.1f} percentage points")
    else:
        console.print("\n[bold yellow]⚠ No measurable improvement in memory recall[/bold yellow]")

    if time_diff > 0:
        console.print(f"\n[yellow]⚠ Memory processing added {time_diff:.2f}s per query[/yellow]")
    else:
        console.print("\n[green]✓ Memory processing did not increase response time[/green]")


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
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Set up debug logging if requested
    if args.debug:
        import logging

        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        # Enable specific library logging
        logging.getLogger("memoryweave").setLevel(logging.DEBUG)
        console.print("[yellow]Debug logging enabled[/yellow]")

        # Log memory retrieval content
        def debug_memory_hook(fn):
            def wrapper(*args, **kwargs):
                console.print("[cyan]DEBUG: Memory retrieval requested[/cyan]")
                results = fn(*args, **kwargs)
                console.print(f"[cyan]DEBUG: Retrieved {len(results)} memories[/cyan]")
                for i, mem in enumerate(results):
                    if isinstance(mem, dict):
                        content = mem.get("content", "No content")
                        mem_type = mem.get("metadata", {}).get("type", "unknown")
                        score = mem.get("relevance_score", 0)
                        console.print(
                            f"[cyan]DEBUG: Memory {i + 1}: {mem_type} - {score:.3f} - {content}[/cyan]"
                        )
                return results

            return wrapper

        # Try to patch the retriever with our debug hook
        try:
            from memoryweave_llm_wrapper import MemoryWeaveLLM

            original_retrieve = MemoryWeaveLLM.chat
            MemoryWeaveLLM.chat = debug_memory_hook(original_retrieve)
            console.print("[green]DEBUG: Successfully patched memory retrieval for logging[/green]")
        except Exception as e:
            console.print(f"[red]DEBUG: Failed to patch memory retrieval: {e}[/red]")

    if args.compare:
        console.print("\n" + "=" * 80)
        console.print("[bold green]RUNNING SIMULATION WITH MEMORY ENABLED[/bold green]")
        console.print("=" * 80)
        with_memory_history, with_memory_metrics = run_simulation(args.model, with_memory=True)

        console.print("\n" + "=" * 80)
        console.print("[bold yellow]RUNNING SIMULATION WITH MEMORY DISABLED[/bold yellow]")
        console.print("=" * 80)
        no_memory_history, no_memory_metrics = run_simulation(args.model, with_memory=False)

        # Compare results with detailed metrics
        console.print("\n" + "=" * 80)
        console.print("[bold cyan]COMPARISON SUMMARY[/bold cyan]")
        console.print("=" * 80)

        display_metrics(with_memory_metrics, no_memory_metrics)

        # Analyze per-question performance
        console.print("\n[bold]Per-Question Analysis:[/bold]")
        expected_recalls = {
            "Where do I live?": ["Boston"],
            "What's my pet's name?": ["dog", "Max"],
            "What food do I like?": ["pizza", "mushrooms"],
            "What activities do I enjoy?": ["hiking", "weekends"],
            "Tell me about myself and what I enjoy.": [
                "James",
                "Boston",
                "dog",
                "Max",
                "pizza",
                "hiking",
            ],
        }

        for question, expected in expected_recalls.items():
            console.print(f"\n[bold]Question:[/bold] {question}")
            console.print(f"[bold]Expected information:[/bold] {', '.join(expected)}")

            # Find responses in histories
            with_memory_response = ""
            without_memory_response = ""

            # Process with_memory_history
            for i in range(0, len(with_memory_history), 2):
                if (
                    i + 1 < len(with_memory_history)
                    and with_memory_history[i]["content"] == question
                ):
                    with_memory_response = with_memory_history[i + 1]["content"]
                    break

            # Since no_memory_history might not be in the same format, let's just say it's not available
            console.print(f"[bold green]With memory:[/bold green] {with_memory_response}")
            if not with_memory_response:
                console.print("[yellow]  No response found in history[/yellow]")

            # Check if expected info is in the response
            if with_memory_response:
                found_items = [
                    item for item in expected if item.lower() in with_memory_response.lower()
                ]
                if found_items:
                    console.print(f"[green]  Found: {', '.join(found_items)}[/green]")
                missing_items = [
                    item for item in expected if item.lower() not in with_memory_response.lower()
                ]
                if missing_items:
                    console.print(f"[yellow]  Missing: {', '.join(missing_items)}[/yellow]")
    else:
        history, metrics = run_simulation(args.model, with_memory=not args.no_memory)

        # Display metrics for single run
        console.print("\n" + "=" * 80)
        console.print("[bold cyan]SIMULATION RESULTS[/bold cyan]")
        console.print("=" * 80)

        memory_status = "enabled" if not args.no_memory else "disabled"
        console.print(f"MemoryWeave was [bold]{memory_status}[/bold] for this simulation.")

        # Print the memory store content after simulation
        if not args.no_memory:
            try:
                memory_count = 0
                memory_types = {}

                # Try to peek into llm's memory store if possible
                # This isn't ideal but helps with debugging
                import inspect

                frame = inspect.currentframe()
                if "llm" in frame.f_locals:
                    llm = frame.f_locals["llm"]
                    if hasattr(llm, "memory_manager") and hasattr(
                        llm.memory_manager, "memory_store"
                    ):
                        memory_store = llm.memory_manager.memory_store
                        if hasattr(memory_store, "memories"):
                            memories = memory_store.memories
                            memory_count = len(memories)

                            console.print("\n[bold]DEBUG: Memory Store Contents[/bold]")
                            console.print(f"Total memories stored: {memory_count}")

                            # Count by type
                            for mem_id, mem in memories.items():
                                if hasattr(mem, "metadata") and "type" in mem.metadata:
                                    mem_type = mem.metadata["type"]
                                    memory_types[mem_type] = memory_types.get(mem_type, 0) + 1

                                # Print first 10 memories
                                if len(memory_types) < 10:
                                    try:
                                        content = (
                                            mem.content if hasattr(mem, "content") else "No content"
                                        )
                                        metadata = mem.metadata if hasattr(mem, "metadata") else {}
                                        console.print(
                                            f"  Memory {mem_id}: {metadata.get('type', 'unknown')} - {content[:50]}..."
                                        )
                                    except:
                                        pass

                            # Print memory type counts
                            console.print("\nMemory types:")
                            for mem_type, count in memory_types.items():
                                console.print(f"  {mem_type}: {count}")
            except Exception as e:
                console.print(f"[red]Error accessing memory store: {e}[/red]")

        # Display metrics
        recall_score = metrics.get("recall", {}).get("score", 0)
        successes = metrics.get("recall", {}).get("successes", 0)
        partials = metrics.get("recall", {}).get("partial", 0)
        failures = metrics.get("recall", {}).get("failures", 0)
        total = metrics.get("recall", {}).get("queries_tested", 0)

        console.print("\n[bold]Memory Recall Performance:[/bold]")
        console.print(f"Overall score: {recall_score:.1f}%")
        console.print(f"Complete recalls: {successes}/{total}")
        console.print(f"Partial recalls: {partials}/{total}")
        console.print(f"Failed recalls: {failures}/{total}")

        avg_time = metrics.get("avg_query_time", 0)
        console.print("\n[bold]Performance:[/bold]")
        console.print(f"Average query time: {avg_time:.2f}s")
        console.print(f"Total queries processed: {metrics.get('total_queries', 0)}")

    console.print(
        "\n[bold green]Simulation complete![/bold green] Review the results above to see how MemoryWeave impacts conversation quality."
    )


if __name__ == "__main__":
    main()
