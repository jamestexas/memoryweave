"""
Metrics for evaluating conversational coherence and memory effectiveness.
"""

from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def coherence_score(conversation: list[dict], embedding_model: Any, window_size: int = 3) -> float:
    """
    Calculate the overall coherence score for a conversation.

    Args:
        conversation: list of conversation turns with messages and responses
        embedding_model: Model to create embeddings for coherence analysis
        window_size: Size of sliding window for coherence calculation

    Returns:
        Coherence score between 0 and 1
    """
    if len(conversation) < 2:
        return 1.0  # Single turn is trivially coherent

    # Get embeddings for each turn
    turn_embeddings = []
    for turn in conversation:
        # Combine message and response for a turn
        message = turn.get("message", "")
        response = turn.get("response", "")
        combined = f"{message} {response}".strip()

        # Get embedding
        embedding = embedding_model.encode(combined)
        turn_embeddings.append(embedding)

    turn_embeddings = np.array(turn_embeddings)

    # Calculate coherence using sliding window
    coherence_scores = []
    for i in range(len(turn_embeddings) - window_size + 1):
        window = turn_embeddings[i : i + window_size]
        # Calculate pairwise similarities within window
        sim_matrix = cosine_similarity(window)

        # Average the off-diagonal elements (pairwise similarities)
        n = sim_matrix.shape[0]
        total_sim = sim_matrix.sum() - n  # Subtract diagonal elements
        avg_sim = total_sim / (n * (n - 1))  # Divide by number of pairs

        coherence_scores.append(avg_sim)

    # Return average coherence
    return float(np.mean(coherence_scores)) if coherence_scores else 1.0


def context_relevance(
    retrieved_memories: list[dict], current_query: str, embedding_model: Any
) -> float:
    """
    Measure how relevant the retrieved memories are to the current query.

    Args:
        retrieved_memories: list of memory entries retrieved by the system
        current_query: The current user query
        embedding_model: Model for creating embeddings

    Returns:
        Relevance score between 0 and 1
    """
    if not retrieved_memories:
        return 0.0

    query_embedding = embedding_model.encode(current_query)

    # Get embeddings for memory content
    memory_embeddings = []
    for memory in retrieved_memories:
        content = memory.get("text", "") or memory.get("content", "")
        if content:
            embedding = embedding_model.encode(content)
            memory_embeddings.append(embedding)

    if not memory_embeddings:
        return 0.0

    # Calculate cosine similarities with query
    memory_embeddings = np.array(memory_embeddings)
    similarities = cosine_similarity([query_embedding], memory_embeddings)[0]

    # Return weighted average (higher weight to more similar memories)
    weights = similarities / similarities.sum()
    weighted_avg = float(np.sum(similarities * weights))

    return weighted_avg


def response_consistency(
    response: str, retrieved_memories: list[dict], embedding_model: Any
) -> float:
    """
    Measure how consistent the response is with the retrieved memories.

    Args:
        response: The model's response
        retrieved_memories: list of memory entries retrieved by the system
        embedding_model: Model for creating embeddings

    Returns:
        Consistency score between 0 and 1
    """
    if not retrieved_memories:
        return 1.0  # No memories means nothing to be inconsistent with

    response_embedding = embedding_model.encode(response)

    # Create combined memory content
    memory_texts = []
    for memory in retrieved_memories:
        content = memory.get("text", "") or memory.get("content", "")
        if content:
            memory_texts.append(content)

    if not memory_texts:
        return 1.0

    combined_memory = " ".join(memory_texts)
    memory_embedding = embedding_model.encode(combined_memory)

    # Calculate cosine similarity
    similarity = cosine_similarity([response_embedding], [memory_embedding])[0][0]

    return float(similarity)


def evaluate_conversation(
    conversation: list[dict], memory_system: dict, embedding_model: Any
) -> dict[str, float]:
    """
    Comprehensive evaluation of a conversation using memory augmentation.

    Args:
        conversation: list of conversation turns
        memory_system: The memory system components
        embedding_model: Model for creating embeddings

    Returns:
        dictionary of evaluation metrics
    """
    retriever = memory_system.get("retriever")
    if not retriever:
        raise ValueError("Memory system must include a retriever")

    # Overall coherence score
    overall_coherence = coherence_score(conversation, embedding_model)

    # Per-turn metrics
    turn_scores = []
    for i, turn in enumerate(conversation):
        if i == 0:  # Skip first turn (no context yet)
            continue

        query = turn.get("message", "")
        response = turn.get("response", "")

        # Get conversation history up to this point
        history = conversation[:i]

        # Retrieve memories for this context
        retrieved_memories = retriever.retrieve_for_context(query, history, top_k=5)

        # Calculate metrics
        relevance = context_relevance(retrieved_memories, query, embedding_model)
        consistency = response_consistency(response, retrieved_memories, embedding_model)

        turn_scores.append({"turn": i, "relevance": relevance, "consistency": consistency})

    # Aggregate metrics
    avg_relevance = np.mean([s["relevance"] for s in turn_scores]) if turn_scores else 0.0
    avg_consistency = np.mean([s["consistency"] for s in turn_scores]) if turn_scores else 0.0

    return {
        "overall_coherence": overall_coherence,
        "average_relevance": float(avg_relevance),
        "average_consistency": float(avg_consistency),
        "turn_scores": turn_scores,
    }
