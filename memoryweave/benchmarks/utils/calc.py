import logging
from dataclasses import dataclass, field

import numpy as np
from rich.console import Console
from rich.logging import RichHandler

# Configure rich logging
console = Console(highlight=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(show_path=False, rich_tracebacks=True, console=console)],
)
logger = logging.getLogger("contextual_recall")

__all__ = [
    "RetrievalMetrics",
    "calculate_recall_at_k",
    "calculate_mrr",
    "calculate_ndcg",
    "calculate_temporal_accuracy",
    "calculate_contextual_relevance",
]


@dataclass
class RetrievalMetrics:
    """Holds metrics for retrieval evaluation."""

    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    temporal_accuracy: float = 0.0
    contextual_relevance: float = 0.0

    query_count: int = 0
    avg_retrieval_time: float = 0.0
    avg_inference_time: float = 0.0

    # Additional data for detailed analysis
    per_query_results: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        result = {k: v for k, v in self.__dict__.items() if k != "per_query_results"}
        result["per_query_results"] = {
            query_id: {k: v for k, v in query_data.items() if not callable(v)}
            for query_id, query_data in self.per_query_results.items()
        }
        return result


def calculate_recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Calculate recall@k with improved ID matching."""
    if not relevant_ids:
        return 0.0

    # Get top-k retrieved IDs
    top_k_ids = retrieved_ids[:k]

    # Create normalized versions of relevant IDs for matching
    normalized_relevant = set()
    for rel_id in relevant_ids:
        # Add original ID
        normalized_relevant.add(rel_id)
        # Add just the numeric part if it exists
        if "_" in rel_id:
            normalized_relevant.add(rel_id.split("_")[-1])
        # Add as int if it's numeric
        if rel_id.isdigit():
            normalized_relevant.add(int(rel_id))

    # Calculate intersection using normalized IDs
    found = set()
    for item_id in top_k_ids:
        # Try string and int versions of the ID
        str_id = str(item_id)
        int_id = int(item_id) if str_id.isdigit() else None

        if str_id in normalized_relevant or int_id in normalized_relevant:
            found.add(str_id)
            continue

        # Handle prefixed IDs (e.g., "conv_3")
        if "_" in str_id:
            base_id = str_id.split("_")[-1]
            if base_id in normalized_relevant or (
                base_id.isdigit() and int(base_id) in normalized_relevant
            ):
                found.add(str_id)

    # Calculate recall
    return len(found) / len(relevant_ids)


def calculate_mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Calculate Mean Reciprocal Rank with improved ID matching."""
    if not relevant_ids:
        return 0.0

    # Create normalized versions of relevant IDs for matching
    normalized_relevant = set()
    for rel_id in relevant_ids:
        # Add original ID
        normalized_relevant.add(rel_id)
        # Add just the numeric part if it exists
        if isinstance(rel_id, str) and "_" in rel_id:
            normalized_relevant.add(rel_id.split("_")[-1])
        # Add as int if it's numeric
        if isinstance(rel_id, str) and rel_id.isdigit():
            normalized_relevant.add(int(rel_id))

    # Find the first relevant ID in the retrieved list
    for i, item_id in enumerate(retrieved_ids):
        # Try string and int versions of the ID
        str_id = str(item_id)
        int_id = int(item_id) if str_id.isdigit() else None

        if str_id in normalized_relevant or int_id in normalized_relevant:
            return 1.0 / (i + 1)

        # Handle prefixed IDs (e.g., "conv_3")
        if isinstance(str_id, str) and "_" in str_id:
            base_id = str_id.split("_")[-1]
            if base_id in normalized_relevant or (
                base_id.isdigit() and int(base_id) in normalized_relevant
            ):
                return 1.0 / (i + 1)

    # No relevant items found
    return 0.0


def calculate_ndcg(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    """Calculate Normalized Discounted Cumulative Gain."""
    if not relevant_ids:
        return 0.0

    # Get top-k retrieved IDs
    top_k_ids = retrieved_ids[:k]

    # Calculate DCG
    dcg = 0.0
    for i, item_id in enumerate(top_k_ids):
        if item_id in relevant_ids:
            # Use binary relevance (1 if relevant, 0 if not)
            dcg += 1.0 / np.log2(i + 2)  # +2 because log_2(1) = 0

    # Calculate ideal DCG
    ideal_rankings = list(relevant_ids)[:k]
    idcg = sum(1.0 / np.log2(i + 2) for i in range(len(ideal_rankings)))

    # Prevent division by zero
    if idcg == 0:
        return 0.0

    return dcg / idcg


def calculate_temporal_accuracy(
    retrieved_ids: list[str], relevant_ids: set[str], temporal_relevance: dict[str, float]
) -> float:
    """
    Calculate how well the retrieval system handles temporal references.

    Args:
        retrieved_ids: list of retrieved memory IDs
        relevant_ids: set of relevant memory IDs
        temporal_relevance: dictionary mapping memory IDs to temporal relevance scores

    Returns:
        Temporal accuracy score (0.0-1.0)
    """
    if not relevant_ids or not temporal_relevance:
        return 0.0

    # Get temporal relevance scores for retrieved items
    retrieved_scores = [temporal_relevance.get(item_id, 0.0) for item_id in retrieved_ids[:5]]

    # Get ideal scores (for relevant items)
    ideal_scores = sorted(
        [temporal_relevance.get(item_id, 0.0) for item_id in relevant_ids], reverse=True
    )[:5]

    # Calculate normalized temporal score
    retrieved_sum = sum(retrieved_scores)
    ideal_sum = sum(ideal_scores)

    if ideal_sum == 0:
        return 0.0

    return retrieved_sum / ideal_sum


def calculate_contextual_relevance(
    query: str, retrieved_texts: list[str], conversation_history: list[dict], embedding_model
) -> float:
    """
    Calculate how well the retrieved texts fit the broader conversation context.

    Args:
        query: Current query
        retrieved_texts: list of retrieved memory texts
        conversation_history: Previous conversation turns
        embedding_model: Model for computing embeddings

    Returns:
        Contextual relevance score (0.0-1.0)
    """
    if not retrieved_texts or not conversation_history:
        return 0.0

    # Combine conversation history into context
    history_texts = [turn.get("text", "") for turn in conversation_history[-3:]]
    context = " ".join(history_texts + [query])

    # Generate embeddings
    try:
        context_embedding = embedding_model.encode(context)
        retrieved_embeddings = [embedding_model.encode(text) for text in retrieved_texts[:5]]

        # Calculate cosine similarities
        scores = []
        for emb in retrieved_embeddings:
            # Normalize
            context_norm = np.linalg.norm(context_embedding)
            emb_norm = np.linalg.norm(emb)

            if context_norm > 0 and emb_norm > 0:
                similarity = np.dot(context_embedding, emb) / (context_norm * emb_norm)
                scores.append(similarity)
            else:
                scores.append(0.0)

        # Return average contextual relevance
        return sum(scores) / len(scores) if scores else 0.0

    except Exception as e:
        logger.error(f"Error calculating contextual relevance: {e}")
        return 0.0
