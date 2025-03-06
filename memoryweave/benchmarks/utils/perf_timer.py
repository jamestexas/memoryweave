# Add these imports at the top of your benchmark script
import time
from typing import Any

from sentence_transformers import util


def compute_accuracy(response: str, expected_answers: list[str], embedder) -> dict:
    """
    Compute accuracy using multiple methods for more robust evaluation.

    Returns a dictionary with multiple accuracy scores and detailed metrics.
    """
    if not expected_answers:
        return {"combined": 0.0, "keyword": 0.0, "cosine": 0.0, "found_keywords": []}

    # 1. Cosine similarity with embedding model
    response_emb = embedder.encode(response, convert_to_tensor=True)
    expected_embs = embedder.encode(expected_answers, convert_to_tensor=True)
    cos_scores = util.cos_sim(response_emb, expected_embs)[0]
    best_score = cos_scores.max().item()

    # 2. Enhanced keyword matching
    response_lower = response.lower()
    found_keywords = []
    missed_keywords = []

    for keyword in expected_answers:
        if keyword.lower() in response_lower:
            found_keywords.append(keyword)
        else:
            missed_keywords.append(keyword)

    keyword_recall = len(found_keywords) / len(expected_answers) if expected_answers else 0

    # 3. Combined score with weighted components
    combined_score = (best_score * 0.5) + (keyword_recall * 0.5)

    return {
        "combined": combined_score,
        "keyword": keyword_recall,
        "cosine": best_score,
        "found_keywords": found_keywords,
        "missed_keywords": missed_keywords,
        "detail": f"Found {len(found_keywords)}/{len(expected_answers)} keywords, cosine: {best_score:.2f}",
    }


# Add this class for detailed timing
class PerformanceTimer:
    """Track detailed performance metrics for each benchmark operation."""

    def __init__(self):
        self.timings = {
            "retrieval": [],
            "inference": [],
            "total": [],
            "memory_ops": [],
        }
        self._start_times = {}

    def start(self, operation: str) -> None:
        """Start timing an operation."""
        self._start_times[operation] = time.time()

    def stop(self, operation: str) -> float:
        """Stop timing an operation and return elapsed time."""
        if operation not in self._start_times:
            return 0.0

        elapsed = time.time() - self._start_times[operation]
        self.timings.setdefault(operation, []).append(elapsed)
        del self._start_times[operation]
        return elapsed

    def get_average(self, operation: str) -> float:
        """Get average time for an operation."""
        times = self.timings.get(operation, [])
        return sum(times) / len(times) if times else 0.0

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics for all operations."""
        summary = {}
        for op, times in self.timings.items():
            if times:
                summary[op] = {
                    "avg": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "total": sum(times),
                    "count": len(times),
                }
        return summary
