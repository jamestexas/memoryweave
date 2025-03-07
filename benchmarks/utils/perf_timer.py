# Add these imports at the top of your benchmark script
import functools
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


class PerformanceTimer:
    """Track detailed performance metrics for benchmark operations."""

    def __init__(self):
        self.timings = {}
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


# Global timer for use with decorators
_global_timer = PerformanceTimer()


def get_global_timer() -> PerformanceTimer:
    """Get the global timer instance for reporting."""
    return _global_timer


def timer(operation: str = None, timer_instance: PerformanceTimer = None):
    """
    Decorator to time operations. Can be used on methods or classes.

    When used on a class, it adds a timer instance to the class.
    When used on a method, it times the method execution.

    Args:
        operation: Name of the operation to time (only for method decoration)
        timer_instance: Optional timer instance to use instead of the global one
    """

    def _get_timer_instance(instance=None):
        """Get the timer instance to use based on context."""
        # If a timer instance was explicitly provided, use it
        if timer_instance is not None:
            return timer_instance

        # If decorating a method of a class that has a timer, use that
        if (
            instance is not None
            and hasattr(instance, "timer")
            and isinstance(instance.timer, PerformanceTimer)
        ):
            return instance.timer

        # Otherwise use the global timer
        return _global_timer

    def class_decorator(cls):
        """Add a timer instance to the class."""
        original_init = cls.__init__

        @functools.wraps(original_init)
        def init_with_timer(self, *args, **kwargs):
            # Add timer instance to the class
            self.timer = PerformanceTimer()
            # Call the original __init__
            original_init(self, *args, **kwargs)

        # Replace __init__ with our version
        cls.__init__ = init_with_timer
        return cls

    def method_decorator(func):
        """Time a method execution."""

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Get the appropriate timer
            timer = _get_timer_instance(self)

            # Use the function name if no operation name provided
            op_name = operation or func.__name__

            # Start timing
            timer.start(op_name)
            try:
                # Run the function
                result = func(self, *args, **kwargs)
                return result
            finally:
                # Stop timing even if an exception occurs
                timer.stop(op_name)

        return wrapper

    # Handle both class and method decoration
    def decorator(obj):
        if isinstance(obj, type):
            # It's a class - add a timer to it
            return class_decorator(obj)
        else:
            # It's a function/method - time it
            return method_decorator(obj)

    # Handle when called with or without arguments
    if callable(operation):
        # Called as @timer without parentheses
        return decorator(operation)
    else:
        # Called as @timer() or @timer("name")
        return decorator


# For direct method timing without decoration
def time_operation(operation_name, func, *args, timer_obj=None, **kwargs):
    """Time a function call directly without decoration."""
    t = timer_obj or _global_timer
    t.start(operation_name)
    try:
        result = func(*args, **kwargs)
        return result
    finally:
        t.stop(operation_name)
