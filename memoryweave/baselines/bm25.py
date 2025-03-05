"""
BM25 baseline retriever implementation using Whoosh.
"""

import os
import tempfile
import time
from typing import Any, Dict, List, Optional

import numpy as np
from whoosh.analysis import StandardAnalyzer
from whoosh.fields import ID, STORED, TEXT, Schema
from whoosh.index import create_in
from whoosh.qparser import QueryParser
from whoosh.scoring import BM25F

from memoryweave.baselines.base import BaselineRetriever
from memoryweave.interfaces.retrieval import Query
from memoryweave.interfaces.memory import Memory


class BM25Retriever(BaselineRetriever):
    """BM25 retrieval baseline using Whoosh search engine.

    This retriever implements the industry-standard BM25 algorithm
    for text search, which is a strong non-neural baseline for
    information retrieval tasks.
    """

    name: str = "bm25"

    def __init__(self, b: float = 0.75, k1: float = 1.2, persist_path: Optional[str] = None):
        """Initialize BM25 retriever.

        Args:
            b: BM25 length normalization parameter (0.0-1.0)
            k1: BM25 term frequency scaling parameter
            persist_path: Path to store the index, if None a temporary directory is used
        """
        self.b = b
        self.k1 = k1

        self.analyzer = StandardAnalyzer()
        self.schema = Schema(
            id=ID(stored=True, unique=True),
            content=TEXT(analyzer=self.analyzer, stored=True),
            metadata=STORED,
        )

        if persist_path:
            self.index_dir = persist_path
            os.makedirs(self.index_dir, exist_ok=True)
        else:
            self.temp_dir = tempfile.TemporaryDirectory()
            self.index_dir = self.temp_dir.name

        self.index = create_in(self.index_dir, self.schema)
        self.memory_lookup: Dict[str, Memory] = {}
        self.stats = {
            "index_size": 0,
            "query_times": [],
            "avg_query_time": 0,
        }

    def index_memories(self, memories: List[Memory]) -> None:
        """Index a list of memories using BM25.

        Args:
            memories: List of memories to index
        """
        start_time = time.time()
        writer = self.index.writer()

        for memory in memories:
            # Store memory text and ID in the index
            memory_text = ""
            if isinstance(memory.content, dict) and "text" in memory.content:
                memory_text = memory.content["text"]
            elif hasattr(memory, "text"):
                memory_text = memory.text

            # Ensure we have text to index
            if not memory_text:
                print(f"Warning: Memory {memory.id} has no text content to index")
                memory_text = " "  # Add a space to prevent indexing errors

            writer.add_document(id=str(memory.id), content=memory_text, metadata=memory.metadata)
            self.memory_lookup[str(memory.id)] = memory

        writer.commit()
        self.stats["index_size"] = len(self.memory_lookup)
        self.stats["indexing_time"] = time.time() - start_time

    def retrieve(
        self, query: Query, top_k: int = 10, threshold: float = 0.0, **kwargs
    ) -> Dict[str, Any]:
        """Retrieve memories using BM25.

        Args:
            query: The query to search for
            top_k: Maximum number of results to return
            threshold: Minimum score threshold (normalized 0-1)
            **kwargs: Additional parameters (ignored)

        Returns:
            RetrievalResult containing matched memories
        """
        start_time = time.time()

        # Create query parser for the content field
        parser = QueryParser("content", schema=self.index.schema)

        # Add logging for debugging
        print(f"BM25Retriever: Query text: '{query.text}'")
        print(f"BM25Retriever: Index size: {self.stats['index_size']} memories")

        # Parse the query text
        query_text = query.text

        # Ensure the query text isn't empty to prevent parsing errors
        if not query_text or query_text.strip() == "":
            query_text = " "

        # Clean query text to avoid Whoosh parsing errors
        # Replace special characters with spaces and escape or remove syntax elements
        import re

        # Replace characters that might cause parser errors
        cleaned_query = re.sub(r'[?!&|\'"-:;.,()~/*]', " ", query_text)
        # Replace multiple spaces with single space
        cleaned_query = re.sub(r"\s+", " ", cleaned_query).strip()

        print(f"BM25Retriever: Cleaned query: '{cleaned_query}'")

        # Use keywords instead if available (often safer for search)
        if hasattr(query, "extracted_keywords") and query.extracted_keywords:
            keywords_query = " OR ".join(query.extracted_keywords)
            print(f"BM25Retriever: Using keywords query: '{keywords_query}'")
            cleaned_query = keywords_query

        try:
            # Use keyword search with OR operator for more relaxed matching
            if cleaned_query.strip():
                q = parser.parse(cleaned_query)
            else:
                # If we have nothing left after cleaning, return empty results
                return {
                    "memories": [],
                    "scores": [],
                    "strategy": self.name,
                    "parameters": {"max_results": top_k, "threshold": threshold},
                    "metadata": {
                        "query_time": 0.0,
                        "error": "Empty query after cleaning",
                        "bm25_params": {"b": self.b, "k1": self.k1},
                    },
                }
        except Exception as e:
            print(f"Error parsing query '{cleaned_query}': {e}")
            # Return empty results on parsing error
            return {
                "memories": [],
                "scores": [],
                "strategy": self.name,
                "parameters": {"max_results": top_k, "threshold": threshold},
                "metadata": {
                    "query_time": 0.0,
                    "error": str(e),
                    "bm25_params": {"b": self.b, "k1": self.k1},
                },
            }

        with self.index.searcher(weighting=BM25F(B=self.b, K1=self.k1)) as searcher:
            results = searcher.search(q, limit=top_k)

            # Convert results to RetrievalResult format
            memories = []
            scores = []

            # Check if results exist and get the max score
            max_score = 1.0
            if results and len(results) > 0:
                if hasattr(results, "top_score"):
                    max_score = results.top_score
                elif hasattr(results, "score") and len(results) > 0:
                    # If top_score doesn't exist, try to find the highest score from individual results
                    max_score = max([r.score for r in results]) if len(results) > 0 else 1.0
                else:
                    # Default max_score if neither is available
                    max_score = 1.0

            print(f"BM25Retriever: Found {len(results)} results with max_score={max_score}")

            for result in results:
                memory_id = result["id"]
                if memory_id in self.memory_lookup:
                    # Get result score with error handling
                    if hasattr(result, "score"):
                        score = result.score
                    else:
                        # If no score attribute, try to access it as a dictionary item
                        score = result.get("score", 0.0)

                    # Normalize to 0-1 range
                    score = score / max_score if max_score > 0 else 0.0

                    if score >= threshold:
                        memories.append(self.memory_lookup[memory_id])
                        scores.append(score)

            query_time = time.time() - start_time
            self.stats["query_times"].append(query_time)
            self.stats["avg_query_time"] = np.mean(self.stats["query_times"])

            parameters = {"max_results": top_k, "threshold": threshold}

            return {
                "memories": memories,
                "scores": scores,
                "strategy": self.name,
                "parameters": parameters,
                "metadata": {"query_time": query_time, "bm25_params": {"b": self.b, "k1": self.k1}},
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get retrieval statistics for BM25.

        Returns:
            Dictionary of index statistics
        """
        return self.stats

    def clear(self) -> None:
        """Clear the index."""
        if os.path.exists(self.index_dir):
            self.index = create_in(self.index_dir, self.schema)
        self.memory_lookup = {}
        self.stats = {
            "index_size": 0,
            "query_times": [],
            "avg_query_time": 0,
        }
