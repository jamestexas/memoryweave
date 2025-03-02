"""
BM25 baseline retriever implementation using Whoosh.
"""

import os
import tempfile
import time
from typing import Dict, List, Any, Optional, Tuple, Set

import numpy as np
from whoosh.analysis import StandardAnalyzer
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.index import create_in, open_dir
from whoosh.qparser import QueryParser
from whoosh.scoring import BM25F

from memoryweave.baselines.base import BaselineRetriever
from memoryweave.interfaces.retrieval import Query, RetrievalParameters
from memoryweave.storage.memory_store import Memory


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
            metadata=STORED
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
            writer.add_document(
                id=str(memory.id),
                content=memory.content["text"],
                metadata=memory.metadata
            )
            self.memory_lookup[str(memory.id)] = memory
            
        writer.commit()
        self.stats["index_size"] = len(self.memory_lookup)
        self.stats["indexing_time"] = time.time() - start_time
        
    def retrieve(
        self, 
        query: Query, 
        top_k: int = 10, 
        threshold: float = 0.0,
        **kwargs
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
        
        # Parse the query text
        q = parser.parse(query.text)
        
        with self.index.searcher(weighting=BM25F(B=self.b, K1=self.k1)) as searcher:
            results = searcher.search(q, limit=top_k)
            
            # Convert results to RetrievalResult format
            memories = []
            scores = []
            
            max_score = results.top_score if results and len(results) > 0 else 1.0
            
            for result in results:
                memory_id = result["id"]
                if memory_id in self.memory_lookup:
                    score = result.score / max_score  # Normalize to 0-1 range
                    
                    if score >= threshold:
                        memories.append(self.memory_lookup[memory_id])
                        scores.append(score)
            
            query_time = time.time() - start_time
            self.stats["query_times"].append(query_time)
            self.stats["avg_query_time"] = np.mean(self.stats["query_times"])
            
            parameters = {
                "max_results": top_k,
                "threshold": threshold
            }
            
            return {
                "memories": memories,
                "scores": scores,
                "strategy": self.name,
                "parameters": parameters,
                "metadata": {
                    "query_time": query_time,
                    "bm25_params": {"b": self.b, "k1": self.k1}
                }
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