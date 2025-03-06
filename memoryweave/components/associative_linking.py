# memoryweave/components/associative_linking.py
"""
Associative memory linking component for MemoryWeave.

This module implements components for creating and traversing "fabric" connections
between semantically related memories. These connections enable associative memory
retrieval that goes beyond pure similarity matching.
"""

import heapq
import time
from collections import defaultdict
from typing import Any, Optional

import numpy as np

from memoryweave.components.base import Component, MemoryComponent
from memoryweave.components.component_names import ComponentName
from memoryweave.interfaces.memory import MemoryID
from memoryweave.storage.refactored.base_store import BaseMemoryStore


class AssociativeMemoryLinker(MemoryComponent):
    """
    Component that establishes and maintains associative links between memories.

    This component creates the "fabric" structure of MemoryWeave by:
    1. Finding semantically related memories
    2. Establishing bidirectional links between related memories
    3. Calculating link strength based on similarity and temporal proximity
    4. Enabling associative traversal during retrieval

    The associative fabric enables retrieval beyond direct matches, allowing for
    multi-hop connections and cognitive-inspired memory access patterns.
    """

    def __init__(self, memory_store: Optional[BaseMemoryStore] = None):
        """
        Initialize the associative memory linker.

        Args:
            memory_store: Optional memory store to link memories from
        """
        self.memory_store = memory_store
        self.similarity_threshold = 0.5
        self.temporal_weight = 0.3
        self.semantic_weight = 0.7
        self.max_links_per_memory = 10
        self.rebuild_frequency = 100  # Rebuild full graph every N new memories
        self.memory_count = 0
        self.component_id = ComponentName.ASSOCIATIVE_MEMORY_LINKER

        # Store links as adjacency list: {memory_id: [(linked_id, strength), ...]}
        self.associative_links: dict[MemoryID, list[tuple[MemoryID, float]]] = defaultdict(list)

        # Track last full rebuild
        self.last_rebuild_time = 0
        self.memories_since_rebuild = 0

    def initialize(self, config: dict[str, Any]) -> None:
        """
        Initialize the component with configuration.

        Args:
            config: Configuration dictionary with parameters:
                - similarity_threshold: Minimum similarity for creating links (default: 0.5)
                - temporal_weight: Weight for temporal proximity in link strength (default: 0.3)
                - semantic_weight: Weight for semantic similarity in link strength (default: 0.7)
                - max_links_per_memory: Maximum number of links per memory (default: 10)
                - rebuild_frequency: How often to rebuild the entire link structure (default: 100)
        """
        self.similarity_threshold = config.get("similarity_threshold", 0.5)
        self.temporal_weight = config.get("temporal_weight", 0.3)
        self.semantic_weight = config.get("semantic_weight", 0.7)
        self.max_links_per_memory = config.get("max_links_per_memory", 10)
        self.rebuild_frequency = config.get("rebuild_frequency", 100)

        # Set memory store if provided
        if "memory_store" in config:
            self.memory_store = config["memory_store"]

        # Build initial link structure if memory store is available
        if self.memory_store is not None:
            self._rebuild_all_links()

    def process(self, data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """
        Process memory data to establish associative links.

        Args:
            data: Memory data to process
            context: Context information

        Returns:
            Updated memory data with associative links
        """
        # Extract memory ID
        memory_id = data.get("id")
        if memory_id is None:
            return data

        # Create a deep copy of the data to avoid modifying the original
        result = dict(data)

        # Get the memory store from context if not already set
        if self.memory_store is None and "memory_store" in context:
            self.memory_store = context["memory_store"]

        # If no memory store available, return original data
        if self.memory_store is None:
            return data

        # Find and establish links for this memory
        self._establish_links_for_memory(memory_id, data, context)

        # Increment memory count
        self.memory_count += 1
        self.memories_since_rebuild += 1

        # Check if we need to rebuild all links
        if self.memories_since_rebuild >= self.rebuild_frequency:
            self._rebuild_all_links()
            self.memories_since_rebuild = 0

        # Add link information to the result
        result["associative_links"] = self.associative_links.get(memory_id, [])

        return result

    def _establish_links_for_memory(
        self, memory_id: MemoryID, memory_data: dict[str, Any], context: dict[str, Any]
    ) -> None:
        """
        Establish associative links for a single memory.

        Args:
            memory_id: ID of the memory to establish links for
            memory_data: Data for the memory
            context: Context information
        """
        # Extract embedding and creation time
        embedding = memory_data.get("embedding")
        creation_time = memory_data.get("created_at", time.time())

        if embedding is None or self.memory_store is None:
            return

        # Get all existing memories
        all_memories = self.memory_store.get_all()

        # Calculate similarity and link strength for each memory
        candidate_links = []

        for other_memory in all_memories:
            # Skip self
            if other_memory.id == memory_id:
                continue

            # Calculate semantic similarity
            other_embedding = other_memory.embedding
            semantic_similarity = self._calculate_similarity(embedding, other_embedding)

            # Skip if below threshold
            if semantic_similarity < self.similarity_threshold:
                continue

            # Calculate temporal proximity (normalize to 0-1)
            other_creation_time = other_memory.metadata.get("created_at", 0)
            if other_creation_time > 0:
                time_diff = abs(creation_time - other_creation_time)
                # Normalize with a 1-day scale
                normalized_time_diff = min(1.0, time_diff / (86400 * 7))  # 7-day scale
                temporal_proximity = 1.0 - normalized_time_diff
            else:
                temporal_proximity = 0.0

            # Calculate overall link strength
            link_strength = (
                self.semantic_weight * semantic_similarity
                + self.temporal_weight * temporal_proximity
            )

            # Add to candidate links
            candidate_links.append((other_memory.id, link_strength))

        # Sort candidate links by strength (descending)
        candidate_links.sort(key=lambda x: x[1], reverse=True)

        # Take the top N links
        top_links = candidate_links[: self.max_links_per_memory]

        # Store links in both directions (bidirectional)
        self.associative_links[memory_id] = top_links

        # Add reverse links
        for other_id, strength in top_links:
            # Check if link already exists
            existing_links = self.associative_links.get(other_id, [])
            exists = False

            for i, (linked_id, _) in enumerate(existing_links):
                if linked_id == memory_id:
                    # Update existing link strength
                    existing_links[i] = (memory_id, strength)
                    exists = True
                    break

            if not exists:
                # Add new link
                existing_links.append((memory_id, strength))

                # Sort and limit size
                existing_links.sort(key=lambda x: x[1], reverse=True)
                self.associative_links[other_id] = existing_links[: self.max_links_per_memory]

    def _rebuild_all_links(self) -> None:
        """Rebuild all associative links from scratch."""
        if self.memory_store is None:
            return

        # Clear existing links
        self.associative_links.clear()

        # Get all memories
        all_memories = self.memory_store.get_all()
        self.memory_count = len(all_memories)

        # Build embeddings matrix for fast computation
        embeddings = []
        ids = []
        creation_times = []

        for memory in all_memories:
            embeddings.append(memory.embedding)
            ids.append(memory.id)
            creation_times.append(memory.metadata.get("created_at", 0))

        # Convert to numpy arrays
        embeddings_matrix = np.array(embeddings)

        # Calculate pairwise similarities
        similarities = np.dot(embeddings_matrix, embeddings_matrix.T)

        # Process each memory's links
        for i, memory_id in enumerate(ids):
            # Get similarities for this memory
            memory_similarities = similarities[i]

            # Find indices of memories above threshold (excluding self)
            candidate_indices = np.where(
                (memory_similarities >= self.similarity_threshold) & (np.arange(len(ids)) != i)
            )[0]

            # Calculate link strengths
            links = []

            for j in candidate_indices:
                # Calculate temporal proximity
                time_i = creation_times[i]
                time_j = creation_times[j]

                if time_i > 0 and time_j > 0:
                    time_diff = abs(time_i - time_j)
                    # Normalize with a 1-day scale
                    normalized_time_diff = min(1.0, time_diff / (86400 * 7))  # 7-day scale
                    temporal_proximity = 1.0 - normalized_time_diff
                else:
                    temporal_proximity = 0.0

                # Calculate overall link strength
                semantic_similarity = float(memory_similarities[j])
                link_strength = (
                    self.semantic_weight * semantic_similarity
                    + self.temporal_weight * temporal_proximity
                )

                links.append((ids[j], link_strength))

            # Sort by strength and take top N
            links.sort(key=lambda x: x[1], reverse=True)
            self.associative_links[memory_id] = links[: self.max_links_per_memory]

        # Update rebuild time
        self.last_rebuild_time = time.time()
        self.memories_since_rebuild = 0

    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score (0-1)
        """
        # Calculate cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def get_associative_links(self, memory_id: MemoryID) -> list[tuple[MemoryID, float]]:
        """
        Get associative links for a memory.

        Args:
            memory_id: ID of the memory to get links for

        Returns:
            list of (memory_id, strength) tuples for linked memories
        """
        return self.associative_links.get(memory_id, [])

    def traverse_associative_network(
        self, start_id: MemoryID, max_hops: int = 2, min_strength: float = 0.3
    ) -> dict[MemoryID, float]:
        """
        Traverse the associative network from a starting memory.

        Args:
            start_id: Starting memory ID
            max_hops: Maximum number of hops to traverse
            min_strength: Minimum link strength to follow

        Returns:
            dictionary mapping memory IDs to activation levels
        """
        # Initialize results with starting node
        results: dict[MemoryID, float] = {start_id: 1.0}

        # Use a queue to track nodes to visit
        # Format: (memory_id, current_hop, current_strength)
        queue = [(start_id, 0, 1.0)]
        visited = {start_id}

        while queue:
            current_id, hop, strength = queue.pop(0)

            # If we've reached the maximum hop count, stop traversing
            if hop >= max_hops:
                continue

            # Get links for current memory
            links = self.associative_links.get(current_id, [])

            # Process each link
            for linked_id, link_strength in links:
                # Calculate cumulative strength (decays with distance)
                cumulative_strength = strength * link_strength

                # Skip if below minimum strength
                if cumulative_strength < min_strength:
                    continue

                # Add or update result
                if linked_id in results:
                    # Take the maximum strength path
                    results[linked_id] = max(results[linked_id], cumulative_strength)
                else:
                    results[linked_id] = cumulative_strength

                # Add to queue if not already visited
                if linked_id not in visited:
                    visited.add(linked_id)
                    queue.append((linked_id, hop + 1, cumulative_strength))

        return results

    def find_path_between_memories(
        self, source_id: MemoryID, target_id: MemoryID, max_hops: int = 3
    ) -> list[tuple[MemoryID, float]]:
        """
        Find a path between two memories in the associative network.

        Args:
            source_id: Starting memory ID
            target_id: Target memory ID
            max_hops: Maximum number of hops to search

        Returns:
            list of (memory_id, strength) tuples representing the path,
            or empty list if no path found
        """
        # Use A* search to find the path
        # Priority queue: (cumulative cost, memory_id, path)
        frontier = [(0, source_id, [(source_id, 1.0)])]
        visited = {source_id}

        while frontier:
            _, current_id, path = heapq.heappop(frontier)

            # If we found the target, return the path
            if current_id == target_id:
                return path

            # If we've reached the maximum hop count, skip
            if len(path) > max_hops:
                continue

            # Get links for current memory
            links = self.associative_links.get(current_id, [])

            # Process each link
            for linked_id, link_strength in links:
                # Skip if already visited
                if linked_id in visited:
                    continue

                # Create new path
                new_path = path + [(linked_id, link_strength)]

                # Calculate path cost (inverse of link strength product)
                path_product = 1.0
                for _, strength in new_path:
                    path_product *= strength

                # Use A* heuristic: path cost + estimate to goal
                # For the estimate, we use 0 for simplicity (becomes Dijkstra's)
                cost = -path_product  # Negative because heapq is min-heap

                # Add to frontier
                heapq.heappush(frontier, (cost, linked_id, new_path))
                visited.add(linked_id)

        # No path found
        return []

    def create_associative_link(
        self,
        source_id: str,
        target_id: str,
        strength: float = 0.5,
    ) -> None:
        """
        Create an associative link between two memories.

        Args:
            source_id: Source memory ID
            target_id: Target memory ID
            strength: Link strength (0-1)
        """
        # Convert to appropriate storage format
        if not isinstance(source_id, str):
            source_id = str(source_id)
        if not isinstance(target_id, str):
            target_id = str(target_id)

        # Create links in both directions with appropriate strengths
        source_links = self.associative_links.get(source_id, [])
        target_links = self.associative_links.get(target_id, [])

        # Update or add source → target link
        source_target_exists = False
        for i, (existing_id, existing_strength) in enumerate(source_links):
            if existing_id == target_id:
                # Update existing link with higher strength
                source_links[i] = (target_id, max(existing_strength, strength))
                source_target_exists = True
                break

        if not source_target_exists:
            source_links.append((target_id, strength))

        # Update or add target → source link (with reduced strength)
        target_source_exists = False
        reduced_strength = max(0.1, strength * 0.8)  # Slightly weaker in reverse

        for i, (existing_id, existing_strength) in enumerate(target_links):
            if existing_id == source_id:
                # Update existing link with higher strength
                target_links[i] = (source_id, max(existing_strength, reduced_strength))
                target_source_exists = True
                break

        if not target_source_exists:
            target_links.append((source_id, reduced_strength))

        # Store updated links
        self.associative_links[source_id] = source_links
        self.associative_links[target_id] = target_links

        # Ensure we don't exceed max links per memory
        if hasattr(self, "max_links_per_memory") and self.max_links_per_memory > 0:
            if len(source_links) > self.max_links_per_memory:
                # Sort by strength and keep only the strongest links
                source_links.sort(key=lambda x: x[1], reverse=True)
                self.associative_links[source_id] = source_links[: self.max_links_per_memory]

            if len(target_links) > self.max_links_per_memory:
                # Sort by strength and keep only the strongest links
                target_links.sort(key=lambda x: x[1], reverse=True)
                self.associative_links[target_id] = target_links[: self.max_links_per_memory]


class AssociativeNetworkVisualizer(Component):
    """
    Component for visualizing the associative memory network.

    This component provides methods to visualize the fabric structure,
    showing connections between memories and their strengths.
    """

    def __init__(self, linker: Optional[AssociativeMemoryLinker] = None):
        """
        Initialize the visualizer.

        Args:
            linker: Optional associative memory linker to visualize
        """
        self.linker = linker
        self.component_id = "associative_network_visualizer"

    def initialize(self, config: dict[str, Any]) -> None:
        """
        Initialize the component with configuration.

        Args:
            config: Configuration dictionary
        """
        if "linker" in config:
            self.linker = config["linker"]

    def generate_network_data(self, max_nodes: int = 100) -> dict[str, Any]:
        """
        Generate data for network visualization.

        Args:
            max_nodes: Maximum number of nodes to include

        Returns:
            dictionary with nodes and links data
        """
        if self.linker is None:
            return {"nodes": [], "links": []}

        # Get all links
        all_links = self.linker.associative_links

        # Get all unique memory IDs
        memory_ids = set()
        for source_id, links in all_links.items():
            memory_ids.add(source_id)
            for target_id, _ in links:
                memory_ids.add(target_id)

        # Limit to max_nodes
        if len(memory_ids) > max_nodes:
            # Take first max_nodes IDs (could be more sophisticated)
            memory_ids = list(memory_ids)[:max_nodes]

        # Create nodes data
        nodes = [{"id": memory_id} for memory_id in memory_ids]

        # Create links data
        links = []

        for source_id, memory_links in all_links.items():
            if source_id not in memory_ids:
                continue

            for target_id, strength in memory_links:
                if target_id not in memory_ids:
                    continue

                links.append({"source": source_id, "target": target_id, "strength": strength})

        return {"nodes": nodes, "links": links}

    def visualize_activation_spread(
        self, start_id: MemoryID, max_hops: int = 2, min_strength: float = 0.3
    ) -> dict[str, Any]:
        """
        Visualize activation spreading from a starting memory.

        Args:
            start_id: Starting memory ID
            max_hops: Maximum number of hops
            min_strength: Minimum link strength

        Returns:
            dictionary with visualization data
        """
        if self.linker is None:
            return {"nodes": [], "links": [], "activations": {}}

        # Get activations by traversing the network
        activations = self.linker.traverse_associative_network(
            start_id=start_id, max_hops=max_hops, min_strength=min_strength
        )

        # Get memory IDs in the activated subgraph
        memory_ids = set(activations.keys())

        # Create nodes data
        nodes = [
            {"id": memory_id, "activation": activations[memory_id]} for memory_id in memory_ids
        ]

        # Create links data
        links = []

        for source_id in memory_ids:
            memory_links = self.linker.associative_links.get(source_id, [])

            for target_id, strength in memory_links:
                if target_id not in memory_ids:
                    continue

                links.append({"source": source_id, "target": target_id, "strength": strength})

        return {"nodes": nodes, "links": links, "activations": activations}
