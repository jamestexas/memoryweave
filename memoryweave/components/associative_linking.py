"""
Associative memory linking component for MemoryWeave.

This module implements components for creating and traversing "fabric" connections
between semantically related memories. These connections enable associative memory
retrieval that goes beyond pure similarity matching.
"""

import heapq
import time
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field

from memoryweave.components.base import Component, MemoryComponent
from memoryweave.components.component_names import ComponentName
from memoryweave.interfaces.memory import MemoryID
from memoryweave.storage import StandardMemoryStore


class AssociativeLink(BaseModel):
    """Model for an associative link between memories."""

    target_id: str
    strength: float = Field(ge=0.0, le=1.0)


class AssociativeLinks(BaseModel):
    """Model for storing all associative links for a memory."""

    links: dict[str, list[AssociativeLink]] = Field(default_factory=dict)


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

    memory_store: Optional[StandardMemoryStore] = Field(default=None)
    similarity_threshold: float = 0.5
    temporal_weight: float = 0.3
    semantic_weight: float = 0.7
    max_links_per_memory: int = 10
    rebuild_frequency: int = 100
    memory_count: int = 0
    component_id: str = Field(default=ComponentName.ASSOCIATIVE_MEMORY_LINKER)
    links_store: AssociativeLinks = Field(default_factory=AssociativeLinks)
    last_rebuild_time: float = Field(default=0)
    memories_since_rebuild: int = 0

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
        self.similarity_threshold = config.get("similarity_threshold", self.similarity_threshold)
        self.temporal_weight = config.get("temporal_weight", self.temporal_weight)
        self.semantic_weight = config.get("semantic_weight", self.semantic_weight)
        self.max_links_per_memory = config.get("max_links_per_memory", self.max_links_per_memory)
        self.rebuild_frequency = config.get("rebuild_frequency", self.rebuild_frequency)

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

        # Convert memory_id to string for consistent key type
        memory_id = str(memory_id)

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
        result["associative_links"] = self.get_associative_links(memory_id)

        return result

    def _establish_links_for_memory(
        self,
        memory_id: str,
        memory_data: dict[str, Any],
        context: dict[str, Any],
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
            if str(other_memory.id) == memory_id:
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
            candidate_links.append((str(other_memory.id), link_strength))

        # Sort candidate links by strength (descending)
        candidate_links.sort(key=lambda x: x[1], reverse=True)

        # Take the top N links
        top_links = candidate_links[: self.max_links_per_memory]

        # Store links in both directions (bidirectional)
        for other_id, strength in top_links:
            # Add forward link
            self.create_associative_link(memory_id, other_id, strength)

    def _rebuild_all_links(self) -> None:
        """Rebuild all associative links from scratch."""
        if self.memory_store is None:
            return

        # Clear existing links
        self.links_store.links.clear()

        # Get all memories
        all_memories = self.memory_store.get_all()
        self.memory_count = len(all_memories)

        # Build embeddings matrix for fast computation
        embeddings = []
        ids = []
        creation_times = []

        for memory in all_memories:
            embeddings.append(memory.embedding)
            ids.append(str(memory.id))  # Convert to string for consistent keys
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
                    # Normalize with a 7-day scale
                    normalized_time_diff = min(1.0, time_diff / (86400 * 7))
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

            # Add top links to memory store
            for other_id, strength in links[: self.max_links_per_memory]:
                self.create_associative_link(memory_id, other_id, strength)

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

        similarity = float(dot_product / (norm1 * norm2))
        return similarity

    def get_associative_links(self, memory_id: MemoryID) -> list[tuple[str, float]]:
        """
        Get associative links for a memory.

        Args:
            memory_id: ID of the memory to get links for

        Returns:
            list of (memory_id, strength) tuples for linked memories
        """
        # Convert memory_id to string for consistent key type
        memory_id_str = str(memory_id)

        # Get the links from the store
        if memory_id_str in self.links_store.links:
            # Convert from AssociativeLink model to tuple format
            return [
                (link.target_id, link.strength) for link in self.links_store.links[memory_id_str]
            ]
        return []

    def traverse_associative_network(
        self, start_id: MemoryID, max_hops: int = 2, min_strength: float = 0.3
    ) -> dict[str, float]:
        """
        Traverse the associative network from a starting memory.

        Args:
            start_id: Starting memory ID
            max_hops: Maximum number of hops
            min_strength: Minimum link strength

        Returns:
            dictionary mapping memory IDs (as strings) to activation strength
        """
        # Convert start_id to string
        start_id_str = str(start_id)

        # Initialize results with starting node
        results = {start_id_str: 1.0}

        # Use a queue to track nodes to visit
        queue = [(start_id_str, 0, 1.0)]
        visited = {start_id_str}

        while queue:
            current_id, hop, strength = queue.pop(0)

            # If we've reached the maximum hop count, stop traversing
            if hop >= max_hops:
                continue

            # Get links for current memory
            links = self.get_associative_links(current_id)

            # Process each link
            for target_id, link_strength in links:
                # Calculate cumulative strength (decays with distance)
                cumulative_strength = strength * link_strength

                # Skip if below minimum strength
                if cumulative_strength < min_strength:
                    continue

                # Add or update result
                if target_id in results:
                    # Take the maximum strength path
                    results[target_id] = max(results[target_id], cumulative_strength)
                else:
                    results[target_id] = cumulative_strength

                # Add to queue if not already visited
                if target_id not in visited:
                    visited.add(target_id)
                    queue.append((target_id, hop + 1, cumulative_strength))

        return results

    def find_path_between_memories(
        self,
        source_id: MemoryID,
        target_id: MemoryID,
        max_hops: int = 3,
    ) -> list[tuple[str, float]]:
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
        # Convert IDs to strings
        source_id_str = str(source_id)
        target_id_str = str(target_id)

        # Use A* search to find the path
        # Priority queue: (cumulative cost, memory_id, path)
        frontier = [(0, source_id_str, [(source_id_str, 1.0)])]
        visited = {source_id_str}

        while frontier:
            _, current_id, path = heapq.heappop(frontier)

            # If we found the target, return the path
            if current_id == target_id_str:
                return path

            # If we've reached the maximum hop count, skip
            if len(path) > max_hops:
                continue

            # Get links for current memory
            links = self.get_associative_links(current_id)

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
        source_id: MemoryID,
        target_id: MemoryID,
        strength: float = 0.5,
    ) -> None:
        """
        Create an associative link between two memories.

        Args:
            source_id: Source memory ID
            target_id: Target memory ID
            strength: Link strength (0-1)
        """
        # Convert IDs to strings for consistent keys
        source_id_str = str(source_id)
        target_id_str = str(target_id)

        # Add forward link
        self._add_or_update_link(source_id_str, target_id_str, strength)

        # Add reverse link with slightly reduced strength
        reduced_strength = max(0.1, strength * 0.8)  # Slightly weaker in reverse
        self._add_or_update_link(target_id_str, source_id_str, reduced_strength)

    def _add_or_update_link(self, source_id: str, target_id: str, strength: float) -> None:
        """Helper to add or update a link with pruning."""
        # Initialize links list for source if it doesn't exist
        if source_id not in self.links_store.links:
            self.links_store.links[source_id] = []

        # Check if link already exists
        link_updated = False
        for i, link in enumerate(self.links_store.links[source_id]):
            if link.target_id == target_id:
                # Update strength if new one is higher
                if strength > link.strength:
                    self.links_store.links[source_id][i].strength = strength
                link_updated = True
                break

        # Add new link if not updated
        if not link_updated:
            self.links_store.links[source_id].append(
                AssociativeLink(target_id=target_id, strength=strength)
            )

        # Prune if exceeded max links
        if len(self.links_store.links[source_id]) > self.max_links_per_memory:
            # Sort by strength and keep top N
            self.links_store.links[source_id].sort(key=lambda x: x.strength, reverse=True)
            self.links_store.links[source_id] = self.links_store.links[source_id][
                : self.max_links_per_memory
            ]

    def process_query(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Process a query using associative links.

        This method enhances retrieval by activating memories that are
        associatively linked to the query's most relevant results.

        Args:
            query: Query string
            context: Context information

        Returns:
            Updated context with associative information
        """
        # Check if we already have some results to use for associative expansion
        initial_results = context.get("results", [])
        updated_context = context.copy()

        if not initial_results or self.memory_store is None:
            return updated_context

        # Get seed memory IDs from top results
        seed_memories = []
        for result in initial_results[:3]:  # Use top 3 results as seeds
            memory_id = result.get("memory_id")
            if memory_id is not None:
                seed_memories.append(memory_id)

        if not seed_memories:
            return updated_context

        # Find associatively linked memories
        associated_memories = {}
        for seed_id in seed_memories:
            # Traverse network from this seed
            activations = self.traverse_associative_network(seed_id, max_hops=2, min_strength=0.3)

            # Update with strongest activations
            for memory_id, activation in activations.items():
                # Try to convert back to original ID type for API consistency
                try:
                    # If the original seed was an int, try to convert back to int
                    if isinstance(seed_id, int):
                        memory_id_converted = int(memory_id)
                    else:
                        memory_id_converted = memory_id
                except (ValueError, TypeError):
                    memory_id_converted = memory_id

                if memory_id_converted in associated_memories:
                    associated_memories[memory_id_converted] = max(
                        associated_memories[memory_id_converted], activation
                    )
                else:
                    associated_memories[memory_id_converted] = activation

        # Add to context
        updated_context["associative_memories"] = associated_memories

        return updated_context

    @property
    def associative_links(self):
        """Legacy property for backward compatibility."""
        # Convert from Pydantic model to dictionary of tuples
        links_dict = {}
        for source_id, links in self.links_store.links.items():
            # Convert links to tuple format
            tuple_links = [(link.target_id, link.strength) for link in links]

            # Add with both string and int keys for backward compatibility
            links_dict[source_id] = tuple_links

            # Also add with int key if possible
            try:
                int_id = int(source_id)
                links_dict[int_id] = tuple_links
            except (ValueError, TypeError):
                pass

        return links_dict


class AssociativeNetworkVisualizer(Component):
    """
    Component for visualizing the associative memory network.

    This component provides methods to visualize the fabric structure,
    showing connections between memories and their strengths.
    """

    linker: Optional[AssociativeMemoryLinker] = None
    component_id: str = "associative_network_visualizer"

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
            memory_ids.add(str(source_id))
            for target_id, _ in links:
                memory_ids.add(str(target_id))

        # Limit to max_nodes
        if len(memory_ids) > max_nodes:
            # Take first max_nodes IDs (could be more sophisticated)
            memory_ids = list(memory_ids)[:max_nodes]

        # Create nodes data
        nodes = [{"id": memory_id} for memory_id in memory_ids]

        # Create links data
        links = []

        for source_id, memory_links in all_links.items():
            if str(source_id) not in memory_ids:
                continue

            for target_id, strength in memory_links:
                if str(target_id) not in memory_ids:
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
            memory_links = self.linker.get_associative_links(source_id)

            for target_id, strength in memory_links:
                if target_id not in memory_ids:
                    continue

                links.append({"source": source_id, "target": target_id, "strength": strength})

        return {"nodes": nodes, "links": links, "activations": activations}
