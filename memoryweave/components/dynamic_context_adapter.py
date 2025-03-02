# memoryweave/components/dynamic_context_adapter.py
"""
DynamicContextAdapter

A component that dynamically adjusts retrieval parameters based on contextual
signals without requiring explicit human feedback.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict

from memoryweave.components.base import Component
from memoryweave.components.component_names import ComponentName
from memoryweave.interfaces.memory import IMemoryStore

class DynamicContextAdapter(Component):
    """
    Dynamically adapts retrieval parameters based on multiple contextual signals.
    
    This adapter:
    - Processes query characteristics, memory metrics, and context signals
    - Dynamically adjusts parameters for each query
    - Uses internal metrics to evaluate retrieval quality
    - Adapts differently for various memory sizes and query types
    """
    
    def __init__(self):
        """Initialize the dynamic context adapter."""
        self.logger = logging.getLogger(__name__)
        self.component_id = ComponentName.DYNAMIC_CONTEXT_ADAPTER
        
        # Default parameters (will be initialized with config)
        self.default_parameters = {}
        
        # Configuration variables
        self.adaptation_strength = 1.0
        self.enable_logging = False
        self.enable_memory_size_adaptation = True
        self.adaptation_history = []
        self.max_history_size = 20
        
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the component with configuration.
        
        Args:
            config: Configuration dictionary
        """
        # Set adaptation strength (0.0-1.0)
        self.adaptation_strength = config.get("adaptation_strength", 1.0)
        
        # Enable/disable features
        self.enable_logging = config.get("enable_logging", False)
        self.enable_memory_size_adaptation = config.get("enable_memory_size_adaptation", True)
        
        # History tracking
        self.max_history_size = config.get("max_history_size", 20)
        
        # Initialize default parameters from config
        self.default_parameters = {
            # Core retrieval parameters
            "confidence_threshold": config.get("confidence_threshold", 0.1),
            "similarity_weight": config.get("similarity_weight", 0.5),
            "associative_weight": config.get("associative_weight", 0.3),
            "temporal_weight": config.get("temporal_weight", 0.1),
            "activation_weight": config.get("activation_weight", 0.1),
            
            # Advanced parameters
            "max_associative_hops": config.get("max_associative_hops", 2),
            "first_stage_k": config.get("first_stage_k", 20),
            "first_stage_threshold_factor": config.get("first_stage_threshold_factor", 0.7),
            "min_results": config.get("min_results", 5),
            "max_candidates": config.get("max_candidates", 50),
            
            # Feature flags
            "use_progressive_filtering": config.get("use_progressive_filtering", False),
            "use_batched_computation": config.get("use_batched_computation", False),
        }
        
        if self.enable_logging:
            self.logger.info(f"DynamicContextAdapter initialized with adaptation_strength={self.adaptation_strength}")
            
    def process_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a query to dynamically adapt retrieval parameters.
        
        Args:
            query: The query string
            context: Context dictionary with query analysis, memory store, etc.
            
        Returns:
            Dictionary with adapted retrieval parameters
        """
        # Skip processing if adaptation is disabled
        if context.get("enable_dynamic_adaptation", True) is False:
            return {"adapted_retrieval_params": self.default_parameters.copy()}
        
        # Start with default parameters
        adapted_params = self.default_parameters.copy()
        
        # Extract signals from context
        signals = self._extract_signals(query, context)
        
        # Adapt parameters based on signals
        if self.enable_memory_size_adaptation:
            adapted_params = self._adapt_for_memory_size(adapted_params, signals)
            
        # Adapt based on query type
        adapted_params = self._adapt_for_query_type(adapted_params, signals)
        
        # Adapt based on query specificity and complexity
        adapted_params = self._adapt_for_query_characteristics(adapted_params, signals)
        
        # Apply adaptation strength to control how much parameters are changed
        final_params = self._apply_adaptation_strength(self.default_parameters, adapted_params)
        
        # Set a flag to indicate that parameters were adapted by this component
        final_params["adapted_by_dynamic_context"] = True
        
        # Log adaptation if enabled
        if self.enable_logging:
            self._log_adaptation(query, signals, final_params)
            
        # Store adaptation in history
        self._store_adaptation(query, signals, final_params)
        
        return {"adapted_retrieval_params": final_params}
    
    def _extract_signals(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract contextual signals from query and context.
        
        Args:
            query: Query string
            context: Context dictionary
            
        Returns:
            Dictionary of extracted signals
        """
        signals = {}
        
        # Query characteristics
        signals["query_length"] = len(query.split())
        signals["query_type"] = context.get("primary_query_type", "default")
        signals["query_confidence"] = context.get("query_type_confidence", 1.0)
        signals["has_entity"] = len(context.get("entities", [])) > 0
        signals["has_temporal_reference"] = context.get("has_temporal_reference", False)
        signals["query_complexity"] = min(1.0, signals["query_length"] / 20)
        
        # Calculate query specificity based on keyword ratio and entity presence
        keywords = context.get("keywords", [])
        important_keywords = context.get("important_keywords", [])
        keyword_ratio = len(important_keywords) / max(1, len(keywords)) if keywords else 0
        signals["query_specificity"] = min(1.0, (keyword_ratio * 0.7) + (0.3 if signals["has_entity"] else 0))
        
        # Memory store characteristics
        memory_store = context.get("memory_store")
        if memory_store and hasattr(memory_store, "memory_embeddings"):
            # Get memory size
            memory_size = len(memory_store.memory_embeddings)
            signals["memory_size"] = memory_size
            
            # Calculate embedding space density (approximation)
            # This is computationally expensive for large stores, so we'll skip it
            # if memory_size > 100 and use a default value instead
            if memory_size <= 100 and hasattr(memory_store, "memory_embeddings"):
                # Calculate average pairwise similarity to estimate density
                embedding_sample = memory_store.memory_embeddings
                if memory_size > 20:
                    # Take a sample of embeddings for large stores
                    indices = np.random.choice(memory_size, 20, replace=False)
                    embedding_sample = embedding_store.memory_embeddings[indices]
                
                # Calculate pairwise similarities
                similarity_matrix = np.dot(embedding_sample, embedding_sample.T)
                # Mask diagonal (self-similarity)
                np.fill_diagonal(similarity_matrix, 0)
                # Calculate average similarity
                avg_similarity = np.mean(similarity_matrix)
                signals["embedding_density"] = avg_similarity
            else:
                # Default density for large stores
                signals["embedding_density"] = 0.5
        else:
            signals["memory_size"] = 0
            signals["embedding_density"] = 0.5
            
        # Activation metrics if available
        activation_manager = context.get("activation_manager")
        if activation_manager and hasattr(activation_manager, "get_activated_memories"):
            activations = activation_manager.get_activated_memories(threshold=0.0)
            signals["activation_count"] = len(activations)
            signals["activation_density"] = len(activations) / max(1, signals["memory_size"]) if signals["memory_size"] > 0 else 0
            signals["max_activation"] = max([v for k, v in activations.items()]) if activations else 0
        else:
            signals["activation_count"] = 0
            signals["activation_density"] = 0
            signals["max_activation"] = 0
            
        return signals
    
    def _adapt_for_memory_size(self, params: Dict[str, Any], signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt parameters based on memory store characteristics.
        
        Args:
            params: Current parameters
            signals: Extracted signals
            
        Returns:
            Adapted parameters
        """
        memory_size = signals.get("memory_size", 0)
        embedding_density = signals.get("embedding_density", 0.5)
        
        # If memory store is empty, return unchanged parameters
        if memory_size == 0:
            return params
            
        # Create a copy of parameters
        adapted = params.copy()
        
        # Apply logarithmic scaling for smooth degradation with size
        size_scaling_factor = max(0.3, 1.0 / (1 + np.log(max(1, memory_size)/100)))
        
        # Apply memory size-specific adaptations
        if memory_size > 500:  # Large memory stores
            # Enable optimization for large stores
            adapted["use_progressive_filtering"] = True
            adapted["use_batched_computation"] = True
            adapted["first_stage_filter_size"] = min(200, int(memory_size / 5))
            adapted["batch_size"] = 200
            
            # Scale weights for large stores
            adapted["activation_weight"] = params["activation_weight"] * size_scaling_factor
            adapted["associative_weight"] = params["associative_weight"] * size_scaling_factor
            
            # Compensate by increasing similarity weight
            weight_reduction = (params["activation_weight"] - adapted["activation_weight"]) + \
                              (params["associative_weight"] - adapted["associative_weight"])
            adapted["similarity_weight"] = min(0.8, params["similarity_weight"] + weight_reduction * 0.7)
            
            # Set temporal weight so weights sum to 1.0
            adapted["temporal_weight"] = 1.0 - (adapted["similarity_weight"] + 
                                              adapted["associative_weight"] + 
                                              adapted["activation_weight"])
                                              
            # Increase min_results to ensure diversity in large stores
            adapted["min_results"] = max(5, min(10, int(params["min_results"] * 1.5)))
            
        elif memory_size < 50:  # Small memory stores
            # Optimize for small stores
            adapted["use_progressive_filtering"] = False
            adapted["use_batched_computation"] = False
            
            # Adjust weights for small stores (more balanced)
            adapted["similarity_weight"] = 0.4  # Reduce from default 0.5
            adapted["associative_weight"] = 0.2  # Reduce from default 0.3
            adapted["temporal_weight"] = 0.2     # Increase from default 0.1
            adapted["activation_weight"] = 0.2   # Increase from default 0.1
            
            # Use lower threshold for small stores
            adapted["confidence_threshold"] = max(0.05, params["confidence_threshold"] * 0.8)
            
            # Decrease min_results for small stores
            adapted["min_results"] = max(3, int(params["min_results"] * 0.6))
            
        # Adjust for embedding space density
        if embedding_density > 0.7:  # Dense embedding space
            # Higher threshold needed to discriminate in dense spaces
            adapted["confidence_threshold"] = min(0.9, params["confidence_threshold"] * 1.2)
        elif embedding_density < 0.3:  # Sparse embedding space
            # Lower threshold for sparse spaces
            adapted["confidence_threshold"] = max(0.05, params["confidence_threshold"] * 0.8)
            
        return adapted
    
    def _adapt_for_query_type(self, params: Dict[str, Any], signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt parameters based on query type.
        
        Args:
            params: Current parameters
            signals: Extracted signals
            
        Returns:
            Adapted parameters
        """
        query_type = signals.get("query_type", "default")
        query_confidence = signals.get("query_confidence", 1.0)
        
        # Create a copy of parameters
        adapted = params.copy()
        
        # Apply type-specific adaptations with confidence weighting
        confidence_factor = max(0.1, min(1.0, query_confidence))
        adaptation_amount = 0.7 * confidence_factor  # Scale adaptation by confidence
        
        if query_type == "personal":
            # Personal queries need higher precision
            adapted["confidence_threshold"] = min(0.9, params["confidence_threshold"] + 0.15 * adaptation_amount)
            adapted["similarity_weight"] = min(0.8, params["similarity_weight"] + 0.1 * adaptation_amount)
            adapted["activation_weight"] = min(0.3, params["activation_weight"] + 0.1 * adaptation_amount)
            adapted["associative_weight"] = max(0.1, params["associative_weight"] - 0.1 * adaptation_amount)
            adapted["temporal_weight"] = max(0.05, params["temporal_weight"] - 0.1 * adaptation_amount)
            
            # Normalize weights to ensure they sum to 1.0
            total = (adapted["similarity_weight"] + adapted["associative_weight"] + 
                     adapted["temporal_weight"] + adapted["activation_weight"])
            if total > 1.0:
                factor = 1.0 / total
                adapted["similarity_weight"] *= factor
                adapted["associative_weight"] *= factor
                adapted["temporal_weight"] *= factor
                adapted["activation_weight"] *= factor
                
        elif query_type == "factual":
            # Factual queries need better recall
            adapted["confidence_threshold"] = max(0.05, params["confidence_threshold"] - 0.05 * adaptation_amount)
            adapted["similarity_weight"] = min(0.9, params["similarity_weight"] + 0.2 * adaptation_amount)
            adapted["associative_weight"] = max(0.05, params["associative_weight"] - 0.1 * adaptation_amount)
            adapted["max_candidates"] = int(params["max_candidates"] * (1 + 0.5 * adaptation_amount))
            adapted["first_stage_k"] = int(params["first_stage_k"] * (1 + 0.5 * adaptation_amount))
            
        elif query_type == "temporal":
            # Temporal queries need temporal context emphasis
            adapted["temporal_weight"] = min(0.4, params["temporal_weight"] + 0.3 * adaptation_amount)
            # Reduce other weights to compensate
            reduction = (adapted["temporal_weight"] - params["temporal_weight"]) / 3
            adapted["similarity_weight"] = max(0.3, params["similarity_weight"] - reduction)
            adapted["associative_weight"] = max(0.1, params["associative_weight"] - reduction)
            adapted["activation_weight"] = max(0.05, params["activation_weight"] - reduction)
            
        # Ensure required parameters are within valid ranges
        adapted["confidence_threshold"] = max(0.01, min(0.9, adapted["confidence_threshold"]))
        adapted["min_results"] = max(1, int(adapted["min_results"]))
        adapted["max_candidates"] = max(10, int(adapted["max_candidates"]))
            
        return adapted
    
    def _adapt_for_query_characteristics(self, params: Dict[str, Any], signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt parameters based on query specificity and complexity.
        
        Args:
            params: Current parameters
            signals: Extracted signals
            
        Returns:
            Adapted parameters
        """
        query_specificity = signals.get("query_specificity", 0.5)
        query_complexity = signals.get("query_complexity", 0.5)
        has_temporal_reference = signals.get("has_temporal_reference", False)
        has_entity = signals.get("has_entity", False)
        
        # Create a copy of parameters
        adapted = params.copy()
        
        # Adapt for query specificity
        if query_specificity > 0.7:  # Highly specific query
            # Increase similarity weight for specific queries
            adapted["similarity_weight"] = min(0.8, params["similarity_weight"] * 1.2)
            # Allow lower confidence threshold for better recall
            adapted["confidence_threshold"] = max(0.05, params["confidence_threshold"] * 0.9)
            # Reduce associative influence for specific queries
            adapted["associative_weight"] = max(0.1, params["associative_weight"] * 0.8)
            
        elif query_specificity < 0.3:  # Vague/general query
            # For vague queries, associative and temporal context matter more
            adapted["associative_weight"] = min(0.5, params["associative_weight"] * 1.3)
            # Increase confidence threshold to filter noise
            adapted["confidence_threshold"] = min(0.9, params["confidence_threshold"] * 1.2)
            # For vague queries with entities, allow more results
            if has_entity:
                adapted["max_candidates"] = int(params["max_candidates"] * 1.5)
                
        # Adapt for query complexity
        if query_complexity > 0.7:  # Complex query
            # Complex queries may need more results to capture all aspects
            adapted["min_results"] = min(15, int(params["min_results"] * 1.5))
            adapted["max_candidates"] = int(params["max_candidates"] * 1.3)
            
        elif query_complexity < 0.3:  # Simple query
            # Simple queries should have more focused results
            adapted["min_results"] = max(3, int(params["min_results"] * 0.8))
            
        # Adapt for temporal references
        if has_temporal_reference:
            # Emphasize temporal weight for queries with time references
            adapted["temporal_weight"] = min(0.4, params["temporal_weight"] * 2.0)
            
            # Reduce other weights to compensate for increased temporal weight
            weight_increase = adapted["temporal_weight"] - params["temporal_weight"]
            reduction_per_weight = weight_increase / 3
            
            adapted["similarity_weight"] = max(0.3, params["similarity_weight"] - reduction_per_weight)
            adapted["associative_weight"] = max(0.1, params["associative_weight"] - reduction_per_weight)
            adapted["activation_weight"] = max(0.05, params["activation_weight"] - reduction_per_weight)
            
        # Ensure weights sum to 1.0
        total_weight = (adapted["similarity_weight"] + adapted["associative_weight"] + 
                         adapted["temporal_weight"] + adapted["activation_weight"])
                         
        if abs(total_weight - 1.0) > 0.01:
            # Normalize weights
            factor = 1.0 / total_weight
            adapted["similarity_weight"] *= factor
            adapted["associative_weight"] *= factor
            adapted["temporal_weight"] *= factor
            adapted["activation_weight"] *= factor
            
        return adapted
    
    def _apply_adaptation_strength(
        self, 
        original_params: Dict[str, Any], 
        adapted_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply adaptation strength to control parameter changes.
        
        Args:
            original_params: Original parameters
            adapted_params: Fully adapted parameters
            
        Returns:
            Parameters with adaptation strength applied
        """
        # If adaptation strength is 1.0, return fully adapted parameters
        if self.adaptation_strength >= 1.0:
            return adapted_params
            
        # If adaptation strength is 0.0, return original parameters
        if self.adaptation_strength <= 0.0:
            return original_params.copy()
            
        # Otherwise, interpolate between original and adapted
        result = {}
        for key, original_value in original_params.items():
            if key in adapted_params:
                adapted_value = adapted_params[key]
                
                # For numeric values, interpolate
                if isinstance(original_value, (int, float)) and isinstance(adapted_value, (int, float)):
                    # For integers, round after interpolation
                    if isinstance(original_value, int):
                        result[key] = int(round(original_value + self.adaptation_strength * (adapted_value - original_value)))
                    else:
                        result[key] = original_value + self.adaptation_strength * (adapted_value - original_value)
                # For booleans and other values, use original unless adaptation_strength > 0.5
                else:
                    result[key] = adapted_value if self.adaptation_strength > 0.5 else original_value
            else:
                # Keep original value if not in adapted_params
                result[key] = original_value
                
        # Copy any additional keys from adapted_params
        for key, value in adapted_params.items():
            if key not in result:
                result[key] = value
                
        return result
    
    def _log_adaptation(self, query: str, signals: Dict[str, Any], params: Dict[str, Any]) -> None:
        """
        Log adaptation decisions.
        
        Args:
            query: Query string
            signals: Extracted signals
            params: Adapted parameters
        """
        # Truncate query if too long
        query_str = query if len(query) < 50 else query[:47] + "..."
        
        # Log basic info
        self.logger.info(f"Dynamic adaptation for query: '{query_str}'")
        self.logger.info(f"Memory size: {signals.get('memory_size', 'unknown')}, "
                       f"Query type: {signals.get('query_type', 'unknown')}")
                       
        # Log key parameters
        self.logger.info(f"Confidence threshold: {params['confidence_threshold']:.3f}")
        self.logger.info(f"Weights: sim={params['similarity_weight']:.2f}, "
                       f"assoc={params['associative_weight']:.2f}, "
                       f"temp={params['temporal_weight']:.2f}, "
                       f"act={params['activation_weight']:.2f}")
                       
        # Log optimization flags
        if params.get("use_progressive_filtering", False):
            self.logger.info("Optimization: Using progressive filtering")
        if params.get("use_batched_computation", False):
            self.logger.info("Optimization: Using batched computation")
    
    def _store_adaptation(self, query: str, signals: Dict[str, Any], params: Dict[str, Any]) -> None:
        """
        Store adaptation in history.
        
        Args:
            query: Query string
            signals: Extracted signals
            params: Adapted parameters
        """
        # Create adaptation record
        adaptation = {
            "timestamp": time.time(),
            "query": query,
            "signals": {k: signals[k] for k in signals if k in [
                "query_type", "memory_size", "embedding_density", 
                "query_specificity", "activation_density", "query_complexity",
                "has_temporal_reference"
            ]},
            "params": params.copy()
        }
        
        # Store in history
        self.adaptation_history.append(adaptation)
        if len(self.adaptation_history) > self.max_history_size:
            self.adaptation_history.pop(0)