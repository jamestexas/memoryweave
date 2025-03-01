# memoryweave/components/post_processors.py
from typing import Any, Dict, List

from memoryweave.components.base import PostProcessor


class KeywordBoostProcessor(PostProcessor):
    """
    Boosts relevance scores of results containing important keywords.
    """

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.keyword_boost_weight = config.get("keyword_boost_weight", 0.5)

    def process_results(
        self, results: list[dict[str, Any]], query: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Process retrieved results by boosting for keyword matches."""
        # Get important keywords from query analysis
        keywords = context.get("important_keywords", set())
        if not keywords:
            return results

        # Apply keyword boost
        for result in results:
            content = str(result.get("content", "")).lower()

            # Count keyword matches
            keyword_matches = sum(1 for kw in keywords if kw.lower() in content)

            # Apply boost proportional to matches and weight
            if keyword_matches > 0:
                boost = min(self.keyword_boost_weight * keyword_matches / len(keywords), 0.5)

                # Apply boost to relevance score
                current_score = result.get("relevance_score", 0)
                new_score = min(current_score + boost * (1.0 - current_score), 1.0)
                result["relevance_score"] = new_score
                result["keyword_boost_applied"] = True

        return results


class SemanticCoherenceProcessor(PostProcessor):
    """
    Adjusts relevance scores based on semantic coherence with query.
    """

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.coherence_threshold = config.get("coherence_threshold", 0.2)

    def process_results(
        self, results: list[dict[str, Any]], query: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Process retrieved results checking semantic coherence."""
        # Get query type from context
        query_type = context.get("primary_query_type", "factual")

        # Penalize incoherent results based on query type
        for result in results:
            # Check for type mismatch
            result_type = result.get("type", "unknown")

            if query_type == "personal" and result_type == "factual":
                # Penalize factual results for personal queries
                result["relevance_score"] = max(0, result.get("relevance_score", 0) - 0.2)

            elif query_type == "factual" and result_type == "personal":
                # Slightly penalize personal results for factual queries
                result["relevance_score"] = max(0, result.get("relevance_score", 0) - 0.1)

        return results


class AdaptiveKProcessor(PostProcessor):
    """
    Adjusts the number of results based on query characteristics.
    """

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.adaptive_k_factor = config.get("adaptive_k_factor", 0.3)

    def process_results(
        self, results: list[dict[str, Any]], query: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Process retrieved results by adjusting the number based on scores."""
        if not results:
            return results

        # Get original top_k
        original_k = context.get("top_k", 5)

        # Check result quality
        avg_score = sum(r.get("relevance_score", 0) for r in results) / len(results)

        # Adjust number of results based on quality
        if avg_score > 0.7:
            # High quality results - keep fewer
            adaptive_k = max(1, int(original_k * (1.0 - self.adaptive_k_factor)))
            return sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)[
                :adaptive_k
            ]
        elif avg_score < 0.3:
            # Low quality results - keep more for diversity
            return results
        else:
            # Medium quality - sort and return original amount
            return sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)[
                :original_k
            ]


class MinimumResultGuaranteeProcessor(PostProcessor):
    """
    Ensures a minimum number of results are returned even if they don't meet the confidence threshold.
    
    This processor implements a fallback retrieval strategy when the initial retrieval
    doesn't return enough results, ensuring that queries always receive a response.
    """
    
    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.min_results = config.get("min_results", 1)
        self.fallback_threshold_factor = config.get("fallback_threshold_factor", 0.5)
        self.min_fallback_threshold = config.get("min_fallback_threshold", 0.05)
        self.memory = config.get("memory", None)
        
    def process_results(
        self, results: list[dict[str, Any]], query: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Process retrieved results by ensuring a minimum number of results."""
        # If we already have enough results, no action needed
        if len(results) >= self.min_results:
            return results
            
        # Check if we have access to the necessary components to perform fallback retrieval
        if not self.memory or "query_embedding" not in context:
            return results
            
        # Get the original confidence threshold used
        original_threshold = context.get("confidence_threshold", 0.0)
        
        # Calculate fallback threshold - lower than original but with a minimum
        fallback_threshold = max(
            self.min_fallback_threshold, 
            original_threshold * self.fallback_threshold_factor
        )
        
        # Get existing memory IDs to avoid duplicates
        existing_ids = {r.get("id") for r in results if "id" in r}
        
        # Try to get additional results with lower threshold
        try:
            # If memory has a direct similarity search method
            if hasattr(self.memory, "search_by_embedding"):
                # Calculate how many more results we need
                additional_count = self.min_results - len(results)
                
                # Get query embedding
                query_embedding = context["query_embedding"]
                
                # Get additional results with lower threshold
                fallback_results = self.memory.search_by_embedding(
                    query_embedding, 
                    k=additional_count + len(existing_ids),  # Request extra to account for duplicates
                    threshold=fallback_threshold
                )
                
                # Filter out existing IDs
                fallback_results = [
                    r for r in fallback_results if r.get("id") not in existing_ids
                ]
                
                # Add to the original results until min_results is reached
                for result in fallback_results[:additional_count]:
                    result["from_fallback"] = True
                    results.append(result)
                    
        except Exception as e:
            # Log error but don't fail
            print(f"Error in minimum result guarantee fallback: {str(e)}")
            
        return results


class PersonalAttributeProcessor(PostProcessor):
    """
    Enhances retrieval results based on personal attributes from the query.
    
    This processor analyzes the query for personal attribute references and boosts
    results that contain relevant attributes. It can also generate synthetic
    attribute memory entries for direct attribute questions.
    """
    
    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.attribute_boost_factor = config.get("attribute_boost_factor", 0.6)
        self.add_direct_responses = config.get("add_direct_responses", True)
        self.min_relevance_threshold = config.get("min_relevance_threshold", 0.3)
    
    def process_results(
        self, results: list[dict[str, Any]], query: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Process retrieved results by incorporating personal attributes."""
        # Check if personal attributes are available in context
        if "personal_attributes" not in context:
            return results
            
        personal_attributes = context.get("personal_attributes", {})
        relevant_attributes = context.get("relevant_attributes", {})
        
        if not relevant_attributes:
            return results
            
        # Create a copy of results to modify
        enhanced_results = list(results)
        
        # 1. Boost existing results that contain relevant attributes
        for result in enhanced_results:
            content = str(result.get("content", "")).lower()
            
            # Check for attribute matches in content
            attribute_matches = 0
            for attr_type, attr_value in relevant_attributes.items():
                if isinstance(attr_value, str) and attr_value.lower() in content:
                    attribute_matches += 1
                elif isinstance(attr_value, list):
                    for value in attr_value:
                        if value.lower() in content:
                            attribute_matches += 1
            
            # Apply boost based on matches
            if attribute_matches > 0:
                boost = min(self.attribute_boost_factor * attribute_matches / len(relevant_attributes), 0.8)
                current_score = result.get("relevance_score", 0)
                new_score = min(current_score + boost * (1.0 - current_score), 1.0)
                result["relevance_score"] = new_score
                result["attribute_boost_applied"] = True
        
        # 2. For direct attribute questions, create a synthetic result if needed
        if self.add_direct_responses and relevant_attributes:
            # Check if query is likely a direct question about an attribute
            direct_query_types = ["what is my", "where do i", "who is my", "tell me my", "what's my"]
            is_direct_query = any(query.lower().startswith(prefix) for prefix in direct_query_types)
            
            # If direct query and no high relevance results exist, create synthetic response
            has_high_relevance = any(r.get("relevance_score", 0) > self.min_relevance_threshold for r in enhanced_results)
            
            if is_direct_query and (not has_high_relevance or not enhanced_results):
                # Create synthetic attribute response
                attribute_memory = self._create_attribute_memory(query, relevant_attributes)
                if attribute_memory:
                    # Add as highest relevance result
                    enhanced_results.insert(0, attribute_memory)
        
        return enhanced_results
    
    def _create_attribute_memory(self, query: str, relevant_attributes: dict[str, Any]) -> dict[str, Any]:
        """Create a synthetic memory entry from personal attributes relevant to the query."""
        if not relevant_attributes:
            return None
            
        # Determine which attribute is most relevant to query
        attr_key = next(iter(relevant_attributes.keys()))
        attr_value = relevant_attributes[attr_key]
        
        # Format depends on attribute type
        attr_category = attr_key.split("_")[0] if "_" in attr_key else "attribute"
        attr_type = attr_key.split("_")[1] if "_" in attr_key else attr_key
        
        # Generate content based on attribute type
        if attr_category == "preferences":
            content = f"Your favorite {attr_type} is {attr_value}."
        elif attr_category == "demographic":
            if attr_type == "location":
                content = f"You live in {attr_value}."
            elif attr_type == "occupation":
                content = f"You work as a {attr_value}."
            else:
                content = f"Your {attr_type} is {attr_value}."
        elif attr_category == "relationship":
            content = f"Your {attr_type} is {attr_value}."
        elif attr_category == "trait":
            if attr_type == "hobbies" and isinstance(attr_value, list):
                hobbies_str = ", ".join(attr_value)
                content = f"Your hobbies include {hobbies_str}."
            else:
                content = f"Your {attr_type} is {attr_value}."
        else:
            content = f"Your {attr_type} is {attr_value}."
        
        # Create memory entry
        return {
            "content": content,
            "relevance_score": 1.0,  # Highest relevance
            "type": "attribute",
            "source": "personal_attribute",
            "embedding": None,
            "id": f"attribute-{attr_key}",
            "timestamp": None,
            "is_synthetic": True
        }
