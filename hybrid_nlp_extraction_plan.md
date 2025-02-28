# Hybrid Regex-NLP Extraction Implementation Plan

## Overview
This plan outlines how to enhance MemoryWeave's extraction logic by integrating NLP techniques alongside existing regex patterns. The approach will start with regex for efficiency, then fall back to more sophisticated NLP techniques when confidence is low or patterns aren't matched.

## Implementation Details

### 1. Add NLP Dependencies

Add spaCy as a dependency for robust NLP capabilities:

```python
# Add to pyproject.toml dependencies
"spacy": "^3.7.0",
```

Install a language model:
```bash
python -m spacy download en_core_web_md
```

### 2. Create NLP Extraction Utilities

```python
# memoryweave/utils/nlp_extraction.py
import spacy
from typing import Dict, List, Optional, Set, Any, Tuple

class NLPExtractor:
    """NLP-based attribute extractor using spaCy."""
    
    def __init__(self, model_name="en_core_web_md"):
        """Initialize with specified spaCy model."""
        self.nlp = spacy.load(model_name)
        
    def extract_personal_attributes(self, text: str) -> Dict[str, Any]:
        """
        Extract personal attributes using NLP techniques.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of extracted attributes
        """
        attributes = {
            "preferences": {},
            "demographics": {},
            "traits": {},
            "relationships": {}
        }
        
        if not text:
            return attributes
            
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract preferences
        self._extract_preferences(doc, attributes)
        
        # Extract demographic information
        self._extract_demographics(doc, attributes)
        
        # Extract traits and characteristics
        self._extract_traits(doc, attributes)
        
        # Extract relationships
        self._extract_relationships(doc, attributes)
        
        return attributes
    
    def _extract_preferences(self, doc, attributes: Dict) -> None:
        """Extract preferences using NLP techniques."""
        # Look for preference patterns in dependency tree
        for sent in doc.sents:
            for token in sent:
                # Check for "favorite X is Y" pattern
                if token.lemma_ in ["favorite", "prefer", "like", "love", "enjoy"]:
                    # Find the object of preference
                    preference_type = None
                    preference_value = None
                    
                    # Look for the type (what they prefer)
                    for child in token.children:
                        if child.dep_ in ["amod", "compound", "dobj"]:
                            preference_type = child.text.lower()
                    
                    # Look for the value (what specifically they prefer)
                    for token2 in sent:
                        if token2.lemma_ == "be" and token2.head == token:
                            for child in token2.children:
                                if child.dep_ in ["attr", "dobj"]:
                                    preference_value = child.text
                                    # Get any additional words that modify this
                                    for grandchild in child.children:
                                        if grandchild.dep_ in ["amod", "compound"]:
                                            preference_value = f"{grandchild.text} {preference_value}"
                    
                    if preference_type and preference_value:
                        attributes["preferences"][preference_type] = preference_value
    
    def _extract_demographics(self, doc, attributes: Dict) -> None:
        """Extract demographic information using NLP."""
        # Extract locations
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                # Check if they live there
                for sent in doc.sents:
                    if ent.text in sent.text:
                        for token in sent:
                            if token.lemma_ in ["live", "reside", "stay"] and token.dep_ == "ROOT":
                                attributes["demographics"]["location"] = ent.text
                                break
        
        # Extract occupation
        for sent in doc.sents:
            occupation_patterns = [
                # "I am a doctor"
                {"ROOT": {"POS": "AUX"}, "SUBJ": {"LEMMA": "I"}, "ATTR": {"POS": "NOUN"}},
                # "I work as a teacher"
                {"ROOT": {"LEMMA": "work"}, "SUBJ": {"LEMMA": "I"}, "PREP": {"LEMMA": "as"}}
            ]
            
            # Simplified pattern matching - in real implementation would be more sophisticated
            for token in sent:
                if token.dep_ == "ROOT" and token.lemma_ == "be":
                    subject = None
                    attr = None
                    
                    for child in token.children:
                        if child.dep_ == "nsubj" and child.lemma_ in ["I", "i"]:
                            subject = child
                        elif child.dep_ == "attr":
                            attr = child
                    
                    if subject and attr and attr.pos_ == "NOUN":
                        attributes["demographics"]["occupation"] = attr.text
    
    def _extract_traits(self, doc, attributes: Dict) -> None:
        """Extract personality traits and hobbies."""
        # Look for hobbies/activities
        hobby_verbs = ["enjoy", "like", "love", "prefer"]
        activity_nouns = set()
        
        for token in doc:
            if token.lemma_ in hobby_verbs:
                for child in token.children:
                    if child.dep_ == "dobj" or (child.dep_ == "xcomp" and child.pos_ == "VERB"):
                        # Found potential hobby
                        activity = child.text
                        # Get any modifiers
                        for grandchild in child.children:
                            if grandchild.dep_ in ["compound", "amod"]:
                                activity = f"{grandchild.text} {activity}"
                        
                        activity_nouns.add(activity)
        
        if activity_nouns:
            if "hobbies" not in attributes["traits"]:
                attributes["traits"]["hobbies"] = []
            attributes["traits"]["hobbies"].extend(list(activity_nouns))
    
    def _extract_relationships(self, doc, attributes: Dict) -> None:
        """Extract relationship information."""
        # Extract family relationships
        relationship_terms = ["wife", "husband", "spouse", "partner", "girlfriend", 
                             "boyfriend", "child", "son", "daughter", "mother", 
                             "father", "brother", "sister"]
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Check if this person is mentioned with a relationship term
                for sent in doc.sents:
                    if ent.text in sent.text:
                        for rel_term in relationship_terms:
                            if rel_term in sent.text.lower():
                                # Found potential relationship
                                if "family" not in attributes["relationships"]:
                                    attributes["relationships"]["family"] = {}
                                attributes["relationships"]["family"][rel_term] = ent.text
                                break
                                
    def identify_query_type(self, query: str) -> Dict[str, float]:
        """
        Identify the type of query using NLP techniques.
        
        Args:
            query: The query text
            
        Returns:
            Dictionary with query type probabilities
        """
        doc = self.nlp(query)
        
        # Initialize scores
        scores = {
            "factual": 0.0,
            "personal": 0.0,
            "opinion": 0.0,
            "instruction": 0.0
        }
        
        # Check for personal pronouns (indicates personal query)
        personal_pronouns = ["i", "me", "my", "mine", "myself"]
        if any(token.text.lower() in personal_pronouns for token in doc):
            scores["personal"] += 0.5
        
        # Check for question words (indicates factual query)
        question_words = ["what", "who", "where", "when", "why", "how"]
        if any(token.text.lower() in question_words for token in doc):
            scores["factual"] += 0.3
        
        # Check for opinion indicators
        opinion_words = ["think", "opinion", "believe", "feel", "view"]
        if any(token.lemma_.lower() in opinion_words for token in doc):
            scores["opinion"] += 0.5
        
        # Check for imperative verbs (indicates instruction)
        if len(doc) > 0 and doc[0].pos_ == "VERB":
            scores["instruction"] += 0.4
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            for key in scores:
                scores[key] /= total
        
        return scores
```

### 3. Enhance Retriever with Hybrid Extraction

Modify `ContextualRetriever` to use both regex and NLP techniques:

```python
# Update ContextualRetriever __init__ method
def __init__(
    self,
    # ... existing parameters ...
    use_nlp_extraction: bool = False,
    nlp_confidence_threshold: float = 0.3,
    nlp_model_name: str = "en_core_web_md"
):
    # ... existing initialization ...
    
    # NLP extraction parameters
    self.use_nlp_extraction = use_nlp_extraction
    self.nlp_confidence_threshold = nlp_confidence_threshold
    self.nlp_extractor = None
    self.nlp_model_name = nlp_model_name
    
    # Initialize NLP extractor if enabled
    if self.use_nlp_extraction:
        from memoryweave.utils.nlp_extraction import NLPExtractor
        self.nlp_extractor = NLPExtractor(model_name=nlp_model_name)
```

Modify the extraction method to use a hybrid approach:

```python
def _extract_personal_attributes(self, text: str) -> None:
    """
    Extract personal attributes using regex first, then NLP if needed.
    
    Args:
        text: Text to extract attributes from
    """
    if not text:
        return
        
    text_lower = text.lower()
    
    # Track confidence of regex extractions
    regex_confidence = 0.0
    extraction_count = 0
    
    # Try regex extraction first
    original_attributes = self.personal_attributes.copy()
    
    # Extract with regex
    self._extract_preferences(text_lower)
    self._extract_demographics(text_lower)
    self._extract_traits(text_lower)
    self._extract_relationships(text_lower)
    
    # Count how many new attributes were extracted
    new_attrs_count = 0
    for category in self.personal_attributes:
        if category not in original_attributes:
            new_attrs_count += len(self.personal_attributes[category])
            continue
            
        for key in self.personal_attributes[category]:
            if key not in original_attributes[category]:
                new_attrs_count += 1
    
    # Calculate regex confidence
    if new_attrs_count > 0:
        regex_confidence = min(1.0, 0.3 + (0.1 * new_attrs_count))
    
    # If confidence is low and NLP is enabled, try NLP extraction
    if regex_confidence < self.nlp_confidence_threshold and self.use_nlp_extraction and self.nlp_extractor:
        nlp_attributes = self.nlp_extractor.extract_personal_attributes(text)
        
        # Merge NLP attributes with regex attributes, prioritizing regex for conflicts
        for category in nlp_attributes:
            if category not in self.personal_attributes:
                self.personal_attributes[category] = {}
                
            for key, value in nlp_attributes[category].items():
                # Only add if not already extracted by regex
                if (category not in original_attributes or 
                    key not in original_attributes[category]):
                    self.personal_attributes[category][key] = value
```

### 4. Enhance Query Type Detection

Update the query type detection to use NLP when confidence is low:

```python
def _is_factual_query(self, query: str) -> bool:
    """
    Determine if a query is factual/general knowledge rather than personal.
    
    Args:
        query: The query text
        
    Returns:
        True if the query appears to be factual, False otherwise
    """
    # First try regex patterns for speed
    regex_is_factual = False
    regex_confidence = 0.0
    
    # Check if query matches factual patterns
    for pattern in self.factual_query_patterns:
        if pattern.search(query):
            regex_is_factual = True
            regex_confidence += 0.25  # Add confidence for each matching pattern
            
    # Check if it's a personal query (contains "my", "I", etc.)
    personal_indicators = ["my", " i ", "i'm", "i've", "i'll", "i'd", "me", "mine"]
    has_personal_indicators = any(indicator in query.lower() for indicator in personal_indicators)
    
    if has_personal_indicators:
        regex_is_factual = False
        regex_confidence = 0.7  # High confidence if personal indicators are present
    
    # If we're not confident and NLP is available, use NLP
    if regex_confidence < 0.6 and self.use_nlp_extraction and self.nlp_extractor:
        query_types = self.nlp_extractor.identify_query_type(query)
        
        # Decide based on NLP scores
        if query_types["factual"] > query_types["personal"]:
            return True
        else:
            return False
    
    return regex_is_factual
```

### 5. Integrate with Two-Stage Retrieval

Update the two-stage retrieval to incorporate NLP-extracted data:

```python
def retrieve_for_context(
    self,
    current_input: str,
    conversation_history: Optional[list[dict]] = None,
    top_k: int = 5,
    confidence_threshold: float = None,
) -> list[dict]:
    # ... existing code ...
    
    # Extract important keywords for direct reference matching
    important_keywords = self._extract_important_keywords(current_input)
    
    # Extract and update personal attributes
    # ... existing code ...
    
    # If query type adaptation is enabled and NLP is available, use more sophisticated detection
    if self.query_type_adaptation and self.use_nlp_extraction and self.nlp_extractor:
        query_types = self.nlp_extractor.identify_query_type(current_input)
        
        # Adjust threshold based on query type scores
        if query_types["factual"] > 0.7:
            # More aggressive lowering for strongly factual queries
            adjusted_threshold = max(self.min_confidence_threshold, confidence_threshold * 0.5)
            adjusted_adaptive_k_factor = max(0.05, self.adaptive_k_factor * 0.4)
        elif query_types["factual"] > query_types["personal"]:
            # Standard lowering for somewhat factual queries
            adjusted_threshold = max(self.min_confidence_threshold, confidence_threshold * 0.6)
            adjusted_adaptive_k_factor = max(0.05, self.adaptive_k_factor * 0.5)
        else:
            adjusted_threshold = confidence_threshold
            adjusted_adaptive_k_factor = self.adaptive_k_factor
    else:
        # Fall back to regex-based detection
        # ... existing code ...
    
    # ... rest of the retrieval method ...
```

### 6. Add Testing and Benchmarking for NLP Extraction

Create a dedicated test script to evaluate the NLP extraction:

```python
# test_nlp_extraction.py
import torch
from transformers import AutoModel, AutoTokenizer
from memoryweave.utils.nlp_extraction import NLPExtractor
from memoryweave.core import ContextualMemory, ContextualRetriever, MemoryEncoder

def main():
    # Initialize NLP extractor
    extractor = NLPExtractor()
    
    # Test personal attribute extraction
    test_texts = [
        "My name is Alex and I live in Seattle. I work as a software engineer.",
        "I enjoy hiking in the mountains on weekends. My favorite color is blue.",
        "My wife Sarah and I have two children, Emma and Jack.",
        "I prefer Thai food and particularly enjoy spicy curries.",
        "I graduated from Stanford with a degree in Computer Science."
    ]
    
    print("Testing NLP-based attribute extraction:")
    for text in test_texts:
        print(f"\nText: {text}")
        attributes = extractor.extract_personal_attributes(text)
        print("Extracted attributes:")
        for category, items in attributes.items():
            if items:
                print(f"  {category.capitalize()}:")
                for key, value in items.items():
                    print(f"    {key}: {value}")
    
    # Test query type identification
    test_queries = [
        "What is the capital of France?",
        "Tell me about quantum computing",
        "What is my favorite color?",
        "Where do I live?",
        "What's your opinion on climate change?",
        "Should I invest in stocks or bonds?",
        "Write a poem about mountains",
        "Summarize the main points of the article"
    ]
    
    print("\nTesting query type identification:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        query_types = extractor.identify_query_type(query)
        for qtype, score in query_types.items():
            print(f"  {qtype}: {score:.2f}")
        
        # Determine primary type
        primary_type = max(query_types.items(), key=lambda x: x[1])[0]
        print(f"  Primary type: {primary_type}")
    
    # Test comparison with regex approach
    print("\nComparing regex and NLP extraction:")
    
    # Initialize embedding model for retriever
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    class EmbeddingWrapper:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
        
        def encode(self, text):
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Mean pooling
            attention_mask = inputs["attention_mask"]
            embeddings = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            masked_embeddings = embeddings * mask
            summed = torch.sum(masked_embeddings, 1)
            counts = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = summed / counts
            
            return mean_pooled.numpy()[0]
    
    embedding_model = EmbeddingWrapper(model, tokenizer)
    
    # Create retriever with regex only
    memory = ContextualMemory(embedding_dim=model.config.hidden_size)
    regex_retriever = ContextualRetriever(
        memory=memory,
        embedding_model=embedding_model,
        use_nlp_extraction=False
    )
    
    # Create retriever with NLP enabled
    nlp_retriever = ContextualRetriever(
        memory=memory,
        embedding_model=embedding_model,
        use_nlp_extraction=True
    )
    
    # Compare attribute extraction
    for text in test_texts:
        print(f"\nText: {text}")
        
        # Reset attributes
        regex_retriever.personal_attributes = {
            "preferences": {}, 
            "demographics": {},
            "traits": {},
            "relationships": {}
        }
        
        nlp_retriever.personal_attributes = {
            "preferences": {}, 
            "demographics": {},
            "traits": {},
            "relationships": {}
        }
        
        # Extract with both methods
        regex_retriever._extract_personal_attributes(text)
        nlp_retriever._extract_personal_attributes(text)
        
        print("Regex extraction:")
        _print_attributes(regex_retriever.personal_attributes)
        
        print("NLP extraction:")
        _print_attributes(nlp_retriever.personal_attributes)

def _print_attributes(attributes):
    for category, items in attributes.items():
        if items:
            print(f"  {category.capitalize()}:")
            for key, value in items.items():
                print(f"    {key}: {value}")

if __name__ == "__main__":
    main()
```

## Implementation Steps

1. Add spaCy and necessary dependencies to the project
2. Create `nlp_extraction.py` with the NLPExtractor class
3. Modify ContextualRetriever to support hybrid extraction:
   - Add NLP extraction parameters to __init__
   - Update _extract_personal_attributes method to use both approaches
   - Enhance _is_factual_query to use NLP for ambiguous cases
   - Integrate with two-stage retrieval
4. Create test and benchmark scripts for NLP extraction
5. Update documentation to explain the hybrid approach

## Performance Considerations

- Lazy loading: Only initialize spaCy models when needed
- Configuration options: Allow users to disable NLP if performance is a concern
- Caching: Cache NLP processing results for frequently seen text
- Selective application: Use NLP only when regex confidence is low 

## Bias Mitigation Benefits

1. More flexible matching: NLP techniques are less rigid than regex patterns
2. Language understanding: Better handles variations in how people express the same concept
3. Contextual relevance: Considers broader context beyond pattern matching
4. Learning capability: spaCy models are trained on diverse texts, capturing more language patterns
5. Adaptability: Works with grammar and syntax variations rather than exact phrasings
