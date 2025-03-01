"""
Keyword expansion component for MemoryWeave.

This component implements sophisticated keyword expansion for queries,
improving recall in retrieval by including variants, synonyms, and
related terms.
"""

from typing import Any, Dict, List, Optional, Set

from memoryweave.components.base import Component


class KeywordExpander(Component):
    """
    Expands keywords with variants and synonyms to improve retrieval.
    
    This component implements sophisticated keyword expansion that includes:
    - Handling singular/plural forms (including irregular plurals)
    - Adding common synonyms and related terms
    - Domain-specific expansions for certain categories
    """
    
    def __init__(self):
        """Initialize the keyword expander component."""
        self.enable_expansion = True
        self.max_expansions_per_keyword = 5
        self.synonyms = {}
        self.initialize_synonym_map()
        
        # Common irregular plurals
        self.irregular_plurals = {
            # singular: plural
            "child": "children",
            "person": "people",
            "man": "men",
            "woman": "women",
            "foot": "feet",
            "tooth": "teeth",
            "goose": "geese",
            "mouse": "mice",
            "ox": "oxen",
            "leaf": "leaves",
            "life": "lives",
            "knife": "knives",
            "wife": "wives",
            "wolf": "wolves",
            "half": "halves",
            "elf": "elves",
            "loaf": "loaves",
            "potato": "potatoes",
            "tomato": "tomatoes",
            "cactus": "cacti",
            "focus": "foci",
            "fungus": "fungi",
            "nucleus": "nuclei",
            "syllabus": "syllabi",
            "analysis": "analyses",
            "diagnosis": "diagnoses",
            "oasis": "oases",
            "thesis": "theses",
            "crisis": "crises",
            "phenomenon": "phenomena",
            "criterion": "criteria",
            "datum": "data",
            "bacterium": "bacteria",
            "medium": "media",
        }
        
        # Add reverse mapping for plural to singular
        self.plural_to_singular = {v: k for k, v in self.irregular_plurals.items()}
    
    def initialize_synonym_map(self):
        """Initialize the synonym map with common synonyms and related terms."""
        # General synonyms and related terms
        self.synonyms = {
            # Colors
            "red": ["crimson", "scarlet", "ruby"],
            "blue": ["azure", "navy", "cobalt"],
            "green": ["emerald", "lime", "olive"],
            "yellow": ["gold", "amber", "lemon"],
            "purple": ["violet", "lavender", "indigo"],
            
            # Actions
            "run": ["sprint", "jog", "dash"],
            "walk": ["stroll", "hike", "trek"],
            "talk": ["speak", "chat", "converse"],
            "eat": ["consume", "dine", "devour"],
            "sleep": ["rest", "nap", "slumber"],
            
            # Emotions
            "happy": ["joyful", "glad", "pleased"],
            "sad": ["unhappy", "depressed", "gloomy"],
            "angry": ["furious", "irate", "mad"],
            "afraid": ["scared", "frightened", "terrified"],
            "surprised": ["amazed", "astonished", "shocked"],
            
            # Common objects
            "car": ["vehicle", "automobile", "auto"],
            "house": ["home", "residence", "dwelling"],
            "book": ["text", "novel", "publication"],
            "phone": ["telephone", "cell", "mobile"],
            "computer": ["pc", "laptop", "desktop"],
            
            # Programming terms
            "function": ["method", "procedure", "routine"],
            "variable": ["var", "field", "attribute"],
            "algorithm": ["procedure", "routine", "process"],
            "database": ["db", "datastore", "repository"],
            "interface": ["api", "ui", "protocol"],
            
            # Technology
            "internet": ["web", "net", "network"],
            "email": ["mail", "message", "e-mail"],
            "software": ["program", "app", "application"],
            "hardware": ["equipment", "device", "gear"],
            "website": ["site", "webpage", "web page"],
            
            # Time
            "today": ["now", "currently", "presently"],
            "yesterday": ["recently", "before", "previously"],
            "tomorrow": ["soon", "later", "next"],
            
            # Size
            "big": ["large", "huge", "enormous"],
            "small": ["tiny", "little", "miniature"],
            
            # Quality
            "good": ["great", "excellent", "superior"],
            "bad": ["poor", "terrible", "awful"],
            
            # Common modifiers
            "very": ["extremely", "highly", "greatly"],
            "many": ["numerous", "several", "multiple"],
            "few": ["some", "couple", "handful"],
        }
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.enable_expansion = config.get("enable_expansion", True)
        self.max_expansions_per_keyword = config.get("max_expansions_per_keyword", 5)
        
        # Add custom synonyms if provided
        custom_synonyms = config.get("custom_synonyms", {})
        if custom_synonyms:
            self.synonyms.update(custom_synonyms)
    
    def process(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process data by expanding keywords.
        
        Args:
            data: Input data dictionary
            context: Processing context
            
        Returns:
            Updated data with expanded keywords
        """
        if not self.enable_expansion:
            return data
        
        # Get original keywords from context
        original_keywords = set(context.get("important_keywords", []))
        if not original_keywords:
            return data
        
        # Expand keywords
        expanded_keywords = self.expand_keywords(original_keywords)
        
        # Update data with expanded keywords
        data["original_keywords"] = original_keywords
        data["expanded_keywords"] = expanded_keywords
        
        # Store in context as well
        context["original_keywords"] = original_keywords
        context["expanded_keywords"] = expanded_keywords
        
        return data
    
    def expand_keywords(self, keywords: Set[str]) -> Set[str]:
        """
        Expand a set of keywords using various expansion techniques.
        
        Args:
            keywords: Original set of keywords
            
        Returns:
            Expanded set of keywords including original keywords
        """
        if not keywords:
            return set()
        
        expanded = set(keywords)  # Start with original keywords
        
        for keyword in list(keywords):
            keyword_lowercase = keyword.lower()
            
            # Add singular/plural forms
            singular, plural = self._get_singular_plural(keyword_lowercase)
            if singular and singular != keyword_lowercase:
                expanded.add(singular)
            if plural and plural != keyword_lowercase:
                expanded.add(plural)
            
            # Add synonyms if available
            if keyword_lowercase in self.synonyms:
                synonyms = self.synonyms[keyword_lowercase]
                # Limit the number of synonyms to avoid too much noise
                for synonym in synonyms[:self.max_expansions_per_keyword]:
                    expanded.add(synonym)
        
        return expanded
    
    def _get_singular_plural(self, word: str) -> tuple[Optional[str], Optional[str]]:
        """
        Get the singular and plural forms of a word.
        
        Args:
            word: The word to get forms for
            
        Returns:
            Tuple of (singular_form, plural_form), either may be None
        """
        # Check irregular forms first
        if word in self.irregular_plurals:
            return word, self.irregular_plurals[word]
        
        if word in self.plural_to_singular:
            return self.plural_to_singular[word], word
        
        # Handle regular forms
        if word.endswith('s'):
            # Could be plural, try singular form by removing 's'
            singular = word[:-1]
            return singular, word
        else:
            # Likely singular, add 's' for plural
            plural = word + 's'
            return word, plural