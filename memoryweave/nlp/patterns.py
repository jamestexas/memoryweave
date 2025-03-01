"""Pattern definitions for MemoryWeave NLP utilities.

This module contains regular expression patterns used for text extraction
and pattern matching in the MemoryWeave system.
"""

# Personal attribute patterns
PERSONAL_ATTRIBUTE_PATTERNS = {
    'name': [
        r'my name is (\w+)',
        r'I am ([A-Z][a-z]+ [A-Z][a-z]+)',
        r'call me (\w+)',
        r'name\'s (\w+)'
    ],
    'age': [
        r'I am (\d+) years old',
        r'I\'m (\d+) years old',
        r'I\'m (\d+)',
        r'my age is (\d+)'
    ],
    'location': [
        r'I live in ([A-Za-z\s]+)',
        r'I\'m from ([A-Za-z\s]+)',
        r'I\'m in ([A-Za-z\s]+)',
        r'I\'m currently in ([A-Za-z\s]+)',
        r'I live near ([A-Za-z\s]+)',
        r'I\'m based in ([A-Za-z\s]+)'
    ],
    'occupation': [
        r'I work as an? ([A-Za-z\s]+)',
        r'I\'m an? ([A-Za-z\s]+)(?: by profession| by trade)?',
        r'my job is ([A-Za-z\s]+)',
        r'I\'m employed as an? ([A-Za-z\s]+)'
    ],
    'favorite_color': [
        r'my favorite colou?r is ([A-Za-z]+)',
        r'I like the colou?r ([A-Za-z]+)',
        r'I love ([A-Za-z]+) colou?r'
    ],
    'hobby': [
        r'I enjoy ([A-Za-z\s]+ing)',
        r'I like to ([A-Za-z\s]+)',
        r'my hobby is ([A-Za-z\s]+ing)',
        r'in my spare time, I ([A-Za-z\s]+)'
    ],
    'family': [
        r'my (?:wife|husband|spouse) (?:is )?(?:named )?([A-Za-z]+)',
        r'I have (\d+) (?:kids|children)',
        r'my (?:son|daughter|child) (?:is )?(?:named )?([A-Za-z]+)',
        r'my (?:brother|sister|sibling) (?:is )?(?:named )?([A-Za-z]+)',
        r'my (?:mom|mother|dad|father|parent) (?:is )?(?:named )?([A-Za-z]+)'
    ],
    'education': [
        r'I studied ([A-Za-z\s]+)',
        r'I have a (?:degree|diploma) in ([A-Za-z\s]+)',
        r'I graduated from ([A-Za-z\s]+)',
        r'I went to ([A-Za-z\s]+)(?: University| College)?'
    ],
    'pet': [
        r'I have an? ([A-Za-z]+)(?: named ([A-Za-z]+))?',
        r'my ([A-Za-z]+)(?:\'s name| is named) ([A-Za-z]+)',
        r'I own an? ([A-Za-z]+)'
    ],
    'favorite_food': [
        r'my favorite food is ([A-Za-z\s]+)',
        r'I love eating ([A-Za-z\s]+)',
        r'I enjoy ([A-Za-z\s]+) the most'
    ]
}

# Query type patterns
QUERY_TYPE_PATTERNS = {
    'personal': [
        r'\b(?:my|your|I|me|mine|you|yours)\b',
        r'\b(?:remember|told|said|mentioned|talked about)\b',
        r'\b(?:like|enjoy|love|hate|prefer)\b',
        r'\b(?:favorite|opinion|think|feel|believe)\b',
        r'\b(?:family|friend|relative|parent|child|spouse)\b'
    ],
    'factual': [
        r'\b(?:what is|who is|where is|when is|why is|how is)\b',
        r'\b(?:define|explain|describe|tell me about)\b',
        r'\b(?:fact|information|knowledge|data)\b'
    ],
    'temporal': [
        r'\b(?:when|time|date|period|era|century|year|month|week|day)\b',
        r'\b(?:before|after|during|while|since|until|ago|past|future)\b',
        r'\b(?:recent|latest|newest|oldest|previous|next|last|first)\b'
    ],
    'conceptual': [
        r'\b(?:concept|idea|theory|philosophy|principle)\b',
        r'\b(?:meaning|significance|importance|implication)\b',
        r'\b(?:relationship|connection|correlation|link)\b'
    ],
    'historical': [
        r'\b(?:history|historical|ancient|medieval|modern)\b',
        r'\b(?:civilization|empire|kingdom|dynasty|era|age)\b',
        r'\b(?:revolution|war|battle|conflict|event)\b'
    ]
}

# Entity patterns
ENTITY_PATTERNS = {
    'PERSON': r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
    'LOCATION': r'\b([A-Z][a-z]+(?:,\s+[A-Z][a-z]+)*)\b',
    'ORGANIZATION': r'\b([A-Z][a-z]*(?:\s+[A-Z][a-z]*)+)\b',
    'DATE': r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b|\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:[a-z]*)?(?:,\s+\d{4})?)\b'
}

# Common stopwords to exclude from keywords
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
    'at', 'from', 'by', 'on', 'off', 'for', 'in', 'out', 'over', 'under',
    'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should',
    'can', 'could', 'may', 'might', 'must', 'to', 'of', 'with'
}