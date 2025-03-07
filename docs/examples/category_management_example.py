"""
Example demonstrating the use of the CategoryManager component.

This example shows how to use the CategoryManager component to categorize
memories and enhance retrieval performance.
"""

from sentence_transformers import SentenceTransformer

from memoryweave.components.category_manager import CategoryManager
from memoryweave.components.memory_encoding import MemoryEncoder
from memoryweave.storage.refactored.memory_store import StandardMemoryStore

# Create components
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
memory_store = StandardMemoryStore()
encoder = MemoryEncoder(embedding_model)
category_manager = CategoryManager(memory_store)

# Initialize category manager with custom parameters
category_manager.initialize(
    {
        "vigilance_threshold": 0.85,  # Higher = more categories (more specific)
        "consolidation_threshold": 0.8,  # Threshold for merging similar categories
        "embedding_dim": embedding_model.get_sentence_embedding_dimension(),
        "min_category_size": 3,  # Minimum category size for consolidation
        "memory_store": memory_store,
    }
)

# Create sample topic groups for demonstration
topics = {
    "AI": [
        "Neural networks are computational systems inspired by the human brain.",
        "Deep learning models can extract patterns from large datasets.",
        "Generative models can create realistic text, images and audio.",
        "Reinforcement learning allows agents to learn through trial and error.",
    ],
    "History": [
        "Ancient Rome was founded according to legend in 753 BCE.",
        "The Byzantine Empire was the eastern continuation of the Roman Empire.",
        "The Renaissance was a period of cultural rebirth in Europe.",
        "The Industrial Revolution began in Great Britain in the late 18th century.",
    ],
    "Science": [
        "Quantum mechanics describes nature at the atomic and subatomic scales.",
        "General relativity explains gravity as a geometric property of space and time.",
        "DNA contains the genetic instructions for the development of all organisms.",
        "The periodic table organizes chemical elements by atomic number and properties.",
    ],
}

# Add memories and categorize them
memory_ids = {}
for topic, sentences in topics.items():
    topic_ids = []
    for sentence in sentences:
        # Encode and add to memory store
        embedding = encoder.encode_text(sentence)
        memory_id = memory_store.add(embedding, sentence, {"topic": topic})

        # Add to category system
        category_id = category_manager.add_to_category(memory_id, embedding)
        topic_ids.append((memory_id, category_id))

    memory_ids[topic] = topic_ids

# Print categorization results
print("=== Categorization Results ===")
for topic, ids in memory_ids.items():
    print(f"\nTopic: {topic}")
    for memory_id, category_id in ids:
        print(f"  Memory {memory_id} → Category {category_id}")

# Check if topics were grouped into the same categories
categories = {}
for memory_id, category_id in sum(memory_ids.values(), []):
    if category_id not in categories:
        categories[category_id] = []
    categories[category_id].append(memory_id)

print("\n=== Categories ===")
for category_id, members in categories.items():
    print(f"Category {category_id}: {len(members)} members")
    for memory_id in members[:2]:  # Show first two examples
        memory = memory_store.get(memory_id)
        print(f"  - {memory.content[:50]}...")

# Consolidate similar categories
if len(categories) > 1:
    print("\n=== Attempting Category Consolidation ===")
    consolidated = category_manager.consolidate_categories()

    if consolidated:
        print(f"Consolidated categories: {consolidated}")

        # Check updated categories
        categories_after = {}
        for memory_id, _ in sum(memory_ids.values(), []):
            category_id = category_manager.get_category(memory_id)
            if category_id not in categories_after:
                categories_after[category_id] = []
            categories_after[category_id].append(memory_id)

        print("\n=== Categories After Consolidation ===")
        for category_id, members in categories_after.items():
            print(f"Category {category_id}: {len(members)} members")
    else:
        print("No categories were consolidated (similarity below threshold)")

# Use category manager for query enhancement
print("\n=== Query Enhancement ===")
query_texts = {
    "AI": "How do neural networks learn patterns?",
    "History": "Tell me about ancient Rome.",
    "Science": "Explain quantum mechanics.",
}

for _topic, query in query_texts.items():
    # Encode query
    query_embedding = encoder.encode_text(query)

    # Get query category
    query_result = category_manager.process_query(query, {"query_embedding": query_embedding})
    matched_category = query_result.get("category_match", -1)
    similarity = query_result.get("category_similarity", 0)

    print(f"\nQuery: '{query}'")
    print(f"Matched Category: {matched_category} (similarity: {similarity:.4f})")

    if matched_category >= 0:
        # Get memories from the same category
        category_members = category_manager.get_category_members(matched_category)
        print(f"Found {len(category_members)} memories in the same category")

        # Show some examples
        for memory_id in category_members[:2]:
            memory = memory_store.get(memory_id)
            print(f"  - {memory.content}")

# Get category statistics
stats = category_manager.get_statistics()
print("\n=== Category Statistics ===")
print(f"Total categories: {stats['total_categories']}")
print(f"Total memories categorized: {stats['total_categorized']}")
print(f"New categories created: {stats['new_categories_created']}")
print(f"Average category size: {stats['average_category_size']:.2f} memories")
