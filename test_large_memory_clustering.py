import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme
from collections import Counter
import random
import re
import math
from scipy import stats

from memoryweave.core import ContextualMemory, MemoryEncoder

# Create a custom theme for consistent styling
custom_theme = Theme({
    "info": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "red bold",
    "category": "magenta",
    "score": "cyan",
    "tech": "blue",
    "travel": "green",
    "food": "yellow",
    "science": "purple",
    "history": "gold3",
    "arts": "magenta",
    "title": "bold cyan",
    "section": "bold yellow",
    "stat": "bold white",
    "vigilance": "bold magenta",
})

# Define a list of available styles for categories
AVAILABLE_STYLES = {
    "technology": "tech",
    "travel": "travel",
    "food": "food",
    "science": "science",
    "history": "history",
    "arts": "arts"
}

# Initialize rich console with our theme
console = Console(theme=custom_theme)

# Helper class for sentence embedding
class EmbeddingModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling
        attention_mask = inputs['attention_mask']
        embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / counts
        
        return mean_pooled.numpy()[0]


def generate_large_memory_dataset():
    """Generate a larger dataset of memories for testing."""
    
    # Technology memories (15)
    tech_memories = [
        "Python is a popular programming language for AI development.",
        "TensorFlow and PyTorch are leading deep learning frameworks.",
        "GPT models use transformer architectures for natural language processing.",
        "Reinforcement learning involves agents learning through trial and error.",
        "Neural networks are composed of layers of interconnected nodes.",
        "JavaScript is the primary language for web development and browser interactions.",
        "Rust provides memory safety without garbage collection.",
        "Kubernetes is an open-source container orchestration platform.",
        "Cloud computing enables on-demand access to shared computing resources.",
        "Quantum computing uses quantum bits to perform calculations.",
        "Edge computing processes data closer to where it's generated rather than in a centralized data center.",
        "Blockchain is a distributed ledger technology that enables secure transactions.",
        "Virtual reality creates an immersive, computer-generated environment.",
        "Augmented reality overlays digital content onto the real world.",
        "5G networks offer faster speeds and lower latency than previous generations."
    ]
    
    # Travel memories (10)
    travel_memories = [
        "Paris is known as the City of Light and is famous for the Eiffel Tower.",
        "Japan's cherry blossom season typically occurs in spring.",
        "The Great Barrier Reef in Australia is the world's largest coral reef system.",
        "Venice is famous for its canals and gondola rides.",
        "Machu Picchu is an ancient Incan citadel located in Peru.",
        "The Serengeti in Tanzania is known for its annual wildebeest migration.",
        "Santorini in Greece features white-washed buildings with blue domes overlooking the Aegean Sea.",
        "The Northern Lights can be seen in countries near the Arctic Circle.",
        "The Grand Canyon in Arizona is one of the most spectacular natural formations.",
        "Kyoto in Japan is home to numerous traditional temples and gardens."
    ]
    
    # Food memories (10)
    food_memories = [
        "Sushi is a Japanese dish featuring vinegared rice and various ingredients.",
        "Italian cuisine is known for pasta, pizza, and gelato.",
        "Thai food often combines sweet, sour, salty, and spicy flavors.",
        "French pastries include croissants, macarons, and éclairs.",
        "Indian curry dishes vary widely by region and spice combinations.",
        "Mexican cuisine features corn, beans, and chili peppers as staple ingredients.",
        "Mediterranean diet emphasizes olive oil, fresh vegetables, and lean proteins.",
        "Belgian chocolate is renowned for its high quality and rich flavor.",
        "Korean kimchi is a traditional side dish of fermented vegetables.",
        "Turkish baklava is a sweet pastry made with layers of filo and chopped nuts."
    ]
    
    # Science memories (8)
    science_memories = [
        "The theory of relativity was developed by Albert Einstein.",
        "DNA carries genetic information in all living organisms.",
        "The periodic table organizes chemical elements by their properties.",
        "Photosynthesis is the process by which plants convert light into energy.",
        "Black holes are regions of spacetime with gravitational acceleration so strong that nothing can escape.",
        "The Higgs boson is an elementary particle in the Standard Model of particle physics.",
        "CRISPR-Cas9 is a gene editing technology that can modify DNA sequences.",
        "The human genome contains approximately 3 billion base pairs."
    ]
    
    # History memories (7)
    history_memories = [
        "The Roman Empire was one of the largest empires of the ancient world.",
        "The Renaissance was a period of cultural, artistic, and scientific revival in Europe.",
        "The Industrial Revolution marked the transition to new manufacturing processes.",
        "World War II was a global conflict that lasted from 1939 to 1945.",
        "The Cold War was a period of geopolitical tension between the Soviet Union and the United States.",
        "The ancient Egyptian civilization developed along the Nile River.",
        "The Silk Road was a network of trade routes connecting the East and West."
    ]
    
    # Arts memories (5)
    arts_memories = [
        "The Mona Lisa was painted by Leonardo da Vinci.",
        "Beethoven composed nine symphonies despite his hearing loss.",
        "Impressionism began as a movement among Paris-based artists in the 1860s.",
        "Shakespeare wrote approximately 37 plays and 154 sonnets.",
        "Ballet originated in the Italian Renaissance courts of the 15th century."
    ]
    
    # Combine all memories with their categories
    all_memories = [
        (text, "technology") for text in tech_memories
    ] + [
        (text, "travel") for text in travel_memories
    ] + [
        (text, "food") for text in food_memories
    ] + [
        (text, "science") for text in science_memories
    ] + [
        (text, "history") for text in history_memories
    ] + [
        (text, "arts") for text in arts_memories
    ]
    
    # Shuffle the memories to avoid adding them in category blocks
    random.shuffle(all_memories)
    
    return all_memories


def calculate_entropy(category_counts):
    """Calculate the entropy of the category distribution."""
    total = sum(category_counts.values())
    probabilities = [count / total for count in category_counts.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    max_entropy = math.log2(len(category_counts))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    return entropy, normalized_entropy


def text_histogram(data, bins=10, width=50):
    """Create a text-based histogram."""
    if not data:
        return "No data to display"
        
    hist, bin_edges = np.histogram(data, bins=bins)
    max_count = max(hist)
    
    result = []
    for i, count in enumerate(hist):
        bar_length = int(count / max_count * width) if max_count > 0 else 0
        bar = '█' * bar_length
        result.append(f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}: {bar} ({count})")
    
    return '\n'.join(result)


def main():
    console.print(Panel.fit("Testing Increasing Vigilance Strategy with Large Memory Set", style="title"))
    
    # Load embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    console.print(f"Loading embedding model: [bold]{model_name}[/bold]", style="info")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    embedding_model = EmbeddingModelWrapper(model, tokenizer)
    
    # Initialize memory with increasing vigilance strategy
    embedding_dim = model.config.hidden_size  # 384 for MiniLM-L6
    memory = ContextualMemory(
        embedding_dim=embedding_dim,
        use_art_clustering=True,
        vigilance_threshold=0.5,  # Start with a low vigilance
        learning_rate=0.25,  # Higher learning rate for faster adaptation
        dynamic_vigilance=True,
        vigilance_strategy="increasing",
        min_vigilance=0.5,
        max_vigilance=0.9,
        target_categories=10  # Target for category_based strategy (not used here)
    )
    
    # Create encoder
    encoder = MemoryEncoder(embedding_model)
    
    # Generate large memory dataset
    all_memories = generate_large_memory_dataset()
    console.print(f"\nGenerated [bold]{len(all_memories)}[/bold] memories across multiple categories", style="info")
    
    # Count memories per category
    category_counts = Counter([category for _, category in all_memories])
    
    # Print category distribution
    console.print("\n[bold]Memory Distribution by Category:[/bold]", style="section")
    for category, count in category_counts.items():
        style = AVAILABLE_STYLES.get(category, "info")
        console.print(f"  {category.capitalize()}: {count} memories", style=style)
    
    # Track vigilance changes and category growth
    vigilance_values = []
    category_counts_over_time = []
    
    # Add all memories
    added_memories = []
    
    console.print("\n[bold]Adding memories with increasing vigilance...[/bold]", style="section")
    
    for i, (text, category) in enumerate(all_memories):
        # Create concept embedding
        embedding, metadata = encoder.encode_concept(
            concept=category,
            description=text
        )
        
        # Add to memory
        idx = memory.add_memory(embedding, text, metadata)
        added_memories.append((idx, text, category))
        
        # Track vigilance and category count
        vigilance_values.append(memory.vigilance_threshold)
        
        # Get category statistics
        stats = memory.get_category_statistics()
        category_counts_over_time.append(stats["num_categories"])
        
        # Show progress every 10 memories
        if (i + 1) % 10 == 0 or i == len(all_memories) - 1:
            console.print(f"  Memory #{i+1}: Vigilance = {memory.vigilance_threshold:.3f}, Categories = {stats['num_categories']}", style="vigilance")
    
    # Get final category statistics
    stats = memory.get_category_statistics()
    console.print(f"\n[bold]Final number of categories: {stats['num_categories']}[/bold] (with vigilance = {memory.vigilance_threshold:.3f})", style="category")
    
    # Analyze category distribution
    memory_per_category = list(stats['memories_per_category'].values())
    
    # Calculate distribution metrics
    avg_memories = np.mean(memory_per_category)
    median_memories = np.median(memory_per_category)
    min_memories = np.min(memory_per_category)
    max_memories = np.max(memory_per_category)
    std_memories = np.std(memory_per_category)
    
    # Calculate entropy of distribution
    entropy, normalized_entropy = calculate_entropy(stats['memories_per_category'])
    
    # Print distribution metrics
    console.print("\n[bold]Category Distribution Metrics:[/bold]", style="section")
    console.print(f"  Average memories per category: {avg_memories:.2f}", style="stat")
    console.print(f"  Median memories per category: {median_memories:.1f}", style="stat")
    console.print(f"  Minimum category size: {min_memories}", style="stat")
    console.print(f"  Maximum category size: {max_memories}", style="stat")
    console.print(f"  Standard deviation: {std_memories:.2f}", style="stat")
    console.print(f"  Distribution entropy: {entropy:.2f} (normalized: {normalized_entropy:.2f})", style="stat")
    
    # Create a table for category statistics
    table = Table(title="Memories per Category")
    table.add_column("Category", style="magenta")
    table.add_column("Count", style="cyan")
    table.add_column("Activation", style="green")
    
    # Sort by category size (descending)
    sorted_categories = sorted(stats['memories_per_category'].items(), key=lambda x: x[1], reverse=True)
    
    for cat_idx, count in sorted_categories:
        activation = stats['category_activations'].get(cat_idx, 0)
        table.add_row(
            str(cat_idx), 
            str(count), 
            f"{activation:.2f}"
        )
    
    console.print(table)
    
    # Print histogram of category sizes
    console.print("\n[bold]Histogram of Category Sizes:[/bold]", style="section")
    histogram = text_histogram(memory_per_category, bins=min(10, len(memory_per_category)), width=40)
    console.print(histogram)
    
    # Analyze category coherence
    console.print("\n[bold]Analyzing category coherence...[/bold]", style="section")
    
    # Calculate intra-category similarity for categories with multiple memories
    intra_category_similarities = []
    for cat_idx, count in stats['memories_per_category'].items():
        if count > 1:  # Need at least 2 memories for similarity
            # Get all memories in this category
            cat_memory_indices = np.where(memory.memory_categories == cat_idx)[0]
            cat_embeddings = memory.memory_embeddings[cat_memory_indices]
            
            # Calculate pairwise similarities
            similarities = np.dot(cat_embeddings, cat_embeddings.T)
            
            # Get upper triangle (excluding diagonal)
            upper_tri = similarities[np.triu_indices(len(cat_embeddings), k=1)]
            
            if len(upper_tri) > 0:
                avg_similarity = np.mean(upper_tri)
                intra_category_similarities.append((cat_idx, avg_similarity, count))
    
    # Sort by similarity (descending)
    intra_category_similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Print top 5 most coherent categories
    console.print("\nTop 5 most coherent categories:", style="info")
    for i, (cat_idx, similarity, count) in enumerate(intra_category_similarities[:5]):
        console.print(f"  Category {cat_idx}: {similarity:.3f} (size: {count})", style="category")
    
    # Print bottom 5 least coherent categories
    if len(intra_category_similarities) > 5:
        console.print("\nBottom 5 least coherent categories:", style="info")
        for i, (cat_idx, similarity, count) in enumerate(intra_category_similarities[-5:]):
            console.print(f"  Category {cat_idx}: {similarity:.3f} (size: {count})", style="category")
    
    # Calculate average intra-category similarity
    if intra_category_similarities:
        avg_intra = np.mean([s for _, s, _ in intra_category_similarities])
        console.print(f"\nAverage intra-category similarity: {avg_intra:.3f}", style="stat")
    
    # Calculate inter-category similarity (sample for efficiency)
    inter_category_similarities = []
    categories = list(stats['memories_per_category'].keys())
    
    # Sample up to 50 category pairs for efficiency
    category_pairs = []
    for i, cat1 in enumerate(categories):
        for cat2 in categories[i+1:]:
            category_pairs.append((cat1, cat2))
    
    # Randomly sample if too many pairs
    if len(category_pairs) > 50:
        category_pairs = random.sample(category_pairs, 50)
    
    for cat1, cat2 in category_pairs:
        # Get memories from both categories
        cat1_indices = np.where(memory.memory_categories == cat1)[0]
        cat2_indices = np.where(memory.memory_categories == cat2)[0]
        
        if len(cat1_indices) > 0 and len(cat2_indices) > 0:
            cat1_embeddings = memory.memory_embeddings[cat1_indices]
            cat2_embeddings = memory.memory_embeddings[cat2_indices]
            
            # Calculate cross-category similarities
            similarities = np.dot(cat1_embeddings, cat2_embeddings.T)
            avg_similarity = np.mean(similarities)
            inter_category_similarities.append(((cat1, cat2), avg_similarity))
    
    # Calculate average inter-category similarity
    if inter_category_similarities:
        avg_inter = np.mean([s for _, s in inter_category_similarities])
        console.print(f"Average inter-category similarity: {avg_inter:.3f}", style="stat")
        
        # Calculate category separation
        if intra_category_similarities:
            separation = avg_intra - avg_inter
            console.print(f"Category separation (higher is better): {separation:.3f}", style="stat")
    
    # Test retrieval with queries from different categories
    console.print("\n[bold]Testing retrieval with queries from different categories...[/bold]", style="section")
    
    test_queries = [
        # Technology queries
        "What programming languages are used in AI?",
        "Tell me about virtual and augmented reality.",
        
        # Travel queries
        "What are some famous landmarks in Europe?",
        "Tell me about natural wonders around the world.",
        
        # Food queries
        "What are some popular international cuisines?",
        "Tell me about desserts from different countries.",
        
        # Science queries
        "What are some important scientific discoveries?",
        "Tell me about genetics and DNA.",
        
        # Cross-category queries
        "How has technology changed the way we travel?",
        "What's the connection between science and cooking?",
        "How has art been influenced by historical events?"
    ]
    
    for query in test_queries:
        console.print(f"\n[bold]Query:[/bold] {query}")
        
        # Get query embedding
        query_embedding = embedding_model.encode(query)
        
        # Retrieve with category-based approach
        results = memory.retrieve_memories(
            query_embedding, 
            top_k=3, 
            use_categories=True
        )
        
        if results:
            for idx, score, metadata in results:
                original_idx, text, category = [(i, t, c) for i, t, c in added_memories if i == idx][0]
                style = AVAILABLE_STYLES.get(category, "info")
                console.print(f"  [{category}] {text[:50]}... (Score: {score:.3f})", style=style)
                
                # Show which category this memory belongs to
                cat_idx = memory.memory_categories[idx]
                console.print(f"    Memory belongs to category {cat_idx}", style="category")
        else:
            console.print("  No results found", style="warning")
    
    console.print("\n[bold]Test completed successfully![/bold]", style="success")


if __name__ == "__main__":
    main()
