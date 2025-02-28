import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

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
    "technology": "blue",
})

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


def main():
    console.print(Panel.fit("Testing ART-inspired Memory Clustering", style="bold cyan"))
    
    # Load embedding model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    console.print(f"Loading embedding model: [bold]{model_name}[/bold]", style="info")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    embedding_model = EmbeddingModelWrapper(model, tokenizer)
    
    # Initialize memory with ART clustering - using a lower vigilance threshold
    embedding_dim = model.config.hidden_size  # 384 for MiniLM-L6
    memory = ContextualMemory(
        embedding_dim=embedding_dim,
        use_art_clustering=True,
        vigilance_threshold=0.55,  # Lowered to encourage more clustering
        learning_rate=0.2  # Higher = faster adaptation
    )
    
    # Create encoder
    encoder = MemoryEncoder(embedding_model)
    
    # Test with different categories of information
    console.print("\n[bold]Adding memories from different categories...[/bold]")
    
    # Technology category
    tech_memories = [
        "Python is a popular programming language for AI development.",
        "TensorFlow and PyTorch are leading deep learning frameworks.",
        "GPT models use transformer architectures for natural language processing.",
        "Reinforcement learning involves agents learning through trial and error.",
        "Neural networks are composed of layers of interconnected nodes."
    ]
    
    # Travel category
    travel_memories = [
        "Paris is known as the City of Light and is famous for the Eiffel Tower.",
        "Japan's cherry blossom season typically occurs in spring.",
        "The Great Barrier Reef in Australia is the world's largest coral reef system.",
        "Venice is famous for its canals and gondola rides.",
        "Machu Picchu is an ancient Incan citadel located in Peru."
    ]
    
    # Food category
    food_memories = [
        "Sushi is a Japanese dish featuring vinegared rice and various ingredients.",
        "Italian cuisine is known for pasta, pizza, and gelato.",
        "Thai food often combines sweet, sour, salty, and spicy flavors.",
        "French pastries include croissants, macarons, and éclairs.",
        "Indian curry dishes vary widely by region and spice combinations."
    ]
    
    # Add all memories
    all_memories = []
    all_embeddings = []
    
    console.print("\n[bold]Adding technology memories...[/bold]", style="tech")
    for text in tech_memories:
        embedding, metadata = encoder.encode_concept(
            concept="technology",
            description=text
        )
        idx = memory.add_memory(embedding, text, metadata)
        all_memories.append((idx, text, "technology"))
        all_embeddings.append(embedding)
        console.print(f"  Added: {text[:50]}...", style="tech")
        
    console.print("\n[bold]Adding travel memories...[/bold]", style="travel")
    for text in travel_memories:
        embedding, metadata = encoder.encode_concept(
            concept="travel",
            description=text
        )
        idx = memory.add_memory(embedding, text, metadata)
        all_memories.append((idx, text, "travel"))
        all_embeddings.append(embedding)
        console.print(f"  Added: {text[:50]}...", style="travel")
        
    console.print("\n[bold]Adding food memories...[/bold]", style="food")
    for text in food_memories:
        embedding, metadata = encoder.encode_concept(
            concept="food",
            description=text
        )
        idx = memory.add_memory(embedding, text, metadata)
        all_memories.append((idx, text, "food"))
        all_embeddings.append(embedding)
        console.print(f"  Added: {text[:50]}...", style="food")
    
    # Print category statistics
    console.print("\n[bold]Category statistics:[/bold]", style="category")
    stats = memory.get_category_statistics()
    console.print(f"Number of categories formed: [bold]{stats['num_categories']}[/bold]", style="category")
    
    # Create a table for category statistics
    table = Table(title="Memories per Category")
    table.add_column("Category", style="magenta")
    table.add_column("Count", style="cyan")
    table.add_column("Activation", style="green")
    
    for cat_idx, count in stats['memories_per_category'].items():
        activation = stats['category_activations'].get(cat_idx, 0)
        table.add_row(
            str(cat_idx), 
            str(count), 
            f"{activation:.2f}"
        )
    
    console.print(table)
    
    # Analyze intra-category vs inter-category similarity
    if stats['num_categories'] > 1:
        console.print("\n[bold]Analyzing category coherence...[/bold]")
        
        # Calculate intra-category similarity
        intra_category_similarities = []
        for cat_idx in stats['memories_per_category'].keys():
            # Get all memories in this category
            cat_memory_indices = np.where(memory.memory_categories == cat_idx)[0]
            if len(cat_memory_indices) > 1:  # Need at least 2 memories for similarity
                cat_embeddings = memory.memory_embeddings[cat_memory_indices]
                # Calculate pairwise similarities
                similarities = np.dot(cat_embeddings, cat_embeddings.T)
                # Get upper triangle (excluding diagonal)
                upper_tri = similarities[np.triu_indices(len(cat_embeddings), k=1)]
                if len(upper_tri) > 0:
                    avg_similarity = np.mean(upper_tri)
                    intra_category_similarities.append((cat_idx, avg_similarity))
        
        # Calculate inter-category similarity
        inter_category_similarities = []
        categories = list(stats['memories_per_category'].keys())
        for i, cat1 in enumerate(categories):
            for cat2 in categories[i+1:]:
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
        
        # Print results
        if intra_category_similarities:
            console.print("\nIntra-category similarities (higher is better):")
            for cat_idx, similarity in intra_category_similarities:
                console.print(f"  Category {cat_idx}: {similarity:.3f}")
            
            avg_intra = np.mean([s for _, s in intra_category_similarities])
            console.print(f"Average intra-category similarity: {avg_intra:.3f}")
        
        if inter_category_similarities:
            console.print("\nInter-category similarities (lower is better):")
            for (cat1, cat2), similarity in inter_category_similarities:
                console.print(f"  Categories {cat1}-{cat2}: {similarity:.3f}")
            
            avg_inter = np.mean([s for _, s in inter_category_similarities])
            console.print(f"Average inter-category similarity: {avg_inter:.3f}")
            
            if intra_category_similarities:
                separation = avg_intra - avg_inter
                console.print(f"\nCategory separation (higher is better): {separation:.3f}")
    
    # Test incremental learning
    console.print("\n[bold]Testing incremental learning...[/bold]")
    
    # Add a new memory similar to an existing technology memory
    new_tech_memory = "JavaScript is a widely-used programming language for web development."
    console.print(f"\nAdding new technology memory: {new_tech_memory}", style="tech")
    
    embedding, metadata = encoder.encode_concept(
        concept="technology",
        description=new_tech_memory
    )
    idx = memory.add_memory(embedding, new_tech_memory, metadata)
    all_memories.append((idx, new_tech_memory, "technology"))
    all_embeddings.append(embedding)
    
    # Check if it was assigned to an existing category
    cat_idx = memory.memory_categories[-1]
    console.print(f"Assigned to category: {cat_idx}")
    
    # Count how many memories are in this category now
    cat_count = np.sum(memory.memory_categories == cat_idx)
    console.print(f"Category now contains {cat_count} memories")
    
    # Test retrieval with different queries
    console.print("\n[bold]Testing retrieval with different queries...[/bold]")
    
    test_queries = [
        "Tell me about programming languages",
        "What are some popular tourist destinations?",
        "I'm interested in international cuisine"
    ]
    
    for query in test_queries:
        console.print(f"\n[bold]Query:[/bold] {query}")
        
        # Get query embedding
        query_embedding = embedding_model.encode(query)
        
        # Retrieve with standard approach
        console.print("\n[bold]Standard retrieval results:[/bold]")
        standard_results = memory.retrieve_memories(
            query_embedding, 
            top_k=3, 
            use_categories=False
        )
        
        if standard_results:
            for idx, score, metadata in standard_results:
                original_idx, text, category = [(i, t, c) for i, t, c in all_memories if i == idx][0]
                style = "tech" if category == "technology" else category
                console.print(f"  [{category}] {text[:50]}... (Score: {score:.3f})", style=style)
        else:
            console.print("  No results found", style="warning")
        
        # Retrieve with category-based approach
        console.print("\n[bold]Category-based retrieval results:[/bold]")
        category_results = memory.retrieve_memories(
            query_embedding, 
            top_k=3, 
            use_categories=True
        )
        
        if category_results:
            for idx, score, metadata in category_results:
                original_idx, text, category = [(i, t, c) for i, t, c in all_memories if i == idx][0]
                style = "tech" if category == "technology" else category
                console.print(f"  [{category}] {text[:50]}... (Score: {score:.3f})", style=style)
        else:
            console.print("  No results found", style="warning")
    
    # Visualize memory clusters
    try:
        console.print("\n[bold]Visualizing memory clusters...[/bold]")
        
        # Convert embeddings to 2D using PCA
        all_embeddings_array = np.array(all_embeddings)
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(all_embeddings_array)
        
        # Create a scatter plot
        plt.figure(figsize=(10, 8))
        
        # Define colors for each category
        category_colors = {
            "technology": "blue",
            "travel": "green",
            "food": "orange"
        }
        
        # Plot each memory point
        for i, (idx, text, category) in enumerate(all_memories):
            cat_idx = memory.memory_categories[idx]
            plt.scatter(
                embeddings_2d[i, 0], 
                embeddings_2d[i, 1], 
                color=category_colors.get(category, "gray"),
                s=100,
                alpha=0.7
            )
            plt.text(
                embeddings_2d[i, 0] + 0.02, 
                embeddings_2d[i, 1] + 0.02, 
                f"C{cat_idx}", 
                fontsize=9
            )
        
        # Add category centers
        if memory.use_art_clustering and len(memory.category_prototypes) > 0:
            # Project category prototypes to 2D
            category_centers_2d = pca.transform(memory.category_prototypes)
            
            # Plot category centers
            for cat_idx, center in enumerate(category_centers_2d):
                plt.scatter(
                    center[0], 
                    center[1], 
                    color="red", 
                    marker="*", 
                    s=200, 
                    edgecolors="black"
                )
                plt.text(
                    center[0] + 0.02, 
                    center[1] + 0.02, 
                    f"Cat {cat_idx}", 
                    fontsize=12, 
                    weight="bold"
                )
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="blue", markersize=10, label='Technology'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="green", markersize=10, label='Travel'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="orange", markersize=10, label='Food'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor="red", markersize=15, label='Category Center')
        ]
        plt.legend(handles=legend_elements, loc="upper right")
        
        # Add title and labels
        plt.title("Memory Clusters in 2D Embedding Space")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig("memory_clusters.png")
        console.print("Visualization saved as 'memory_clusters.png'")
    except Exception as e:
        console.print(f"[bold red]Error creating visualization: {e}[/bold red]")
    
    console.print("\n[bold]Test completed successfully![/bold]", style="success")


if __name__ == "__main__":
    main()
