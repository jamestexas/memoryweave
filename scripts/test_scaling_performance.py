"""
Test the scaling performance of the enhanced retrieval system.

This script evaluates how well the retrieval system performs as the number of 
memories increases, measuring precision, recall, F1 scores, and latency.
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import gc

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from memoryweave.core import ContextualMemory, ContextualRetriever, MemoryEncoder
from memoryweave.utils.nlp_extraction import NLPExtractor

# Create output directory
os.makedirs("test_output/scaling", exist_ok=True)

# Global spaCy model for reuse
global_nlp = None

# Helper class for sentence embedding
class EmbeddingModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.cache = {}  # Simple cache for embeddings
        
    def encode(self, text):
        # Check cache first
        if text in self.cache:
            return self.cache[text]
            
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
        
        # Cache the result
        embedding = mean_pooled.numpy()[0]
        self.cache[text] = embedding
        return embedding

def generate_synthetic_memories(count, categories):
    """Generate synthetic memories for testing."""
    memories = []
    
    # Templates for different categories
    templates = {
        "personal": [
            "My name is {name}",
            "I live in {city}, {country}",
            "My favorite color is {color}",
            "I work as a {occupation}",
            "I have a {pet_type} named {pet_name}",
            "My hobby is {hobby}"
        ],
        "factual": [
            "The capital of {country} is {city}",
            "{person} was born in {year}",
            "The {animal} is native to {region}",
            "The chemical formula for {compound} is {formula}",
            "{book} was written by {author}",
            "The {landmark} is located in {location}"
        ],
        "technical": [
            "{language} is a programming language used for {application}",
            "The {algorithm} algorithm is used for {purpose}",
            "{framework} is a framework for {technology}",
            "The {component} is a part of {system}",
            "{protocol} is a protocol used in {field}",
            "The {term} refers to {definition}"
        ]
    }
    
    # Sample data for templates
    data = {
        "name": ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Avery", "Quinn"],
        "city": ["Seattle", "Tokyo", "Paris", "London", "New York", "Berlin", "Sydney", "Toronto"],
        "country": ["USA", "Japan", "France", "UK", "Canada", "Germany", "Australia", "Italy"],
        "color": ["blue", "red", "green", "purple", "yellow", "orange", "black", "white"],
        "occupation": ["engineer", "doctor", "teacher", "artist", "scientist", "writer", "designer", "programmer"],
        "pet_type": ["dog", "cat", "bird", "fish", "hamster", "rabbit", "turtle", "ferret"],
        "pet_name": ["Max", "Luna", "Charlie", "Bella", "Oliver", "Lucy", "Leo", "Daisy"],
        "hobby": ["hiking", "painting", "reading", "cooking", "gardening", "photography", "music", "traveling"],
        "person": ["Einstein", "Newton", "Curie", "Darwin", "Tesla", "Turing", "Hawking", "Goodall"],
        "year": ["1879", "1643", "1867", "1809", "1856", "1912", "1942", "1934"],
        "animal": ["tiger", "panda", "koala", "penguin", "eagle", "dolphin", "elephant", "kangaroo"],
        "region": ["Asia", "Africa", "Australia", "Antarctica", "Europe", "North America", "South America", "Arctic"],
        "compound": ["water", "carbon dioxide", "glucose", "methane", "ammonia", "sodium chloride", "oxygen", "hydrogen"],
        "formula": ["H2O", "CO2", "C6H12O6", "CH4", "NH3", "NaCl", "O2", "H2"],
        "book": ["1984", "Moby Dick", "Pride and Prejudice", "The Great Gatsby", "To Kill a Mockingbird", "War and Peace", "Hamlet", "Don Quixote"],
        "author": ["Orwell", "Melville", "Austen", "Fitzgerald", "Lee", "Tolstoy", "Shakespeare", "Cervantes"],
        "landmark": ["Eiffel Tower", "Statue of Liberty", "Great Wall", "Taj Mahal", "Colosseum", "Pyramids", "Big Ben", "Mount Rushmore"],
        "location": ["Paris", "New York", "China", "India", "Rome", "Egypt", "London", "South Dakota"],
        "language": ["Python", "JavaScript", "Java", "C++", "Rust", "Go", "Swift", "Kotlin"],
        "application": ["web development", "data science", "mobile apps", "game development", "systems programming", "cloud computing", "machine learning", "IoT"],
        "algorithm": ["sorting", "search", "clustering", "hashing", "encryption", "compression", "pathfinding", "optimization"],
        "purpose": ["organizing data", "finding information", "grouping similar items", "fast lookups", "securing data", "reducing size", "navigation", "finding the best solution"],
        "framework": ["React", "TensorFlow", "Django", "Spring Boot", "Angular", "Flutter", "Scikit-learn", "Express"],
        "technology": ["frontend", "machine learning", "web backends", "enterprise applications", "SPAs", "mobile development", "data science", "Node.js applications"],
        "component": ["CPU", "GPU", "RAM", "SSD", "motherboard", "power supply", "network card", "cooling system"],
        "system": ["computer", "game console", "smartphone", "server", "IoT device", "embedded system", "smart home", "vehicle"],
        "protocol": ["HTTP", "TCP/IP", "SMTP", "FTP", "SSH", "Bluetooth", "WiFi", "MQTT"],
        "field": ["web", "networking", "email", "file transfer", "secure connections", "short-range wireless", "wireless networking", "IoT communication"],
        "term": ["algorithm", "data structure", "API", "IDE", "compiler", "database", "repository", "framework"],
        "definition": ["step-by-step procedure", "organized data container", "interface for software", "development environment", "code translator", "data storage system", "code storage", "reusable codebase"]
    }
    
    # Generate memories
    for i in range(count):
        category = np.random.choice(categories)
        template = np.random.choice(templates[category])
        
        # Fill in the template with random data
        memory_text = template
        for key in data:
            if "{" + key + "}" in memory_text:
                memory_text = memory_text.replace("{" + key + "}", np.random.choice(data[key]))
        
        memories.append({
            "text": memory_text,
            "category": category
        })
    
    return memories

def generate_test_queries(memory_data, query_count=10):
    """Generate test queries based on memories."""
    queries = []
    
    # Sample memories to create queries from
    sampled_indices = np.random.choice(len(memory_data), min(query_count, len(memory_data)), replace=False)
    
    for idx in sampled_indices:
        memory = memory_data[idx]
        text = memory["text"]
        category = memory["category"]
        
        if category == "personal":
            # For personal memories, create "what is my X" type queries
            if "name is" in text:
                queries.append({"query": "What is my name?", "expected": text, "category": category})
            elif "live in" in text:
                queries.append({"query": "Where do I live?", "expected": text, "category": category})
            elif "favorite color" in text:
                queries.append({"query": "What is my favorite color?", "expected": text, "category": category})
            elif "work as" in text:
                queries.append({"query": "What is my job?", "expected": text, "category": category})
            elif "have a" in text and "named" in text:
                pet_type = text.split("have a ")[1].split(" named")[0]
                queries.append({"query": f"Do I have a {pet_type}?", "expected": text, "category": category})
            elif "hobby is" in text:
                queries.append({"query": "What are my hobbies?", "expected": text, "category": category})
        elif category == "factual":
            # For factual memories, create fact-based queries
            if "capital of" in text:
                country = text.split("capital of ")[1].split(" is")[0]
                queries.append({"query": f"What is the capital of {country}?", "expected": text, "category": category})
            elif "born in" in text:
                person = text.split(" was born")[0]
                queries.append({"query": f"When was {person} born?", "expected": text, "category": category})
            elif "native to" in text:
                animal = text.split("The ")[1].split(" is native")[0]
                queries.append({"query": f"Where is the {animal} native to?", "expected": text, "category": category})
            elif "chemical formula" in text:
                compound = text.split("formula for ")[1].split(" is")[0]
                queries.append({"query": f"What is the chemical formula for {compound}?", "expected": text, "category": category})
            elif "written by" in text:
                book = text.split(" was written")[0]
                queries.append({"query": f"Who wrote {book}?", "expected": text, "category": category})
            elif "located in" in text:
                landmark = text.split("The ")[1].split(" is located")[0]
                queries.append({"query": f"Where is the {landmark} located?", "expected": text, "category": category})
        else:  # technical
            # For technical memories, create technical queries
            if "programming language" in text:
                language = text.split(" is a programming")[0]
                queries.append({"query": f"What is {language} used for?", "expected": text, "category": category})
            elif "algorithm is used" in text:
                algorithm = text.split("The ")[1].split(" algorithm")[0]
                queries.append({"query": f"What is the {algorithm} algorithm used for?", "expected": text, "category": category})
            elif "framework for" in text:
                framework = text.split(" is a framework")[0]
                queries.append({"query": f"What is {framework} a framework for?", "expected": text, "category": category})
            elif "part of" in text:
                component = text.split("The ")[1].split(" is a part")[0]
                queries.append({"query": f"What system is the {component} a part of?", "expected": text, "category": category})
            elif "protocol used" in text:
                protocol = text.split(" is a protocol")[0]
                queries.append({"query": f"What is {protocol} used for?", "expected": text, "category": category})
            elif "refers to" in text:
                term = text.split("The ")[1].split(" refers")[0]
                queries.append({"query": f"What does the term {term} refer to?", "expected": text, "category": category})
    
    return queries[:query_count]  # Ensure we return at most query_count queries

def populate_memory(memory, encoder, memory_data):
    """Populate memory with data."""
    print(f"Populating memory with {len(memory_data)} items...")
    for item in tqdm(memory_data):
        embedding, metadata = encoder.encode_concept(
            concept=item["category"],
            description=item["text"],
            related_concepts=[item["category"]]
        )
        memory.add_memory(embedding, item["text"], metadata)

def test_retrieval_performance(retriever, queries):
    """Test retrieval performance on a set of queries."""
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    retrieval_times = []
    
    for query in tqdm(queries):
        # Time the retrieval operation
        start_time = time.time()
        retrieved = retriever.retrieve_for_context(query["query"], top_k=5)
        retrieval_time = time.time() - start_time
        retrieval_times.append(retrieval_time)
        
        # Extract retrieved texts
        retrieved_texts = [item.get("text", "") or item.get("content", "") or item.get("description", "") 
                          for item in retrieved]
        
        # Check if expected answer is in retrieved texts
        expected = query["expected"]
        found = any(expected in text for text in retrieved_texts)
        
        # Calculate precision and recall
        precision = 1/len(retrieved) if found else 0
        recall = 1 if found else 0
        
        # Calculate F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
    
    # Calculate averages
    query_count = len(queries)
    avg_precision = total_precision / query_count if query_count else 0
    avg_recall = total_recall / query_count if query_count else 0
    avg_f1 = total_f1 / query_count if query_count else 0
    avg_retrieval_time = sum(retrieval_times) / query_count if query_count else 0
    
    return {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
        "avg_retrieval_time": avg_retrieval_time,
        "retrieval_times": retrieval_times
    }

def test_scaling_performance(embedding_model, memory_sizes=[100, 500, 1000, 2000, 5000]):
    """Test retrieval performance as memory size increases."""
    categories = ["personal", "factual", "technical"]
    results = {}
    
    print(f"Testing scaling performance with memory sizes: {memory_sizes}")
    
    # Initialize spaCy once for all tests
    try:
        import spacy
        global global_nlp
        print("Loading spaCy model once for all tests")
        global_nlp = spacy.load("en_core_web_sm")
    except:
        print("Could not load spaCy model, will use fallback methods")
    
    for size in memory_sizes:
        print(f"\nTesting with {size} memories...")
        
        # Generate synthetic memories and queries
        memory_data = generate_synthetic_memories(size, categories)
        queries = generate_test_queries(memory_data, query_count=20)
        
        # Initialize memory and components
        memory_dim = embedding_model.encode("test").shape[0]
        memory = ContextualMemory(embedding_dim=memory_dim)
        encoder = MemoryEncoder(embedding_model)
        
        # Populate memory
        populate_memory(memory, encoder, memory_data)
        
        # Test with standard retriever
        print("Testing standard retrieval performance...")
        standard_retriever = ContextualRetriever(
            memory=memory,
            embedding_model=embedding_model,
            use_two_stage_retrieval=True,
            query_type_adaptation=True,
            semantic_coherence_check=True,
            adaptive_retrieval=True,
            personal_query_threshold=0.5,
            factual_query_threshold=0.2,
            adaptive_k_factor=0.15
        )
        
        standard_performance = test_retrieval_performance(standard_retriever, queries)
        
        # Test with optimized retriever
        print("Testing optimized retrieval performance...")
        # Import the optimized retriever from test_optimized_retrieval
        from scripts.test_optimized_retrieval import OptimizedContextualRetriever
        
        optimized_retriever = OptimizedContextualRetriever(
            memory=memory,
            embedding_model=embedding_model,
            use_two_stage_retrieval=True,
            query_type_adaptation=True,
            semantic_coherence_check=True,
            adaptive_retrieval=True,
            personal_query_threshold=0.5,
            factual_query_threshold=0.2,
            adaptive_k_factor=0.15,
            use_clustering=True,
            cluster_count=min(50, size // 20),
            use_prefiltering=True,
            prefilter_method="keyword"
        )
        
        optimized_performance = test_retrieval_performance(optimized_retriever, queries)
        
        results[size] = {
            "standard": standard_performance,
            "optimized": optimized_performance,
            "memory_size": size,
            "query_count": len(queries)
        }
        
        print(f"Results for {size} memories:")
        print(f"  Standard:  Precision={standard_performance['precision']:.3f}, Recall={standard_performance['recall']:.3f}, F1={standard_performance['f1']:.3f}, Time={standard_performance['avg_retrieval_time']:.3f}s")
        print(f"  Optimized: Precision={optimized_performance['precision']:.3f}, Recall={optimized_performance['recall']:.3f}, F1={optimized_performance['f1']:.3f}, Time={optimized_performance['avg_retrieval_time']:.3f}s")
        
        # Clear memory to avoid OOM
        gc.collect()
    
    # Save results
    with open("test_output/scaling/scaling_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def plot_scaling_results(results):
    """Plot the scaling results."""
    sizes = sorted([int(size) for size in results.keys()])
    
    # Create figure
    plt.figure(figsize=(15, 12))
    
    # Plot F1 scores
    plt.subplot(2, 2, 1)
    plt.plot(sizes, [results[size]["standard"]["f1"] for size in sizes], 'b-o', label="Standard")
    plt.plot(sizes, [results[size]["optimized"]["f1"] for size in sizes], 'r-o', label="Optimized")
    plt.xlabel('Memory Size')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. Memory Size')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot precision
    plt.subplot(2, 2, 2)
    plt.plot(sizes, [results[size]["standard"]["precision"] for size in sizes], 'b-o', label="Standard")
    plt.plot(sizes, [results[size]["optimized"]["precision"] for size in sizes], 'r-o', label="Optimized")
    plt.xlabel('Memory Size')
    plt.ylabel('Precision')
    plt.title('Precision vs. Memory Size')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot recall
    plt.subplot(2, 2, 3)
    plt.plot(sizes, [results[size]["standard"]["recall"] for size in sizes], 'b-o', label="Standard")
    plt.plot(sizes, [results[size]["optimized"]["recall"] for size in sizes], 'r-o', label="Optimized")
    plt.xlabel('Memory Size')
    plt.ylabel('Recall')
    plt.title('Recall vs. Memory Size')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot retrieval time
    plt.subplot(2, 2, 4)
    plt.plot(sizes, [results[size]["standard"]["avg_retrieval_time"] for size in sizes], 'b-o', label="Standard")
    plt.plot(sizes, [results[size]["optimized"]["avg_retrieval_time"] for size in sizes], 'r-o', label="Optimized")
    plt.xlabel('Memory Size')
    plt.ylabel('Average Retrieval Time (seconds)')
    plt.title('Retrieval Time vs. Memory Size')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("test_output/scaling/scaling_results.png")
    plt.close()  # Close the figure to free memory

def main():
    """Run the scaling performance test."""
    print("MemoryWeave Scaling Performance Test")
    print("===================================")
    
    # Load embedding model
    print("Loading embedding model...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    embedding_model = EmbeddingModelWrapper(model, tokenizer)
    
    # Get memory sizes from command line or use defaults
    import argparse
    parser = argparse.ArgumentParser(description='Test retrieval scaling performance.')
    parser.add_argument('--sizes', type=int, nargs='+', default=[100, 500, 1000, 2000, 5000],
                        help='Memory sizes to test')
    args = parser.parse_args()
    
    # Run test
    results = test_scaling_performance(embedding_model, args.sizes)
    
    # Plot results
    plot_scaling_results(results)
    
    print("\nScaling test completed. Results saved to test_output/scaling directory.")

if __name__ == "__main__":
    main()
