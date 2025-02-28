"""
Test script for NLP-based extraction utilities.

This script tests the extraction capabilities of the NLPExtractor class,
which uses regex patterns with optional NLP enhancement when available.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from memoryweave.utils.nlp_extraction import NLPExtractor

# Initialize rich console
console = Console()


def test_personal_attribute_extraction():
    """Test personal attribute extraction."""
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
    
    console.print("\n[bold blue]Testing personal attribute extraction:[/bold blue]")
    
    for text in test_texts:
        # Create a table for the results
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Category", style="dim")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        
        # Extract attributes
        attributes = extractor.extract_personal_attributes(text)
        
        # Add attributes to table
        has_attributes = False
        for category, items in attributes.items():
            if items:
                has_attributes = True
                if isinstance(items, dict):
                    for key, value in items.items():
                        table.add_row(category.capitalize(), key, str(value))
                elif isinstance(items, list):
                    for item in items:
                        table.add_row(category.capitalize(), "", str(item))
        
        # Display results
        console.print(Panel(f"[yellow]Text:[/yellow] {text}", title="Input", border_style="blue"))
        if has_attributes:
            console.print(table)
        else:
            console.print("[italic]No attributes extracted[/italic]")
        console.print()


def test_query_type_identification():
    """Test query type identification."""
    # Initialize NLP extractor
    extractor = NLPExtractor()
    
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
    
    console.print("\n[bold blue]Testing query type identification:[/bold blue]")
    
    for query in test_queries:
        # Create a table for the results
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Query Type", style="cyan")
        table.add_column("Score", style="green")
        
        # Identify query type
        query_types = extractor.identify_query_type(query)
        
        # Determine primary type
        primary_type = max(query_types.items(), key=lambda x: x[1])[0]
        
        # Add scores to table
        for qtype, score in query_types.items():
            if qtype == primary_type:
                table.add_row(f"[bold]{qtype}[/bold]", f"[bold]{score:.2f}[/bold]")
            else:
                table.add_row(qtype, f"{score:.2f}")
        
        # Display results
        console.print(Panel(f"[yellow]Query:[/yellow] {query}", title="Input", border_style="blue"))
        console.print(table)
        console.print(f"[bold green]Primary type:[/bold green] {primary_type}")
        console.print()


def main():
    """Run all tests."""
    console.print("[bold]NLP Extraction Test[/bold]", style="white on blue")
    
    # Check if spaCy is available
    try:
        import spacy
        console.print("[green]spaCy is available[/green]")
        try:
            nlp = spacy.load("en_core_web_sm")
            console.print("[green]en_core_web_sm model is available[/green]")
            console.print(f"Model pipeline: {', '.join(nlp.pipe_names)}")
        except OSError as e:
            console.print(f"[yellow]en_core_web_sm model is not available: {e}[/yellow]")
            console.print("[yellow]Using fallback extraction methods[/yellow]")
    except ImportError:
        console.print("[yellow]spaCy is not available[/yellow]")
        console.print("[yellow]Using fallback extraction methods[/yellow]")
    
    test_personal_attribute_extraction()
    test_query_type_identification()


if __name__ == "__main__":
    main()
