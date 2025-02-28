"""
Test script for NLP-based extraction utilities.

This script tests the extraction capabilities of the NLPExtractor class,
which uses NLP techniques with optional spaCy enhancement when available.
"""

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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
        "I graduated from Stanford with a degree in Computer Science.",
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
        "Summarize the main points of the article",
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


def compare_extraction_methods():
    """Compare extraction with and without spaCy."""
    # Create two extractors
    extractor_with_spacy = None
    extractor_basic = NLPExtractor()

    # Remember the current state
    is_spacy_originally_available = extractor_basic.is_spacy_available

    # Force basic extractor to not use spaCy by replacing its NLP component
    extractor_basic.is_spacy_available = False
    extractor_basic.nlp = None

    # Try to create a spaCy-enabled extractor
    try:
        import spacy

        try:
            spacy.load("en_core_web_sm")
            extractor_with_spacy = NLPExtractor(model_name="en_core_web_sm")
            # Ensure spaCy is used
            if not extractor_with_spacy.is_spacy_available:
                console.print(
                    "[yellow]Could not enable spaCy on second extractor - comparison skipped[/yellow]"
                )
                return
        except OSError:
            console.print("[yellow]Cannot load spaCy model - comparison skipped[/yellow]")
    except ImportError:
        console.print("[yellow]spaCy not available - comparison skipped[/yellow]")

    if not extractor_with_spacy:
        return

    # Test texts
    test_texts = [
        "I live in New York and work as a data scientist.",
        "My wife Emma and I enjoy traveling to Europe every summer.",
        "I prefer documentaries over action movies and my favorite food is Italian pasta.",
    ]

    console.print("\n[bold blue]Comparing extraction methods:[/bold blue]")

    for text in test_texts:
        # Extract with both methods
        basic_attrs = extractor_basic.extract_personal_attributes(text)
        spacy_attrs = extractor_with_spacy.extract_personal_attributes(text)

        # Create comparison table
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Category", style="dim")
        table.add_column("Key", style="cyan")
        table.add_column("Basic NLP", style="green")
        table.add_column("spaCy NLP", style="blue")

        # Collect all keys from both extractions
        all_categories = set(basic_attrs.keys()) | set(spacy_attrs.keys())

        # Create rows
        rows_added = False
        for category in all_categories:
            basic_cat = basic_attrs.get(category, {})
            spacy_cat = spacy_attrs.get(category, {})

            all_keys = set()
            if isinstance(basic_cat, dict):
                all_keys |= set(basic_cat.keys())
            if isinstance(spacy_cat, dict):
                all_keys |= set(spacy_cat.keys())

            for key in all_keys:
                basic_val = basic_cat.get(key, "")
                spacy_val = spacy_cat.get(key, "")

                if basic_val or spacy_val:
                    rows_added = True
                    table.add_row(category.capitalize(), key, str(basic_val), str(spacy_val))

        # Display results
        console.print(Panel(f"[yellow]Text:[/yellow] {text}", title="Input", border_style="blue"))
        if rows_added:
            console.print(table)
        else:
            console.print("[italic]No attributes extracted by either method[/italic]")
        console.print()

    # Restore original state
    extractor_basic.is_spacy_available = is_spacy_originally_available


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
    compare_extraction_methods()


if __name__ == "__main__":
    main()
