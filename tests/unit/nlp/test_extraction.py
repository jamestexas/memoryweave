from memoryweave.nlp.extraction import NLPExtractor


def test_extract_personal_attributes():
    extractor = NLPExtractor()

    # Test color preference extraction
    text = "my favorite color is blue"
    attributes = extractor.extract_personal_attributes(text)
    assert any(
        attr.attribute == "preferences_color" and attr.value == "blue" for attr in attributes
    )

    # Test location extraction
    text = "I live in Seattle"
    attributes = extractor.extract_personal_attributes(text)
    assert any(
        attr.attribute == "demographics_location" and attr.value == "Seattle" for attr in attributes
    )

    # Test relationship extraction (wife case from code comment)
    text = "my wife's name is Sarah"
    attributes = extractor.extract_personal_attributes(text)
    assert any(
        attr.attribute == "relationships_family"
        and isinstance(attr.value, dict)
        and attr.value.get("wife") == "Sarah"
        for attr in attributes
    )

    # Test empty text
    assert not extractor.extract_personal_attributes("")


def test_extract_important_keywords():
    extractor = NLPExtractor()

    # Test with specific query from code comment
    query = "what is my favorite color"
    keywords = extractor.extract_important_keywords(query)
    assert "color" in keywords

    # Test with location query
    query = "where do i live"
    keywords = extractor.extract_important_keywords(query)
    assert "location" in keywords

    # Test filtering of stopwords
    query = "tell me about the history of artificial intelligence"
    keywords = extractor.extract_important_keywords(query)
    assert "artificial" in keywords
    assert "intelligence" in keywords
    assert "history" in keywords
    assert "the" not in keywords
    assert "of" not in keywords

    # Test with empty text
    assert not extractor.extract_important_keywords("")


def test_identify_query_type():
    extractor = NLPExtractor()

    # Test personal query detection
    scores = extractor.identify_query_type("what is my favorite color")
    assert scores["personal"] > scores["factual"]

    # Test factual query detection
    scores = extractor.identify_query_type("what is the capital of France")
    assert scores["factual"] > scores["personal"]

    # Test with questions
    scores = extractor.identify_query_type("why do birds fly south?")
    assert scores["factual"] > 0.3

    # Test instruction query
    scores = extractor.identify_query_type("please write me a summary")
    assert scores["instruction"] > 0.3
