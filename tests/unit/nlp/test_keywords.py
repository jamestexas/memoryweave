from memoryweave.nlp.keywords import expand_keywords, extract_keywords, rank_keywords


def test_extract_keywords():
    # Test basic extraction
    text = "Artificial intelligence and machine learning are transforming technology"
    keywords = extract_keywords(text)

    assert "artificial" in keywords
    assert "intelligence" in keywords
    assert "machine" in keywords
    assert "learning" in keywords
    assert "transforming" in keywords
    assert "technology" in keywords

    # Test with custom stopwords
    custom_stopwords = {"artificial", "intelligence"}
    keywords = extract_keywords(text, custom_stopwords)
    assert "artificial" not in keywords
    assert "intelligence" not in keywords

    # Test min_length parameter
    keywords = extract_keywords(text, min_length=10)
    assert "artificial" not in keywords  # 10 chars
    assert "intelligence" in keywords  # 12 chars
    assert "technology" in keywords  # 10 chars

    # Test max_keywords parameter
    keywords = extract_keywords(text, max_keywords=3)
    assert len(keywords) <= 3


def test_rank_keywords():
    # Setup text with keyword repetition and position differences
    text = "Python is popular. Python has great libraries for data analysis and data visualization."
    keywords = ["python", "libraries", "data", "analysis", "visualization"]

    ranked = rank_keywords(keywords, text)

    # Check the format of results
    assert isinstance(ranked, list)
    assert all(isinstance(item, tuple) and len(item) == 2 for item in ranked)

    # Convert to dict for easier testing
    ranked_dict = dict(ranked)

    # Test frequency impact
    assert ranked_dict["python"] > ranked_dict["libraries"]  # Python appears twice
    assert ranked_dict["data"] > ranked_dict["visualization"]  # data appears twice

    # Test position impact (earlier = higher score)
    assert ranked_dict["python"] > ranked_dict["data"]  # Python appears first


def test_expand_keywords():
    keywords = ["machine"]
    word_relationships = {
        "machine": ["computer", "equipment", "device"],
        "computer": ["laptop", "desktop", "server"],
        "car": ["automobile", "vehicle"],
    }

    # Test basic expansion
    expanded = expand_keywords(keywords, word_relationships)
    assert "machine" in expanded
    assert "computer" in expanded
    assert "equipment" in expanded
    assert "device" in expanded
    assert "car" not in expanded

    # Test expansion with limit
    expanded = expand_keywords(keywords, word_relationships, expansion_count=1)
    assert "machine" in expanded
    assert "computer" in expanded
    assert "equipment" not in expanded  # Beyond expansion count

    # Test with multiple keywords
    expanded = expand_keywords(["machine", "car"], word_relationships)
    assert "computer" in expanded
    assert "automobile" in expanded
