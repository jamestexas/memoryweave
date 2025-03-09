from memoryweave.nlp.matchers import AttributeMatcher, KeywordMatcher, RegexMatcher


def test_regex_matcher():
    matcher = RegexMatcher()

    # Add test patterns
    matcher.add_pattern("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    matcher.add_pattern("phone", r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")

    # Test finding matches
    text = "Contact john.doe@example.com or 555-123-4567"
    matches = matcher.find_matches(text)

    assert len(matches) == 2
    assert matches[0].pattern_id == "email"
    assert matches[0].text == "john.doe@example.com"
    assert matches[1].pattern_id == "phone"
    assert matches[1].text == "555-123-4567"

    # Test sorting by position
    text = "Call 555-123-4567 or email john@example.com"
    matches = matcher.find_matches(text)
    assert matches[0].pattern_id == "phone"
    assert matches[1].pattern_id == "email"

    # Test with no matches
    assert not matcher.find_matches("No emails or phones here")


def test_attribute_matcher():
    matcher = AttributeMatcher()

    # Test finding name attributes
    text = "My name is John Smith and I am 30 years old"
    results = matcher.find_attributes(text)

    assert "name" in results
    assert len(results["name"]) > 0
    assert any("John Smith" in match.text for match in results["name"])

    assert "age" in results
    assert len(results["age"]) > 0
    assert any("30" in match.text for match in results["age"])

    # Test with multiple attributes
    text = "I live in Boston and I work as a software engineer"
    results = matcher.find_attributes(text)

    assert "location" in results
    assert "occupation" in results
    assert any("Boston" in match.text for match in results["location"])
    assert any("software engineer" in match.text for match in results["occupation"])


def test_keyword_matcher():
    # Initialize with keywords
    keywords = ["python", "programming", "data"]
    matcher = KeywordMatcher(keywords)

    # Test basic matching
    text = "Python is a programming language for data science"
    matches = matcher.find_keywords(text)

    assert len(matches) == 3
    matched_words = [m.text.lower() for m in matches]
    assert "python" in matched_words
    assert "programming" in matched_words
    assert "data" in matched_words

    # Test adding keywords
    matcher.add_keyword("science")
    matches = matcher.find_keywords(text)
    assert len(matches) == 4
    assert any(m.text.lower() == "science" for m in matches)

    # Test removing keywords
    matcher.remove_keyword("python")
    matches = matcher.find_keywords(text)
    assert len(matches) == 3
    assert all(m.text.lower() != "python" for m in matches)
