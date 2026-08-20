"""Offline tests for SUScore aggregation and keyword extraction."""

from halludetect.keywords import extract_keywords
from halludetect.suscore import suscore_from_samples


def test_consistent_keyword_is_trusted():
    """A keyword present in every sample gets a low uncertainty score."""
    result = suscore_from_samples(
        keywords=["shakespeare"],
        samples=["Shakespeare wrote it", "It was Shakespeare", "Shakespeare, of course"],
    )
    assert result["per_keyword"][0]["score"] == 0.0
    assert result["suscore"] == 0.0
    assert result["uncertain"] == []


def test_flickering_keyword_is_uncertain():
    """A keyword that appears in none of the samples scores maximally uncertain."""
    result = suscore_from_samples(
        keywords=["marlowe"],
        samples=["Shakespeare wrote it", "It was Shakespeare", "Definitely Shakespeare"],
    )
    assert result["per_keyword"][0]["score"] == 1.0
    assert result["suscore"] > 0.5
    assert result["uncertain"]


def test_empty_inputs_are_safe():
    """No keywords or no samples yields a zero score, not an error."""
    assert suscore_from_samples([], ["a"])["suscore"] == 0.0
    assert suscore_from_samples(["x"], [])["suscore"] == 0.0


def test_keyword_extraction_heuristic_keeps_entities_and_numbers():
    """The zero-dependency fallback keeps proper nouns and numbers."""
    keywords = extract_keywords("Shakespeare wrote 37 plays in London.")
    assert "shakespeare" in keywords
    assert "london" in keywords
    assert "37" in keywords
    assert "in" not in keywords  # stopword dropped
