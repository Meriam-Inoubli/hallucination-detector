"""Substantive-keyword extraction for SUScore.

SUScore focuses uncertainty on the *substantive* tokens — named entities and
nouns — because that is where factual hallucinations show up (Zhang et al.,
EMNLP 2023). We use spaCy when it is installed, and fall back to a lightweight
heuristic so the package works with zero extra downloads.
"""

from __future__ import annotations

import re

# Very common words that are never "substantive" for our purposes.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "is", "are", "was",
    "were", "be", "been", "to", "of", "in", "on", "for", "with", "as", "by",
    "at", "from", "this", "that", "these", "those", "it", "its", "we", "you",
    "they", "he", "she", "i", "so", "not", "no", "do", "does", "did", "can",
    "will", "would", "should", "could", "have", "has", "had", "there",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
_HAS_SPACY: bool | None = None


def _spacy_available() -> bool:
    global _HAS_SPACY
    if _HAS_SPACY is None:
        try:
            import spacy  # noqa: F401

            _HAS_SPACY = True
        except ImportError:
            _HAS_SPACY = False
    return _HAS_SPACY


def extract_keywords(text: str) -> list[str]:
    """Return the substantive keywords (entities and nouns) of ``text``.

    Uses spaCy's ``en_core_web_sm`` model if available, otherwise a heuristic
    that keeps capitalized words, numbers and long non-stopwords.

    Args:
        text: The text to analyze.

    Returns:
        A list of keyword strings, in order of appearance (deduplicated,
        lower-cased for matching).
    """
    if not text or not text.strip():
        return []
    if _spacy_available():
        keywords = _spacy_keywords(text)
        if keywords:
            return keywords
    return _heuristic_keywords(text)


def _spacy_keywords(text: str) -> list[str]:
    try:
        nlp = _load_spacy_model()
    except OSError:
        return []
    doc = nlp(text)
    seen: set[str] = set()
    keywords: list[str] = []
    for token in doc:
        if token.ent_type_ or token.pos_ in {"NOUN", "PROPN", "NUM"}:
            key = token.text.lower()
            if key not in seen and len(key) > 1:
                seen.add(key)
                keywords.append(key)
    return keywords


_SPACY_MODEL = None


def _load_spacy_model():
    global _SPACY_MODEL
    if _SPACY_MODEL is None:
        import spacy

        _SPACY_MODEL = spacy.load("en_core_web_sm")
    return _SPACY_MODEL


def _heuristic_keywords(text: str) -> list[str]:
    seen: set[str] = set()
    keywords: list[str] = []
    for match in _WORD_RE.finditer(text):
        word = match.group(0)
        lower = word.lower()
        is_capitalized = word[0].isupper() and match.start() != 0
        is_substantive = is_capitalized or (lower not in _STOPWORDS and len(lower) >= 5)
        if is_substantive and lower not in seen:
            seen.add(lower)
            keywords.append(lower)
    # Also keep standalone numbers (often the "facts" that get hallucinated).
    for number in re.findall(r"\b\d+(?:\.\d+)?\b", text):
        if number not in seen:
            seen.add(number)
            keywords.append(number)
    return keywords
