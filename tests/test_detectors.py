"""Detector wiring tests using a fake backend — still no network."""

import re

import numpy as np

from halludetect import HybridDetector, SemanticEntropyDetector, SUScoreDetector


class FakeBackend:
    """A deterministic backend: canned generations and keyword-based embeddings.

    Each response is embedded by a one-hot vector keyed on its first word, so
    responses that start with the same word are "semantically identical".
    """

    def __init__(self, responses):
        self._responses = responses
        self._vocab: dict[str, int] = {}

    def generate(self, prompt, n, temperature):
        return list(self._responses)

    def embed(self, texts):
        vectors = []
        for text in texts:
            words = re.findall(r"[a-z]+", text.lower())
            head = words[0] if words else ""
            idx = self._vocab.setdefault(head, len(self._vocab))
            vectors.append(idx)
        dim = len(self._vocab)
        onehot = np.zeros((len(texts), dim))
        for row, idx in enumerate(vectors):
            onehot[row, idx] = 1.0
        return onehot


def test_semantic_entropy_flags_disagreement():
    """Responses with different meanings push entropy over the threshold."""
    backend = FakeBackend(["Paris is the capital", "Berlin is the capital", "Rome is it"])
    result = SemanticEntropyDetector(backend, threshold=0.4).detect("capital of France?")
    assert result.method == "semantic_entropy"
    assert result.hallucinated is True
    assert result.details["n_clusters"] > 1


def test_semantic_entropy_confident_when_answers_agree():
    """Identical-meaning responses yield zero entropy and no flag."""
    backend = FakeBackend(["Paris.", "Paris is the capital", "Paris, obviously"])
    result = SemanticEntropyDetector(backend, threshold=0.4).detect("capital of France?")
    assert result.hallucinated is False
    assert result.score == 0.0


def test_suscore_detector_runs_end_to_end():
    """The SUScore detector returns a well-formed result via the backend."""
    backend = FakeBackend(["Paris is the capital of France"] * 4)
    result = SUScoreDetector(backend).detect("capital of France?")
    assert result.method == "suscore"
    assert 0.0 <= result.score <= 1.0


def test_hybrid_blends_both_methods():
    """The hybrid result carries both sub-results and a blended score."""
    backend = FakeBackend(["Paris is the capital", "Berlin is the capital"])
    result = HybridDetector(backend).detect("capital of France?")
    assert result.method == "hybrid"
    assert "semantic_entropy" in result.details
    assert "suscore" in result.details
    assert 0.0 <= result.score <= 1.0
