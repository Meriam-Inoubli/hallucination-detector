"""Offline tests for the pure math — no API key or network needed."""

import math

import numpy as np
import pytest

from halludetect.metrics import (
    cluster_entropy,
    combine_scores,
    cosine_similarity_matrix,
    entropy_confidence,
    normalize_entropy,
    shannon_entropy,
)


def test_cosine_similarity_identity_and_orthogonality():
    """Identical vectors score 1.0; orthogonal vectors score 0.0."""
    vecs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    sim = cosine_similarity_matrix(vecs)
    assert sim[0, 1] == pytest.approx(1.0)
    assert sim[0, 2] == pytest.approx(0.0)


def test_shannon_entropy_bounds():
    """Uniform over k outcomes has entropy log(k); a sure thing has 0."""
    assert shannon_entropy([1.0]) == 0.0
    assert shannon_entropy([0.5, 0.5]) == pytest.approx(math.log(2))
    assert shannon_entropy([1.0, 1.0, 1.0, 1.0]) == pytest.approx(math.log(4))


def test_cluster_entropy_all_identical_is_confident():
    """Identical embeddings collapse to one cluster with zero entropy."""
    embeddings = np.tile([1.0, 0.0, 0.0], (5, 1))
    result = cluster_entropy(embeddings)
    assert result["n_clusters"] == 1
    assert result["entropy"] == 0.0


def test_cluster_entropy_scattered_is_uncertain():
    """Mutually dissimilar embeddings split into clusters and raise entropy."""
    embeddings = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [-1.0, 0, 0]])
    result = cluster_entropy(embeddings, eps=0.25)
    assert result["n_clusters"] > 1
    assert result["entropy"] > 0.0


def test_normalize_and_confidence_are_complementary():
    """Normalized entropy and confidence move in opposite directions."""
    entropy = math.log(4)  # max for 4 samples
    assert normalize_entropy(entropy, 4) == pytest.approx(1.0)
    assert entropy_confidence(entropy, 4) == pytest.approx(0.0)
    assert normalize_entropy(0.0, 4) == 0.0
    assert entropy_confidence(0.0, 4) == pytest.approx(1.0)


def test_combine_scores_weighting():
    """The blend respects the configured weights and normalizes them."""
    assert combine_scores(1.0, 0.0, 0.6, 0.4) == pytest.approx(0.6)
    assert combine_scores(0.0, 1.0, 0.6, 0.4) == pytest.approx(0.4)
    assert combine_scores(1.0, 1.0, 3.0, 1.0) == pytest.approx(1.0)


def test_combine_scores_rejects_zero_weights():
    """Non-positive total weight is a programming error."""
    with pytest.raises(ValueError):
        combine_scores(0.5, 0.5, 0.0, 0.0)
