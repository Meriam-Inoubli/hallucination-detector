"""Pure, dependency-light math shared by the detectors.

Everything here is deterministic and network-free, so it can be unit-tested
without an API key. The LLM calls live in :mod:`halludetect.backends`.
"""

from __future__ import annotations

import math

import numpy as np


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Return the pairwise cosine-similarity matrix of a set of vectors.

    Args:
        embeddings: Array of shape ``(n, d)``.

    Returns:
        A ``(n, n)`` symmetric matrix of cosine similarities in ``[-1, 1]``.
    """
    matrix = np.asarray(embeddings, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = matrix / norms
    return normed @ normed.T


def shannon_entropy(distribution: list[float] | np.ndarray, base: float | None = None) -> float:
    """Shannon entropy of a probability distribution.

    Args:
        distribution: Non-negative weights; they are normalized to sum to 1.
        base: Log base. ``None`` (default) uses natural log (nats).

    Returns:
        The entropy. Returns ``0.0`` for an empty or single-outcome input.
    """
    probs = np.asarray([p for p in distribution if p > 0], dtype=float)
    if probs.size <= 1:
        return 0.0
    probs = probs / probs.sum()
    entropy = float(-np.sum(probs * np.log(probs)))
    if base is not None:
        entropy /= math.log(base)
    return entropy


def cluster_entropy(embeddings: np.ndarray, eps: float = 0.25, min_samples: int = 1) -> dict:
    """Cluster responses by meaning and measure the entropy of the clusters.

    This is the practical, embedding-based approximation of *semantic entropy*
    (Farquhar et al., 2024): responses that mean the same thing land in the same
    DBSCAN cluster (cosine distance), and a spread across many clusters — high
    entropy — signals that the model is not committing to one answer.

    Args:
        embeddings: Array ``(n, d)`` of response embeddings.
        eps: DBSCAN neighbourhood radius in cosine distance (``1 - similarity``).
        min_samples: DBSCAN ``min_samples``. ``1`` keeps every response in a
            cluster (no noise), which is the sensible default for a handful of
            generations.

    Returns:
        A dict with ``entropy`` (nats), ``n_clusters``, ``labels`` and the
        normalized ``distribution`` over clusters.
    """
    from sklearn.cluster import DBSCAN

    matrix = np.asarray(embeddings, dtype=float)
    n = len(matrix)
    if n < 2:
        return {"entropy": 0.0, "n_clusters": 1, "labels": [0] * n, "distribution": [1.0]}

    distance = np.clip(1.0 - cosine_similarity_matrix(matrix), 0.0, 2.0)
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit_predict(distance)

    counts: dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1  # noise (-1) counts as its own group
    total = sum(counts.values())
    distribution = [c / total for c in counts.values()]

    return {
        "entropy": shannon_entropy(distribution),
        "n_clusters": len(counts),
        "labels": labels.tolist(),
        "distribution": distribution,
    }


def entropy_confidence(entropy: float, n_samples: int) -> float:
    """Map an entropy value to a ``[0, 1]`` confidence (1 = fully consistent).

    Confidence is ``1 - entropy / log(n_samples)``, i.e. entropy scaled by the
    maximum possible entropy for ``n_samples`` distinct clusters.
    """
    if n_samples < 2:
        return 1.0
    max_entropy = math.log(n_samples)
    return float(1.0 - min(entropy / max_entropy, 1.0)) if max_entropy > 0 else 0.0


def normalize_entropy(entropy: float, n_samples: int) -> float:
    """Scale an entropy value into ``[0, 1]`` by its theoretical maximum."""
    if n_samples < 2:
        return 0.0
    max_entropy = math.log(n_samples)
    return float(min(entropy / max_entropy, 1.0)) if max_entropy > 0 else 0.0


def combine_scores(
    entropy_norm: float,
    suscore: float,
    entropy_weight: float = 0.6,
    suscore_weight: float = 0.4,
) -> float:
    """Weighted combination of the two normalized signals (both in ``[0, 1]``).

    Args:
        entropy_norm: Normalized semantic-entropy score.
        suscore: SUScore, already in ``[0, 1]``.
        entropy_weight: Weight on the entropy signal.
        suscore_weight: Weight on the SUScore signal.

    Returns:
        The weighted hybrid score in ``[0, 1]``.

    Raises:
        ValueError: If the weights do not sum to a positive number.
    """
    total = entropy_weight + suscore_weight
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    return (entropy_weight * entropy_norm + suscore_weight * suscore) / total
