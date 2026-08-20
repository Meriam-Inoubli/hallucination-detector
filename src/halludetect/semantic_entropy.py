"""Semantic-entropy detector (Farquhar et al., 2024).

Idea: ask the model the same thing several times. If the answers cluster into
one meaning, the model is confident; if they scatter across many meanings, the
high *semantic entropy* flags a likely hallucination.
"""

from __future__ import annotations

from .backends import Backend
from .metrics import cluster_entropy, entropy_confidence, normalize_entropy
from .result import DetectionResult


class SemanticEntropyDetector:
    """Detect hallucinations from the semantic spread of repeated generations.

    Args:
        backend: Provider used for generation and embeddings.
        n_samples: How many answers to sample.
        temperature: Sampling temperature (higher surfaces disagreement).
        eps: DBSCAN cosine-distance radius for clustering meanings.
        min_samples: DBSCAN ``min_samples``.
        threshold: Normalized-entropy cutoff above which we flag a hallucination.
    """

    method = "semantic_entropy"

    def __init__(
        self,
        backend: Backend,
        n_samples: int = 6,
        temperature: float = 1.0,
        eps: float = 0.25,
        min_samples: int = 1,
        threshold: float = 0.5,
    ) -> None:
        self.backend = backend
        self.n_samples = n_samples
        self.temperature = temperature
        self.eps = eps
        self.min_samples = min_samples
        self.threshold = threshold

    def detect(self, prompt: str, answer: str | None = None) -> DetectionResult:
        """Check ``prompt`` (optionally scoring a specific ``answer``).

        Args:
            prompt: The question or instruction given to the model.
            answer: An answer to include in the cluster set. If omitted, only
                the freshly sampled generations are compared.

        Returns:
            A :class:`DetectionResult` for the semantic-entropy method.
        """
        responses = self.backend.generate(prompt, self.n_samples, self.temperature)
        if answer:
            responses = [answer, *responses]
        return self._score(responses)

    def _score(self, responses: list[str]) -> DetectionResult:
        responses = [r for r in responses if r and r.strip()]
        if len(responses) < 2:
            return DetectionResult(
                self.method, 0.0, False, self.threshold,
                {"note": "not enough responses to compare", "n_responses": len(responses)},
            )
        embeddings = self.backend.embed(responses)
        clustering = cluster_entropy(embeddings, eps=self.eps, min_samples=self.min_samples)
        score = normalize_entropy(clustering["entropy"], len(responses))
        return DetectionResult(
            self.method,
            score,
            score > self.threshold,
            self.threshold,
            {
                "entropy": round(clustering["entropy"], 4),
                "n_clusters": clustering["n_clusters"],
                "n_responses": len(responses),
                "confidence": round(entropy_confidence(clustering["entropy"], len(responses)), 4),
                "distribution": [round(p, 3) for p in clustering["distribution"]],
            },
        )
