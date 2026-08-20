"""Hybrid detector: combine semantic entropy and SUScore.

The two methods catch different failures — semantic entropy sees when the model
can't commit to one meaning, SUScore sees when specific facts are shaky — so a
weighted blend is more robust than either alone.
"""

from __future__ import annotations

from .backends import Backend
from .metrics import combine_scores
from .result import DetectionResult
from .semantic_entropy import SemanticEntropyDetector
from .suscore import SUScoreDetector


class HybridDetector:
    """Weighted combination of the entropy and SUScore signals.

    Args:
        backend: Provider shared by both sub-detectors.
        entropy_weight: Weight on the semantic-entropy score.
        suscore_weight: Weight on the SUScore.
        threshold: Cutoff on the blended score.
        n_samples: Samples drawn (shared setting for both sub-detectors).
    """

    method = "hybrid"

    def __init__(
        self,
        backend: Backend,
        entropy_weight: float = 0.6,
        suscore_weight: float = 0.4,
        threshold: float = 0.5,
        n_samples: int = 6,
    ) -> None:
        self.entropy_weight = entropy_weight
        self.suscore_weight = suscore_weight
        self.threshold = threshold
        self.entropy = SemanticEntropyDetector(backend, n_samples=n_samples)
        self.suscore = SUScoreDetector(backend, n_samples=n_samples)

    def detect(self, prompt: str, answer: str | None = None) -> DetectionResult:
        """Run both detectors and return their blended verdict."""
        entropy_result = self.entropy.detect(prompt, answer)
        suscore_result = self.suscore.detect(prompt, answer)
        score = combine_scores(
            entropy_result.score,
            suscore_result.score,
            self.entropy_weight,
            self.suscore_weight,
        )
        agree = entropy_result.hallucinated == suscore_result.hallucinated
        return DetectionResult(
            self.method,
            score,
            score > self.threshold,
            self.threshold,
            {
                "semantic_entropy": entropy_result.as_dict(),
                "suscore": suscore_result.as_dict(),
                "agreement": "high" if agree else "mixed",
            },
        )
