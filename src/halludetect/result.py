"""The common result type returned by every detector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DetectionResult:
    """Outcome of a hallucination check.

    Attributes:
        method: Which detector produced this (``"semantic_entropy"``, ``"suscore"``,
            ``"hybrid"``).
        score: Risk score in ``[0, 1]`` — higher means more likely hallucinated.
        hallucinated: Whether ``score`` crossed the detector's threshold.
        threshold: The threshold used for the decision.
        details: Method-specific extras (clusters, per-keyword scores, ...).
    """

    method: str
    score: float
    hallucinated: bool
    threshold: float
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        flag = "⚠️  HALLUCINATION" if self.hallucinated else "✅ OK"
        return f"[{self.method}] {flag}  score={self.score:.3f} (threshold {self.threshold})"

    def as_dict(self) -> dict[str, Any]:
        """Return a plain-dict view, handy for JSON output."""
        return {
            "method": self.method,
            "score": round(self.score, 4),
            "hallucinated": self.hallucinated,
            "threshold": self.threshold,
            "details": self.details,
        }
