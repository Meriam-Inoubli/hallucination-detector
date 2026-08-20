"""halludetect — detect LLM hallucinations on any text.

Three research-backed methods behind one small API:

* :class:`~halludetect.semantic_entropy.SemanticEntropyDetector` — Farquhar et al. 2024
* :class:`~halludetect.suscore.SUScoreDetector` — Zhang et al., EMNLP 2023
* :class:`~halludetect.hybrid.HybridDetector` — a weighted blend of both

Example:
    >>> from halludetect import HybridDetector, GeminiBackend
    >>> detector = HybridDetector(GeminiBackend())
    >>> result = detector.detect("Who wrote the play Hamlet?")
    >>> print(result)
"""

from __future__ import annotations

from .backends import Backend, GeminiBackend, MissingAPIKey
from .hybrid import HybridDetector
from .result import DetectionResult
from .semantic_entropy import SemanticEntropyDetector
from .suscore import SUScoreDetector, suscore_from_samples

__version__ = "0.1.0"

__all__ = [
    "Backend",
    "DetectionResult",
    "GeminiBackend",
    "HybridDetector",
    "MissingAPIKey",
    "SUScoreDetector",
    "SemanticEntropyDetector",
    "__version__",
    "suscore_from_samples",
]
