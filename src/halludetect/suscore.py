"""SUScore detector (Substantive Uncertainty Score, Zhang et al., EMNLP 2023).

The paper's insight: hallucinations concentrate on *substantive* tokens (named
entities, nouns, numbers), so uncertainty should be measured there rather than
averaged over every filler word.

The reference implementation reads native token log-probabilities. Providers
that don't expose those (most chat APIs) can approximate the same signal with
**self-consistency**: sample the model a few times and measure how reliably each
substantive keyword is reproduced. A keyword that appears in every sample is
trusted; one that flickers in and out is uncertain. That approximation is what
:func:`suscore_from_samples` computes — a pure function you can test offline.
"""

from __future__ import annotations

import math

from .backends import Backend
from .keywords import extract_keywords
from .result import DetectionResult


def _idf(keyword: str) -> float:
    """A cheap length-based IDF proxy (rarer/longer terms weigh more).

    The paper pre-computes IDF from a large corpus; without one, token length is
    a reasonable, monotonic stand-in.
    """
    return 1.0 + math.log1p(len(keyword)) / 2.0


def suscore_from_samples(
    keywords: list[str],
    samples: list[str],
    gamma: float = 0.9,
    uncertain_at: float = 0.5,
) -> dict:
    """Compute a SUScore from keyword self-consistency across samples.

    Args:
        keywords: Substantive keywords taken from the answer under test.
        samples: Independently sampled generations for the same prompt.
        gamma: Propagation coefficient — a keyword inherits some uncertainty
            from the substantive keywords before it (Section 3.2 of the paper).
        uncertain_at: A keyword counts as "uncertain" once its hallucination
            score exceeds this value.

    Returns:
        A dict with the aggregate ``suscore`` in ``[0, 1]``, the count of
        uncertain keywords, and a per-keyword breakdown.
    """
    if not keywords or not samples:
        return {"suscore": 0.0, "n_keywords": len(keywords), "uncertain": [], "per_keyword": []}

    lowered = [s.lower() for s in samples]
    base_scores: list[float] = []
    for kw in keywords:
        hits = sum(1 for s in lowered if kw in s)
        consistency = hits / len(lowered)
        base_scores.append(1.0 - consistency)  # inconsistent keyword => uncertain

    # Propagate: each keyword accumulates gamma * mean(previous keyword scores).
    propagated: list[float] = []
    running_sum = 0.0
    for i, base in enumerate(base_scores):
        inherited = gamma * (running_sum / i) if i > 0 else 0.0
        propagated.append(min(base + inherited, 1.0))
        running_sum += base

    weights = [_idf(kw) for kw in keywords]
    weighted = sum(p * w for p, w in zip(propagated, weights)) / sum(weights)

    per_keyword = [
        {"keyword": kw, "score": round(p, 3)} for kw, p in zip(keywords, propagated)
    ]
    uncertain = [pk for pk in per_keyword if pk["score"] > uncertain_at]
    return {
        "suscore": float(min(weighted, 1.0)),
        "n_keywords": len(keywords),
        "uncertain": uncertain,
        "per_keyword": per_keyword,
    }


class SUScoreDetector:
    """Detect hallucinations from substantive-keyword uncertainty.

    Args:
        backend: Provider used to sample generations.
        n_samples: How many samples to draw for the self-consistency estimate.
        temperature: Sampling temperature.
        gamma: Uncertainty-propagation coefficient.
        threshold: SUScore cutoff above which we flag a hallucination.
    """

    method = "suscore"

    def __init__(
        self,
        backend: Backend,
        n_samples: int = 5,
        temperature: float = 0.7,
        gamma: float = 0.9,
        threshold: float = 0.5,
    ) -> None:
        self.backend = backend
        self.n_samples = n_samples
        self.temperature = temperature
        self.gamma = gamma
        self.threshold = threshold

    def detect(self, prompt: str, answer: str | None = None) -> DetectionResult:
        """Score ``answer`` (or the model's own first answer) for ``prompt``."""
        samples = self.backend.generate(prompt, self.n_samples, self.temperature)
        primary = answer or (samples[0] if samples else "")
        keywords = extract_keywords(primary)
        result = suscore_from_samples(keywords, samples, gamma=self.gamma)
        score = result["suscore"]
        return DetectionResult(
            self.method,
            score,
            score > self.threshold,
            self.threshold,
            {
                "n_keywords": result["n_keywords"],
                "n_uncertain": len(result["uncertain"]),
                "uncertain_keywords": [u["keyword"] for u in result["uncertain"][:8]],
                "n_samples": len(samples),
            },
        )
