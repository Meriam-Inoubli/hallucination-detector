"""Command-line interface for halludetect.

Examples:
    halludetect "Who wrote the play Hamlet?"
    halludetect "What is the capital of Australia?" --method semantic_entropy
    halludetect "Explain X" --answer "Some answer to score" --json
"""

from __future__ import annotations

import argparse
import json
import sys

from .backends import GeminiBackend, MissingAPIKey
from .hybrid import HybridDetector
from .semantic_entropy import SemanticEntropyDetector
from .suscore import SUScoreDetector

_DETECTORS = {
    "hybrid": HybridDetector,
    "semantic_entropy": SemanticEntropyDetector,
    "suscore": SUScoreDetector,
}


def _force_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="halludetect", description="Detect LLM hallucinations.")
    parser.add_argument("prompt", help="The prompt/question to check")
    parser.add_argument("--answer", help="A specific answer to score (optional)")
    parser.add_argument(
        "--method", choices=list(_DETECTORS), default="hybrid", help="Detection method"
    )
    parser.add_argument("--samples", type=int, default=6, help="Number of samples to draw")
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    _force_utf8()
    args = build_parser().parse_args(argv)
    try:
        backend = GeminiBackend()
    except MissingAPIKey as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1

    detector_cls = _DETECTORS[args.method]
    kwarg = "n_samples"
    detector = detector_cls(backend, **{kwarg: args.samples})

    print(f"🔎 Checking with '{args.method}' ({args.samples} samples)...\n")
    result = detector.detect(args.prompt, args.answer)

    if args.json:
        print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
    else:
        print(result)
        for key, value in result.details.items():
            if not isinstance(value, (dict, list)):
                print(f"   {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
