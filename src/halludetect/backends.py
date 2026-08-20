"""LLM backends: the only part of the library that talks to the network.

The detectors depend on the small :class:`Backend` protocol below, so you can
plug in any provider — or a fake one in tests — without touching the detection
logic. A ready-made :class:`GeminiBackend` is provided.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Backend(Protocol):
    """Minimal surface a detector needs from a language-model provider."""

    def generate(self, prompt: str, n: int, temperature: float) -> list[str]:
        """Return ``n`` sampled completions for ``prompt``."""

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an ``(len(texts), d)`` array of embeddings."""


class MissingAPIKey(RuntimeError):
    """Raised when a live backend is used without an API key configured."""


class GeminiBackend:
    """A :class:`Backend` backed by Google Gemini.

    Args:
        model: Chat model used for generation.
        embedding_model: Model used for embeddings.
        api_key: Explicit key; defaults to ``GEMINI_API_KEY``/``GOOGLE_API_KEY``.
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        embedding_model: str = "models/text-embedding-004",
        api_key: str | None = None,
    ) -> None:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise MissingAPIKey(
                "No Gemini API key found. Set GEMINI_API_KEY (free key at "
                "https://aistudio.google.com/app/apikey)."
            )
        import google.generativeai as genai

        genai.configure(api_key=key)
        self._genai = genai
        self._model = model
        self._embedding_model = embedding_model

    def generate(self, prompt: str, n: int, temperature: float) -> list[str]:
        client = self._genai.GenerativeModel(self._model)
        out: list[str] = []
        for _ in range(n):
            try:
                response = client.generate_content(
                    prompt,
                    generation_config={"temperature": temperature, "max_output_tokens": 1024},
                )
                out.append(response.text)
            except Exception as exc:  # noqa: BLE001 - one bad sample shouldn't kill the batch
                print(f"warning: a generation failed ({exc})")
        return out

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            result = self._genai.embed_content(
                model=self._embedding_model,
                content=text,
                task_type="semantic_similarity",
            )
            vectors.append(result["embedding"])
        return np.asarray(vectors, dtype=float)
