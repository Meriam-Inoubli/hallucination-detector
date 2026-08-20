# 🔎 halludetect

> Detect **LLM hallucinations** on any text — three research-backed methods behind one small API, running on **Google Gemini**.

[![CI](https://github.com/Meriam-Inoubli/hallucination-detector/actions/workflows/ci.yaml/badge.svg)](https://github.com/Meriam-Inoubli/hallucination-detector/actions/workflows/ci.yaml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-8E75B2?logo=googlegemini&logoColor=white)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Large language models state false things fluently. **halludetect** estimates how
likely a given answer is a hallucination — without a ground-truth reference —
using uncertainty signals from the model itself.

| Method | Signal it uses | Reference |
|--------|----------------|-----------|
| 🌀 **Semantic Entropy** | Sample the model N times; if the answers scatter across many *meanings*, it's unsure | Farquhar et al., 2024 |
| 🎯 **SUScore** | Uncertainty focused on *substantive* tokens (entities, nouns, numbers) — where facts live | Zhang et al., EMNLP 2023 |
| ⚖️ **Hybrid** | A weighted blend of both — more robust than either alone | this project |

---

## 💡 Why it exists

Most hallucination checks need a reference answer you don't have in production.
These three methods are **reference-free**: they read the model's own
(dis)agreement with itself. `halludetect` packages them cleanly:

- **Any text, any prompt** — not tied to a database or a domain.
- **Pluggable backend** — ships with Gemini; swap in your own provider via a
  tiny `Backend` protocol.
- **Offline-testable core** — the math (entropy, clustering, SUScore aggregation)
  is pure and unit-tested with **no API key**, so CI is green and free.

---

## 🚀 Quick Start

```bash
git clone https://github.com/Meriam-Inoubli/hallucination-detector.git
cd hallucination-detector
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env      # paste your free Gemini key
```

Free key: **[aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)**.

```bash
# CLI
halludetect "Who wrote the play Hamlet?"
halludetect "What is the population of the fictional city of Aldovia?" --method semantic_entropy
halludetect "Explain the Black-Scholes model" --json
```

```text
[hybrid] ✅ OK  score=0.180 (threshold 0.5)
   agreement: high
```

---

## 🧑‍💻 Library usage

```python
from halludetect import HybridDetector, SemanticEntropyDetector, GeminiBackend

backend = GeminiBackend()

# One-liner verdict
result = HybridDetector(backend).detect("Who painted the Mona Lisa?")
print(result.hallucinated, result.score)

# Score a specific answer you already have
SemanticEntropyDetector(backend).detect(
    prompt="What is the capital of Australia?",
    answer="The capital of Australia is Sydney.",   # (it isn't — expect a high score)
)
```

The **SUScore core is usable with no API key at all** — feed it samples you
already collected:

```python
from halludetect import suscore_from_samples

suscore_from_samples(
    keywords=["shakespeare", "1601"],
    samples=["Shakespeare wrote Hamlet", "Marlowe maybe", "around 1601", "unclear"],
)["suscore"]   # -> high, the facts don't hold up across samples
```

---

## 🏗️ Architecture

```
src/halludetect/
├── metrics.py           # pure math: cosine sim, cluster entropy, blending   (tested)
├── keywords.py          # substantive-keyword extraction (spaCy optional)    (tested)
├── backends.py          # Backend protocol + GeminiBackend  (the only network code)
├── semantic_entropy.py  # 🌀 SemanticEntropyDetector
├── suscore.py           # 🎯 SUScoreDetector + suscore_from_samples          (tested)
├── hybrid.py            # ⚖️ HybridDetector
└── cli.py               # halludetect command
```

Detectors depend only on the small `Backend` protocol, so tests inject a fake
backend and exercise the full pipeline **offline**.

### A note on SUScore & log-probabilities

The SUScore paper reads native token log-probabilities. Most chat APIs (Gemini
included) don't expose those, so `halludetect` approximates the same signal with
**self-consistency**: a substantive keyword that survives across independent
samples is trusted; one that flickers is uncertain. This is documented honestly
in the code — swap in a log-prob-capable backend to use exact probabilities.

---

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Your Gemini key (required for live calls) |

Optional: `pip install -e ".[spacy]"` and `python -m spacy download en_core_web_sm`
for linguistics-grade keyword extraction (a heuristic fallback works without it).

---

## 🧪 Development

```bash
pip install -e ".[dev]"
pytest -v        # runs fully offline
ruff check .
```

---

## 📚 References & citation

This project reimplements, from scratch and on Gemini, methods introduced in:

- Farquhar, S. et al. (2024). *Detecting Hallucinations in Large Language Models Using Semantic Entropy.*
- Zhang, T. et al. (2023). *Enhancing Uncertainty-Based Hallucination Detection with Stronger Focus.* EMNLP.

It grew out of the author's research on **multi-agent intelligent tutoring with
hallucination detection** (ITS 2026). If you use it, see [`CITATION.cff`](CITATION.cff).

## 📄 License

[MIT](LICENSE) © 2026 Meriam Inoubli
