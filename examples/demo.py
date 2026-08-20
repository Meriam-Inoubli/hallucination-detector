"""Run the detectors on a couple of prompts.

Works with zero setup for the offline part; the live part runs only if a
GEMINI_API_KEY is present in your environment or .env file.

    python examples/demo.py
"""

from halludetect import GeminiBackend, HybridDetector, MissingAPIKey, suscore_from_samples


def offline_demo() -> None:
    print("=== Offline: SUScore from self-consistency (no API key) ===")
    keywords = ["shakespeare", "1601"]
    consistent = ["Shakespeare wrote Hamlet around 1600"] * 4
    shaky = ["Shakespeare wrote Hamlet", "Marlowe maybe", "written in 1601", "unclear author"]

    print("Confident answers :", suscore_from_samples(keywords, consistent)["suscore"])
    print("Shaky answers      :", suscore_from_samples(keywords, shaky)["suscore"])


def live_demo() -> None:
    print("\n=== Live: Hybrid detector via Gemini ===")
    try:
        detector = HybridDetector(GeminiBackend())
    except MissingAPIKey as exc:
        print(f"(skipped — {exc})")
        return

    for prompt in [
        "Who wrote the play Hamlet?",
        "What is the exact population of the fictional city of Aldovia?",
    ]:
        result = detector.detect(prompt)
        print(f"\nPrompt: {prompt}")
        print(result)


if __name__ == "__main__":
    offline_demo()
    live_demo()
