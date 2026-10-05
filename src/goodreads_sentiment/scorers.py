"""Sentiment models. Each one turns review texts into scores from -1 (negative) to +1 (positive)."""

from collections.abc import Callable, Sequence
from typing import Protocol

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# A Hugging Face text-classification pipeline called with top_k=None: per text, every label's score.
Classifier = Callable[..., list[list[dict]]]


class Scorer(Protocol):
    name: str  # short label for tables and charts
    key: str  # identifies the model in the score cache

    def score(self, texts: Sequence[str]) -> list[float]: ...


class VaderScorer:
    """Lexicon and rule based: the compound polarity score, already in [-1, 1]."""

    name, key = "VADER", "vader"

    def __init__(self) -> None:
        self._analyser = SentimentIntensityAnalyzer()

    def score(self, texts: Sequence[str]) -> list[float]:
        return [self._analyser.polarity_scores(text)["compound"] for text in texts]


class TransformerScorer:
    """A fine-tuned transformer: P(positive) - P(negative). Long reviews are cut to 512 tokens."""

    name = "Transformer"

    def __init__(self, model: str, classify: Classifier | None = None, batch_size: int = 16):
        self.key = model
        self._classify = classify or _load_pipeline(model)
        self._batch_size = batch_size

    def score(self, texts: Sequence[str]) -> list[float]:
        results = self._classify(list(texts), batch_size=self._batch_size)
        return [_positive_minus_negative(labels) for labels in results]


def _positive_minus_negative(labels: list[dict]) -> float:
    probs = {item["label"].lower(): item["score"] for item in labels}
    return probs.get("positive", 0.0) - probs.get("negative", 0.0)


def _load_pipeline(model: str) -> Classifier:
    try:
        from transformers import pipeline
    except ImportError:
        raise SystemExit(
            "The transformer model needs extra packages: uv sync --extra transformer"
        ) from None
    return pipeline("text-classification", model=model, top_k=None, truncation=True)
