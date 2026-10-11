import sys

import pytest

from goodreads_sentiment.scorers import TransformerScorer, VaderScorer


def fake_classifier(texts, batch_size):
    """Stands in for a Hugging Face pipeline: positive when the text says 'good'."""
    p = [0.9 if "good" in text else 0.2 for text in texts]
    return [[{"label": "POSITIVE", "score": x}, {"label": "NEGATIVE", "score": 1 - x}] for x in p]


def test_vader_scores_polarity_in_range():
    good, bad = VaderScorer().score(["What a wonderful, joyful book!", "Awful and boring."])

    assert 0 < good <= 1
    assert -1 <= bad < 0


def test_transformer_score_is_positive_minus_negative_probability():
    scorer = TransformerScorer("some/model", classify=fake_classifier)

    assert scorer.score(["good", "meh"]) == pytest.approx([0.8, -0.6])
    assert scorer.key == "some/model"


def test_transformer_ignores_neutral_label_of_three_class_models():
    def classify(texts, batch_size):
        return [
            [
                {"label": "positive", "score": 0.5},
                {"label": "neutral", "score": 0.3},
                {"label": "negative", "score": 0.2},
            ]
        ]

    assert TransformerScorer("m", classify=classify).score(["x"]) == pytest.approx([0.3])


def test_missing_transformer_extra_gives_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(SystemExit, match="--extra transformer"):
        TransformerScorer("some/model")
