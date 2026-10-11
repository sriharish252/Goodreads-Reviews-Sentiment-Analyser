"""Find books whose star rating and review sentiment disagree.

Ratings (1-5 stars) and sentiment (-1 to 1) are on different scales, and each sentiment model
has its own bias: VADER leans positive and the transformer pushes scores towards -1 or +1. So
both are standardised across the books in a run (z-scores), and a book's divergence is

    divergence = z(sentiment) - z(rating)

in standard deviations: positive when reviews read warmer than the stars suggest, negative when
they read colder. A book is flagged when |divergence| >= 1.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import correlation, fmean, pstdev

from goodreads_sentiment.db import BookStats

THRESHOLD = 1.0
MIN_BOOKS = 3  # z-scores over fewer books are meaningless


@dataclass(frozen=True)
class BookResult:
    title: str
    n_reviews: int
    rating: float | None
    rating_source: str
    sentiment: float
    divergence: float | None  # None when the book has no rating or too few books are rated
    flagged: bool


@dataclass(frozen=True)
class ModelResult:
    model: str
    books: list[BookResult]
    correlation: float | None  # Pearson r between book rating and book sentiment
    threshold: float = THRESHOLD

    @property
    def compared(self) -> list[BookResult]:
        return [b for b in self.books if b.divergence is not None]

    @property
    def flagged(self) -> list[BookResult]:
        return [b for b in self.books if b.flagged]

    @property
    def share_flagged(self) -> float | None:
        return len(self.flagged) / len(self.compared) if self.compared else None


def compare(model: str, stats: Sequence[BookStats], threshold: float = THRESHOLD) -> ModelResult:
    rated = [s for s in stats if s.rating is not None]
    ratings = [s.rating for s in rated]
    sentiments = [s.sentiment for s in rated]
    divergence: dict[str, float] = {}
    r = None
    if len(rated) >= MIN_BOOKS and pstdev(ratings) > 0 and pstdev(sentiments) > 0:
        z_ratings, z_sentiments = _standardise(ratings), _standardise(sentiments)
        divergence = {
            s.title: zs - zr for s, zr, zs in zip(rated, z_ratings, z_sentiments, strict=True)
        }
        r = correlation(ratings, sentiments)

    books = [
        BookResult(
            title=s.title,
            n_reviews=s.n_reviews,
            rating=s.rating,
            rating_source=s.rating_source,
            sentiment=s.sentiment,
            divergence=divergence.get(s.title),
            flagged=abs(divergence.get(s.title, 0.0)) >= threshold,
        )
        for s in stats
    ]
    return ModelResult(model, books, r, threshold)


def agreement(a: Mapping[int, float], b: Mapping[int, float]) -> tuple[float, float]:
    """Per-review Pearson r and share of reviews given the same polarity by two models."""
    ids = sorted(a.keys() & b.keys())
    xs, ys = [a[i] for i in ids], [b[i] for i in ids]
    same = sum((x >= 0) == (y >= 0) for x, y in zip(xs, ys, strict=True))
    return correlation(xs, ys), same / len(ids)


def _standardise(values: Sequence[float]) -> list[float]:
    mean, sd = fmean(values), pstdev(values)
    return [(v - mean) / sd for v in values]
