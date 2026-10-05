"""Read reviews from files into plain Review records."""

import csv
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Review:
    book: str
    text: str
    rating: int | None = None  # the reviewer's own 1-5 stars, when known


def load_csv(path: Path) -> list[Review]:
    """Read a CSV with columns book, rating, review. Rating may be blank."""
    with path.open(encoding="utf-8", newline="") as f:
        return [
            Review(
                book=row["book"].strip(),
                text=row["review"].strip(),
                rating=_rating(row.get("rating") or ""),
            )
            for row in csv.DictReader(f)
            if row["review"].strip()
        ]


def load_book_ratings(path: Path) -> dict[str, float]:
    """Read a CSV with columns book, avg_rating: each book's published average rating."""
    with path.open(encoding="utf-8", newline="") as f:
        return {row["book"].strip(): float(row["avg_rating"]) for row in csv.DictReader(f)}


def limit_per_book(reviews: Iterable[Review], min_reviews: int, max_reviews: int) -> list[Review]:
    """Keep books with at least min_reviews reviews, and at most max_reviews of each."""
    by_book: dict[str, list[Review]] = defaultdict(list)
    for review in reviews:
        by_book[review.book].append(review)
    return [
        review
        for book_reviews in by_book.values()
        if len(book_reviews) >= min_reviews
        for review in book_reviews[:max_reviews]
    ]


def _rating(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    rating = int(float(value))
    if not 1 <= rating <= 5:
        raise ValueError(f"rating must be 1-5, got {value!r}")
    return rating
