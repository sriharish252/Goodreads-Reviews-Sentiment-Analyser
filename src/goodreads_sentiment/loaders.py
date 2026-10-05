"""Read reviews from files into plain Review records."""

import csv
import gzip
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

ENGLISH = {"eng", "en", "en-US", "en-GB", "en-CA"}  # UCSD language codes


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


def load_ucsd(reviews_path: Path, books_path: Path) -> list[Review]:
    """Read the UCSD Book Graph Goodreads files (JSON lines, optionally gzipped).

    Keeps reviews of English-language editions that have text and 1-5 stars (0 means none given),
    and groups editions of the same work into one book.
    """
    work_of, titles = {}, {}
    for book in _json_lines(books_path):
        if book["language_code"] in ENGLISH:
            work = book["work_id"] or book["book_id"]
            work_of[book["book_id"]] = work
            titles.setdefault(work, book["title"].strip())
    counts = Counter(titles.values())
    names = {w: t if counts[t] == 1 else f"{t} [{w}]" for w, t in titles.items()}
    return [
        Review(names[work_of[r["book_id"]]], r["review_text"].strip(), int(r["rating"]))
        for r in _json_lines(reviews_path)
        if r["book_id"] in work_of and 1 <= int(r["rating"]) <= 5 and r["review_text"].strip()
    ]


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


def _json_lines(path: Path) -> Iterator[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)
