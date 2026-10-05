"""SQLite storage for books, reviews and sentiment scores.

Scores are cached per (review, model), so re-running a slow model only scores new reviews.
"""

import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from goodreads_sentiment.loaders import Review

SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    id         INTEGER PRIMARY KEY,
    title      TEXT NOT NULL UNIQUE,
    avg_rating REAL CHECK (avg_rating BETWEEN 1 AND 5)  -- published average, if supplied
);
CREATE TABLE IF NOT EXISTS reviews (
    id      INTEGER PRIMARY KEY,
    book_id INTEGER NOT NULL REFERENCES books (id),
    rating  INTEGER CHECK (rating BETWEEN 1 AND 5),       -- the reviewer's own stars, if known
    text    TEXT NOT NULL,
    UNIQUE (book_id, text)
);
CREATE TABLE IF NOT EXISTS scores (
    review_id INTEGER NOT NULL REFERENCES reviews (id),
    model     TEXT NOT NULL,
    score     REAL NOT NULL CHECK (score BETWEEN -1 AND 1),
    PRIMARY KEY (review_id, model)
);
"""


@dataclass(frozen=True)
class BookStats:
    title: str
    n_reviews: int
    rating: float | None  # mean of the reviewers' stars, else the published average
    rating_source: str  # "reviews", "published" or "" when no rating is known
    sentiment: float  # mean review score in [-1, 1]


class ReviewStore:
    def __init__(self, path: Path | str) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._conn.execute("CREATE TEMP TABLE selected (review_id INTEGER PRIMARY KEY)")

    def __enter__(self) -> "ReviewStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self._conn.close()

    def add_reviews(self, reviews: Iterable[Review]) -> list[int]:
        """Store reviews (skipping ones already stored) and return their ids in input order."""
        ids = []
        with self._conn:
            for review in reviews:
                self._conn.execute(
                    "INSERT INTO books (title) VALUES (?) ON CONFLICT (title) DO NOTHING",
                    (review.book,),
                )
                (book_id,) = self._conn.execute(
                    "SELECT id FROM books WHERE title = ?", (review.book,)
                ).fetchone()
                self._conn.execute(
                    "INSERT INTO reviews (book_id, rating, text) VALUES (?, ?, ?)"
                    " ON CONFLICT (book_id, text) DO UPDATE SET rating = excluded.rating",
                    (book_id, review.rating, review.text),
                )
                (review_id,) = self._conn.execute(
                    "SELECT id FROM reviews WHERE book_id = ? AND text = ?",
                    (book_id, review.text),
                ).fetchone()
                ids.append(review_id)
        return ids

    def set_book_ratings(self, ratings: dict[str, float]) -> None:
        with self._conn:
            self._conn.executemany(
                "INSERT INTO books (title, avg_rating) VALUES (?, ?)"
                " ON CONFLICT (title) DO UPDATE SET avg_rating = excluded.avg_rating",
                ratings.items(),
            )

    def unscored(self, review_ids: Sequence[int], model: str) -> list[tuple[int, str]]:
        """Return (id, text) for the given reviews that have no score from this model yet."""
        self._select(review_ids)
        return self._conn.execute(
            "SELECT r.id, r.text FROM selected JOIN reviews r ON r.id = selected.review_id"
            " WHERE NOT EXISTS"
            " (SELECT 1 FROM scores s WHERE s.review_id = r.id AND s.model = ?)"
            " ORDER BY r.id",
            (model,),
        ).fetchall()

    def save_scores(self, model: str, scores: Iterable[tuple[int, float]]) -> None:
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO scores (review_id, model, score) VALUES (?, ?, ?)",
                ((review_id, model, score) for review_id, score in scores),
            )

    def review_scores(self, review_ids: Sequence[int], model: str) -> dict[int, float]:
        self._select(review_ids)
        return dict(
            self._conn.execute(
                "SELECT s.review_id, s.score FROM selected"
                " JOIN scores s ON s.review_id = selected.review_id AND s.model = ?",
                (model,),
            )
        )

    def book_stats(self, review_ids: Sequence[int], model: str) -> list[BookStats]:
        """Per-book averages over the given reviews, for one model's scores."""
        self._select(review_ids)
        rows = self._conn.execute(
            """
            SELECT b.title,
                   COUNT(*),
                   COALESCE(AVG(r.rating), b.avg_rating),
                   CASE WHEN COUNT(r.rating) > 0 THEN 'reviews'
                        WHEN b.avg_rating IS NOT NULL THEN 'published'
                        ELSE '' END,
                   AVG(s.score)
            FROM selected
            JOIN reviews r ON r.id = selected.review_id
            JOIN books b ON b.id = r.book_id
            JOIN scores s ON s.review_id = r.id AND s.model = ?
            GROUP BY b.id
            ORDER BY b.title
            """,
            (model,),
        )
        return [BookStats(*row) for row in rows]

    def _select(self, review_ids: Sequence[int]) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM selected")
            self._conn.executemany(
                "INSERT OR IGNORE INTO selected VALUES (?)", ((i,) for i in review_ids)
            )
