"""analyse: score reviews, find books whose ratings and reviews disagree, and chart them."""

import argparse
import csv
import sys
from collections.abc import Sequence
from itertools import batched
from pathlib import Path

from goodreads_sentiment.analysis import ModelResult, agreement, compare
from goodreads_sentiment.chart import save_chart
from goodreads_sentiment.config import Settings, load_settings
from goodreads_sentiment.db import ReviewStore
from goodreads_sentiment.loaders import (
    Review,
    limit_per_book,
    load_book_ratings,
    load_csv,
    load_ucsd,
)
from goodreads_sentiment.scorers import Scorer, TransformerScorer, VaderScorer

CHUNK = 256  # reviews scored and saved at a time, so an interrupted run keeps its progress
MAX_ROWS = 20


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    sys.stdout.reconfigure(errors="replace")
    settings = load_settings()

    reviews, book_ratings = _load(args.input, args.books)
    reviews = limit_per_book(reviews, args.min_reviews, args.max_reviews)
    if not reviews:
        sys.exit(f"No books with at least {args.min_reviews} reviews in {args.input}")

    scorers = _scorers(args.model, settings)
    with ReviewStore(settings.db_path) as store:
        ids = store.add_reviews(reviews)
        store.set_book_ratings(book_ratings)
        results = []
        for scorer in scorers:
            _score_missing(store, ids, scorer)
            results.append(compare(scorer.name, store.book_stats(ids, scorer.key)))
        models_agree = (
            agreement(*(store.review_scores(ids, s.key) for s in scorers))
            if len(scorers) == 2
            else None
        )

    _print_report(results, models_agree)
    out = settings.output_dir
    _write_csv(results, out / "summary.csv")
    print(f"\nPer-book results: {out / 'summary.csv'}")
    if all(r.compared for r in results):
        print(f"Chart: {save_chart(results, out / 'divergence.png')}")
    else:
        print("No chart: needs star ratings for at least 3 books (see --books).")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyse", description="Find books whose star ratings and review sentiment disagree."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="reviews CSV (book,rating,review) or UCSD goodreads_reviews_*.json.gz",
    )
    parser.add_argument(
        "--books",
        type=Path,
        help="CSV of published ratings (book,avg_rating), or UCSD goodreads_books_*.json.gz",
    )
    parser.add_argument("--model", choices=("vader", "transformer", "both"), default="vader")
    parser.add_argument("--min-reviews", type=int, default=10, help="skip books with fewer")
    parser.add_argument("--max-reviews", type=int, default=50, help="use at most this many/book")
    return parser


def _load(path: Path, books: Path | None) -> tuple[list[Review], dict[str, float]]:
    if path.suffix == ".csv":
        return load_csv(path), load_book_ratings(books) if books else {}
    if books is None:
        sys.exit("UCSD reviews need the matching books file: --books goodreads_books_*.json.gz")
    return load_ucsd(path, books), {}


def _scorers(model: str, settings: Settings) -> list[Scorer]:
    scorers: list[Scorer] = []
    if model in ("vader", "both"):
        scorers.append(VaderScorer())
    if model in ("transformer", "both"):
        scorers.append(TransformerScorer(settings.transformer_model))
    return scorers


def _score_missing(store: ReviewStore, ids: list[int], scorer: Scorer) -> None:
    todo = store.unscored(ids, scorer.key)
    done = 0
    for chunk in batched(todo, CHUNK):
        chunk_ids, texts = zip(*chunk, strict=True)
        store.save_scores(scorer.key, zip(chunk_ids, scorer.score(texts), strict=True))
        done += len(chunk)
        print(f"{scorer.name}: scored {done} of {len(todo)} new reviews", file=sys.stderr)


def _print_report(results: list[ModelResult], models_agree: tuple[float, float] | None) -> None:
    for result in results:
        if result.compared:
            print(
                f"{result.model}: {len(result.flagged)} of {len(result.compared)} books diverge"
                f" ({result.share_flagged:.0%}); rating vs sentiment r = {result.correlation:.2f}"
            )
    if models_agree:
        r, same = models_agree
        print(f"VADER vs transformer per review: r = {r:.2f}, same polarity for {same:.0%}")

    rows = list(zip(*(r.books for r in results), strict=True))
    if len(rows) > MAX_ROWS:
        rows.sort(key=lambda row: -max(abs(b.divergence or 0) for b in row))
        print(f"\nThe {MAX_ROWS} most divergent of {len(rows)} books:")
    header = f"{'Book':<32} {'Reviews':>7} {'Stars':>5}" + "".join(
        f" {r.model[:11]:>11} {'diverge':>8}" for r in results
    )
    print("\n" + header + "\n" + "-" * len(header))
    for row in rows[:MAX_ROWS]:
        book = row[0]
        stars = f"{book.rating:.2f}" if book.rating is not None else "-"
        line = f"{book.title[:32]:<32} {book.n_reviews:>7} {stars:>5}"
        for b in row:
            div = "-" if b.divergence is None else f"{b.divergence:+.2f}" + "* "[not b.flagged]
            line += f" {b.sentiment:>11.2f} {div:>8}"
        print(line)
    print("* |divergence| >= 1 SD: flagged")


def _write_csv(results: list[ModelResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["book", "reviews", "rating", "rating_source"]
            + [
                f"{r.model.lower()}_{col}"
                for r in results
                for col in ("sentiment", "divergence", "flagged")
            ]
        )
        for row in zip(*(r.books for r in results), strict=True):
            book = row[0]
            writer.writerow(
                [book.title, book.n_reviews, book.rating, book.rating_source]
                + [
                    v
                    for b in row
                    for v in (
                        round(b.sentiment, 4),
                        None if b.divergence is None else round(b.divergence, 4),
                        b.flagged,
                    )
                ]
            )


if __name__ == "__main__":
    sys.exit(main())
