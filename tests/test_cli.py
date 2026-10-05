import csv

import pytest

from goodreads_sentiment import scorers
from goodreads_sentiment.cli import main

from .test_scorers import fake_classifier


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setenv("GRSA_DB_PATH", str(tmp_path / "reviews.db"))
    monkeypatch.setenv("GRSA_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(scorers, "_load_pipeline", lambda model: fake_classifier)
    return tmp_path


@pytest.fixture
def books_csv(tmp_path):
    path = tmp_path / "books.csv"
    path.write_text(
        "book,avg_rating\nLife of Pi,3.9\nOrigin,3.8\nAtonement,3.9\nShort,4.1\nIT,4.2\n",
        encoding="utf-8",
    )
    return path


def test_sample_with_both_models_writes_table_csv_and_chart(sample_csv, books_csv, out, capsys):
    assert main(["--input", str(sample_csv), "--books", str(books_csv), "--model", "both"]) == 0

    printed = capsys.readouterr().out
    assert "VADER:" in printed and "Transformer:" in printed
    assert "VADER vs transformer per review" in printed
    with (out / "summary.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    assert {r["rating_source"] for r in rows} == {"published"}
    assert "transformer_divergence" in rows[0]
    assert (out / "divergence.png").stat().st_size > 10_000


def test_rerun_reuses_cached_scores(sample_csv, books_csv, out, capsys):
    main(["--input", str(sample_csv), "--books", str(books_csv)])
    assert "scored 50 of 50" in capsys.readouterr().err

    main(["--input", str(sample_csv), "--books", str(books_csv)])
    assert "scored" not in capsys.readouterr().err


def test_reviewer_ratings_are_used_without_a_books_file(rated_csv, out, capsys):
    main(["--input", str(rated_csv), "--min-reviews", "2"])

    assert "VADER: " in capsys.readouterr().out
    with (out / "summary.csv").open(encoding="utf-8") as f:
        assert {r["rating_source"] for r in csv.DictReader(f)} == {"reviews"}


def test_without_ratings_there_is_no_chart(sample_csv, out, capsys):
    main(["--input", str(sample_csv)])

    assert "No chart" in capsys.readouterr().out
    assert not (out / "divergence.png").exists()


def test_min_reviews_filters_every_book_out(sample_csv, out):
    with pytest.raises(SystemExit, match="at least 11 reviews"):
        main(["--input", str(sample_csv), "--min-reviews", "11"])
