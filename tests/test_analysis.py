import pytest

from goodreads_sentiment.analysis import agreement, compare
from goodreads_sentiment.db import BookStats


def book(title, rating, sentiment, source="reviews"):
    return BookStats(title, 10, rating, source, sentiment)


def test_books_where_sentiment_tracks_rating_are_not_flagged():
    result = compare("m", [book("A", 3.0, 0.0), book("B", 4.0, 0.4), book("C", 5.0, 0.8)])

    assert [b.divergence for b in result.books] == pytest.approx([0, 0, 0])
    assert result.flagged == []
    assert result.share_flagged == 0
    assert result.correlation == pytest.approx(1)


def test_divergence_is_sentiment_z_minus_rating_z():
    stats = [
        book("Low stars, warm reviews", 3.0, 0.8),
        book("Mid", 4.0, 0.4),
        book("Top", 5.0, 0.0),
    ]

    result = compare("m", stats)

    by_title = {b.title: b for b in result.books}
    z = 1.5**0.5  # z-score of the extremes of three evenly spaced values
    assert by_title["Low stars, warm reviews"].divergence == pytest.approx(2 * z)
    assert by_title["Top"].divergence == pytest.approx(-2 * z)
    assert by_title["Mid"].divergence == pytest.approx(0)
    assert [b.title for b in result.flagged] == ["Low stars, warm reviews", "Top"]
    assert result.share_flagged == pytest.approx(2 / 3)


def test_unrated_books_are_reported_but_not_compared():
    stats = [
        book("A", 3.0, 0.1),
        book("B", 4.0, 0.5),
        book("C", 5.0, 0.6),
        book("D", None, 0.9, ""),
    ]

    result = compare("m", stats)

    assert len(result.books) == 4
    assert [b.title for b in result.compared] == ["A", "B", "C"]


def test_too_few_rated_books_means_no_divergence():
    result = compare("m", [book("A", 3.0, 0.1), book("B", 4.0, 0.5)])

    assert result.compared == []
    assert result.share_flagged is None
    assert result.correlation is None


def test_agreement_between_models():
    r, same = agreement({1: 0.5, 2: -0.5, 3: 0.9}, {1: 0.4, 2: 0.2, 3: 0.8})

    assert same == pytest.approx(2 / 3)
    assert 0 < r <= 1
