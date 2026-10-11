import sqlite3

import pytest

from goodreads_sentiment.db import BookStats, ReviewStore
from goodreads_sentiment.loaders import Review


@pytest.fixture
def store():
    with ReviewStore(":memory:") as store:
        yield store


def test_adding_the_same_reviews_twice_reuses_rows(store):
    reviews = [Review("A", "good", 5), Review("A", "bad", 1)]

    assert store.add_reviews(reviews) == store.add_reviews(reviews) == [1, 2]


def test_unscored_skips_reviews_already_scored_by_that_model(store):
    ids = store.add_reviews([Review("A", "one"), Review("A", "two")])
    store.save_scores("vader", [(ids[0], 0.5)])

    assert store.unscored(ids, "vader") == [(ids[1], "two")]
    assert len(store.unscored(ids, "other")) == 2


def test_book_stats_prefer_reviewer_ratings_over_published_average(store):
    ids = store.add_reviews(
        [Review("Rated", "x", 4), Review("Rated", "y", 2), Review("Published", "z")]
    )
    store.set_book_ratings({"Rated": 1.0, "Published": 3.9})
    store.save_scores("vader", zip(ids, [0.5, -0.1, 0.8], strict=True))

    stats = store.book_stats(ids, "vader")

    assert stats == [
        BookStats("Published", 1, 3.9, "published", 0.8),
        BookStats("Rated", 2, 3.0, "reviews", pytest.approx(0.2)),
    ]


def test_book_stats_only_cover_the_selected_reviews(store):
    first = store.add_reviews([Review("A", "old")])
    second = store.add_reviews([Review("B", "new")])
    store.save_scores("vader", [(first[0], 0.1), (second[0], 0.2)])

    assert [s.title for s in store.book_stats(second, "vader")] == ["B"]


def test_scores_outside_minus_one_to_one_are_rejected(store):
    ids = store.add_reviews([Review("A", "x")])

    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        store.save_scores("vader", [(ids[0], 1.5)])
