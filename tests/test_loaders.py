import pytest

from goodreads_sentiment.loaders import (
    Review,
    limit_per_book,
    load_book_ratings,
    load_csv,
    load_ucsd,
)


def test_sample_has_ten_unrated_reviews_for_each_of_five_books(sample_csv):
    reviews = load_csv(sample_csv)

    assert len(reviews) == 50
    assert {r.book for r in reviews} == {"Life of Pi", "Origin", "Atonement", "Short", "IT"}
    assert all(r.rating is None and r.text for r in reviews)


def test_ratings_are_parsed_and_blank_means_unknown(rated_csv):
    ratings = [r.rating for r in load_csv(rated_csv)]

    assert ratings == [5, 4, 1, 2, 3, None]


def test_out_of_range_rating_is_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("book,rating,review\nA,7,text\n", encoding="utf-8")

    with pytest.raises(ValueError, match="1-5"):
        load_csv(path)


def test_book_ratings(tmp_path):
    path = tmp_path / "books.csv"
    path.write_text("book,avg_rating\nIT,4.25\n", encoding="utf-8")

    assert load_book_ratings(path) == {"IT": 4.25}


def test_limit_per_book_drops_small_books_and_caps_large_ones():
    reviews = [Review("A", f"a{i}") for i in range(5)] + [Review("B", "b0")]

    kept = limit_per_book(reviews, min_reviews=2, max_reviews=3)

    assert [r.text for r in kept] == ["a0", "a1", "a2"]


def test_ucsd_keeps_rated_english_reviews_grouped_by_work(ucsd_files):
    reviews = load_ucsd(*ucsd_files)

    assert reviews == [
        Review("Leaves of Grass", "Glorious.", 5),
        Review("Selected Poems [30]", "Uneven.", 2),
        Review("Selected Poems [50]", "Lovely.", 4),
        Review("Leaves of Grass", "Another edition, same work.", 4),
    ]
