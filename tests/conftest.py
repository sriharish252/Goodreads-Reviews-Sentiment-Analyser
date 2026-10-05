import gzip
import json
from pathlib import Path

import pytest

DATA = Path(__file__).parents[1] / "data"


@pytest.fixture
def sample_csv() -> Path:
    return DATA / "sample_reviews.csv"


@pytest.fixture
def rated_csv(tmp_path: Path) -> Path:
    path = tmp_path / "reviews.csv"
    path.write_text(
        "book,rating,review\n"
        "Loved,5,An absolutely wonderful and beautiful book.\n"
        "Loved,4,Great characters and a happy ending.\n"
        "Hated,1,Terrible. Boring and awful from start to finish.\n"
        "Hated,2,I disliked it and was sad it was so bad.\n"
        "Mixed,3,It was fine.\n"
        "Mixed,,Some good parts and some bad parts.\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def ucsd_files(tmp_path: Path) -> tuple[Path, Path]:
    """A tiny pair of files in the UCSD Book Graph format."""
    books = [
        {"book_id": "1", "work_id": "10", "title": "Leaves of Grass", "language_code": "eng"},
        {"book_id": "6", "work_id": "10", "title": "Leaves of Grass", "language_code": "en-GB"},
        {"book_id": "2", "work_id": "20", "title": "Poemas", "language_code": "spa"},
        {"book_id": "3", "work_id": "30", "title": "Selected Poems", "language_code": "en-US"},
        {"book_id": "4", "work_id": "40", "title": "Selected Poems", "language_code": ""},
        {"book_id": "5", "work_id": "50", "title": "Selected Poems", "language_code": "eng"},
    ]
    reviews = [
        {"book_id": "1", "rating": 5, "review_text": "Glorious."},
        {"book_id": "1", "rating": 0, "review_text": "No stars given."},
        {"book_id": "1", "rating": 3, "review_text": "  "},
        {"book_id": "2", "rating": 4, "review_text": "Hermoso."},
        {"book_id": "3", "rating": 2, "review_text": "Uneven."},
        {"book_id": "5", "rating": 4, "review_text": "Lovely."},
        {"book_id": "6", "rating": 4, "review_text": "Another edition, same work."},
    ]
    paths = tmp_path / "reviews.json.gz", tmp_path / "books.json.gz"
    for path, records in zip(paths, (reviews, books), strict=True):
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)
    return paths
