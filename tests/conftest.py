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
