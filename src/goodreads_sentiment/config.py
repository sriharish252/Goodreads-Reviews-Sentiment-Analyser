"""Settings read from environment variables (see .env.example)."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TRANSFORMER = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


@dataclass(frozen=True)
class Settings:
    db_path: Path
    output_dir: Path
    transformer_model: str


def load_settings(env: Mapping[str, str] = os.environ) -> Settings:
    return Settings(
        db_path=Path(env.get("GRSA_DB_PATH", "out/reviews.db")),
        output_dir=Path(env.get("GRSA_OUTPUT_DIR", "out")),
        transformer_model=env.get("GRSA_TRANSFORMER_MODEL", DEFAULT_TRANSFORMER),
    )
