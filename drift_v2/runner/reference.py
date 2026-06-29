from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from drift_v2.runner.config import RunnerConfig
from drift_v2.runner.metrics import compute_label_token_scores
from src.config import NUMERICAL_COLUMNS, RAW_TEXT_COLUMN, TARGET_COLUMN, TEXT_COLUMN
from src.config_loader import load_config


REFERENCE_CSV_NAME = "reference_train.csv"
REFERENCE_METADATA_NAME = "metadata.json"
REFERENCE_STATS_NAME = "reference_stats.json"


def ensure_reference_snapshot(config: RunnerConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    config.reference_dir.mkdir(parents=True, exist_ok=True)
    metadata = build_reference_metadata(config)

    metadata_path = config.reference_dir / REFERENCE_METADATA_NAME
    csv_path = config.reference_dir / REFERENCE_CSV_NAME
    stats_path = config.reference_dir / REFERENCE_STATS_NAME

    if metadata_path.exists() and csv_path.exists() and stats_path.exists():
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing_metadata == metadata:
            df = pd.read_csv(csv_path)
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            return df, stats

    df = _build_reference_train_df(config)
    stats = build_reference_stats(df, config)
    df.to_csv(csv_path, index=False)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return df, stats


def build_reference_metadata(config: RunnerConfig) -> dict[str, Any]:
    training_config = load_config(config.training_config_path)
    return {
        "data_path": str(config.data_path),
        "training_config_path": str(config.training_config_path),
        "dataset_hash": _hash_file(config.data_path),
        "test_size": training_config.training.test_size,
        "random_state": training_config.training.random_state,
        "columns": [
            RAW_TEXT_COLUMN,
            TEXT_COLUMN,
            TARGET_COLUMN,
            *NUMERICAL_COLUMNS,
        ],
    }


def build_reference_stats(df: pd.DataFrame, config: RunnerConfig) -> dict[str, Any]:
    token_counter = Counter()
    for value in df[TEXT_COLUMN]:
        token_counter.update(_split_tokens(value))

    class_distribution = (
        df[TARGET_COLUMN].value_counts(normalize=True).sort_index().to_dict()
    )
    label_scores = compute_label_token_scores(
        df,
        text_column=TEXT_COLUMN,
        label_column=TARGET_COLUMN,
        top_k_per_label=config.top_k_label_tokens,
    )
    top_tokens = token_counter.most_common(config.top_k_tokens)

    return {
        "class_distribution": class_distribution,
        "top_tokens": [{"token": token, "count": count} for token, count in top_tokens],
        "label_token_scores": label_scores,
    }


def _build_reference_train_df(config: RunnerConfig) -> pd.DataFrame:
    training_config = load_config(config.training_config_path)
    df = pd.read_csv(config.data_path, index_col=0)
    required = [RAW_TEXT_COLUMN, TEXT_COLUMN, TARGET_COLUMN, *NUMERICAL_COLUMNS]
    df = df.dropna(subset=required).reset_index(drop=True)

    train_df, _ = train_test_split(
        df,
        test_size=training_config.training.test_size,
        random_state=training_config.training.random_state,
    )
    return train_df.reset_index(drop=True)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_tokens(value: str) -> list[str]:
    return [token for token in str(value).split() if token]
