"""Tests for CSV data loading and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import load_csv


def test_load_valid_csv(sample_csv_path: Path) -> None:
    df = load_csv(sample_csv_path)
    assert len(df) == 3
    assert list(df.columns) == ["id", "platform", "text", "timestamp", "sentiment"]
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_missing_required_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "missing_column.csv"
    pd.DataFrame(
        {
            "id": [1],
            "platform": ["Twitter"],
            "text": ["hello"],
            "timestamp": ["2026-08-01 10:00:00"],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_csv(csv_path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "does_not_exist.csv")


def test_invalid_rows_are_removed(tmp_path: Path) -> None:
    csv_path = tmp_path / "invalid_rows.csv"
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "platform": ["Twitter", "Reddit", ""],
            "text": ["good", "", "bad"],
            "timestamp": ["2026-08-01 10:00:00", "invalid-date", "2026-08-01 11:00:00"],
            "sentiment": ["positive", "negative", "unknown"],
        }
    ).to_csv(csv_path, index=False)

    df = load_csv(csv_path)
    assert len(df) == 1
    assert df.iloc[0]["sentiment"] == "positive"


def test_prepare_realistic_data_pipeline(tmp_path: Path) -> None:
    from scripts.prepare_realistic_data import prepare_realistic_data

    raw_csv = tmp_path / "mock_tweets.csv"
    output_csv = tmp_path / "mock_output.csv"

    pd.DataFrame(
        {
            "tweet_id": [101, 102, 101, 103],  # 101 is duplicate
            "text": ["Great flight!", "Bad delay", "Great flight duplicate", "Neutral note"],
            "tweet_created": [
                "2015-02-20 10:00:00 -0800",
                "2015-02-20 11:00:00 -0800",
                "2015-02-20 10:00:00 -0800",
                "2015-02-20 12:00:00 -0800",
            ],
            "airline_sentiment": ["positive", "negative", "positive", "neutral"],
            "airline": ["Delta", "United", "Delta", "American"],
        }
    ).to_csv(raw_csv, index=False)

    df_out = prepare_realistic_data(raw_csv, output_csv)
    assert len(df_out) == 3
    assert list(df_out.columns) == ["id", "platform", "text", "timestamp", "sentiment"]
    assert (df_out["platform"] == "Twitter").all()
    assert df_out["id"].tolist() == ["101", "102", "103"]
    assert df_out["sentiment"].tolist() == ["positive", "negative", "neutral"]

    # Verify load_csv can load it
    loaded_df = load_csv(output_csv)
    assert len(loaded_df) == 3


def test_load_realistic_dataset_file() -> None:
    project_root = Path(__file__).resolve().parents[1]
    realistic_path = project_root / "data" / "realistic_social_data.csv"
    if realistic_path.exists():
        df = load_csv(realistic_path)
        assert len(df) == 14485
        assert list(df.columns) == ["id", "platform", "text", "timestamp", "sentiment"]
        assert (df["platform"] == "Twitter").all()
        assert set(df["sentiment"].unique()) == {"positive", "negative", "neutral"}
        assert df["id"].duplicated().sum() == 0
