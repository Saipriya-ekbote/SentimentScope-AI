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
