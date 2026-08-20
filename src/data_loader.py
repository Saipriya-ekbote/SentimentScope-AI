"""Load and validate social-media-style CSV datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["id", "platform", "text", "timestamp", "sentiment"]
VALID_SENTIMENTS = {"positive", "negative", "neutral"}


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file and return a cleaned DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing or no usable rows remain.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset is empty: {file_path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Malformed CSV file: {file_path}") from exc

    if df.empty:
        raise ValueError(f"Dataset contains no rows: {file_path}")

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_columns)}"
        )
    if df["id"].duplicated().any():
        raise ValueError("Duplicate IDs found in dataset.")

    df = df[REQUIRED_COLUMNS].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["platform"] = df["platform"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    invalid_mask = (
        df["text"].eq("")
        | df["platform"].eq("")
        | df["timestamp"].isna()
        | ~df["sentiment"].isin(VALID_SENTIMENTS)
    )
    df = df.loc[~invalid_mask].reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No usable rows remain after validation. "
            "Check text, timestamps, and sentiment labels."
        )

    return df
