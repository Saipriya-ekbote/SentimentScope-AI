"""Load and validate social-media-style CSV datasets.

Backward-compatible facade providing legacy `load_csv()` alongside the new
modular data collection system in `src.data_collection`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import (
    NORMALIZED_COLUMNS,
    PROVENANCE_IMPORTED,
    PROVENANCE_REAL,
    PROVENANCE_SYNTHETIC,
)
from src.data_collection.base import (
    BaseDataLoader,
    ConfigurationError,
    DataIngestionError,
)
from src.data_collection.csv_loader import CSVLoader
from src.data_collection.json_loader import JSONLoader

REQUIRED_COLUMNS = ["id", "platform", "text", "timestamp", "sentiment"]
VALID_SENTIMENTS = {"positive", "negative", "neutral"}


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file and return a legacy 5-column cleaned DataFrame.

    Maintains exact backward compatibility with existing tests and pipelines:
    - Verifies file exists (FileNotFoundError).
    - Checks for required columns ['id', 'platform', 'text', 'timestamp', 'sentiment'].
    - Checks for duplicate IDs.
    - Strips whitespace, converts timestamps to datetime, filters invalid sentiments.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing, duplicate IDs exist, or no usable rows remain.
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


def load_normalized_data(
    source: str | Path,
    provenance: str = PROVENANCE_IMPORTED,
    platform: str | None = None,
    column_mapping: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load data from CSV or JSON into the standard 12-field normalized schema.

    Args:
        source: File path (CSV or JSON).
        provenance: 'Real', 'Imported', or 'Synthetic Demo'.
        platform: Optional platform override.
        column_mapping: Optional custom column mapping.

    Returns:
        pd.DataFrame conforming to NORMALIZED_COLUMNS.
    """
    path = Path(source)
    if path.suffix.lower() == ".json":
        loader: BaseDataLoader = JSONLoader()
    else:
        loader = CSVLoader()

    return loader.load(
        source=path,
        provenance=provenance,
        platform=platform,
        column_mapping=column_mapping,
    )


__all__ = [
    "load_csv",
    "load_normalized_data",
    "REQUIRED_COLUMNS",
    "VALID_SENTIMENTS",
    "CSVLoader",
    "JSONLoader",
    "BaseDataLoader",
    "DataIngestionError",
    "ConfigurationError",
]
