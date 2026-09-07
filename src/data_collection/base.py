"""Base loader and schema definitions for SentimentScope AI data collection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path

import pandas as pd

from config import (
    NORMALIZED_COLUMNS,
    PROVENANCE_IMPORTED,
    PROVENANCE_REAL,
    PROVENANCE_SYNTHETIC,
    SUPPORTED_PLATFORMS,
)


class DataIngestionError(Exception):
    """Base exception for data ingestion errors."""


class ConfigurationError(DataIngestionError):
    """Raised when an ingestion source requires missing credentials or configuration."""


# Default source column name mapping to normalized column names
DEFAULT_COLUMN_MAPPINGS: dict[str, str] = {
    "id": "post_id",
    "tweet_id": "post_id",
    "comment_id": "post_id",
    "message_id": "post_id",
    "content": "text",
    "body": "text",
    "message": "text",
    "tweet": "text",
    "created_at": "timestamp",
    "tweet_created": "timestamp",
    "date": "timestamp",
    "datetime": "timestamp",
    "user": "author",
    "username": "author",
    "user_name": "author",
    "screen_name": "author",
    "author_name": "author",
    "airline_sentiment": "sentiment",
    "score": "sentiment_score",
}


class BaseDataLoader(ABC):
    """Abstract base class for all SentimentScope AI data loaders.
    
    Subclasses must implement the load method to ingest data from specific
    sources (files, streams, APIs) and return a DataFrame conforming to
    NORMALIZED_COLUMNS.
    """

    def __init__(
        self,
        default_platform: str = "Other",
        default_provenance: str = PROVENANCE_IMPORTED,
    ) -> None:
        self.default_platform = default_platform
        self.default_provenance = default_provenance

    @abstractmethod
    def load(self, source: Any, **kwargs: Any) -> pd.DataFrame:
        """Ingest data from source and return a normalized pandas DataFrame.
        
        Args:
            source: Source identifier (file path, URI, API parameters, etc.).
            **kwargs: Additional loader-specific options.

        Returns:
            pd.DataFrame conforming to NORMALIZED_COLUMNS.
        """

    def normalize_dataframe(
        self,
        df: pd.DataFrame,
        platform: str | None = None,
        provenance: str | None = None,
        column_mapping: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Convert a raw or partially shaped DataFrame into the standard 12-field schema.

        Args:
            df: Input raw DataFrame.
            platform: Platform name override (e.g. 'Twitter', 'Reddit').
            provenance: Provenance tag ('Real', 'Imported', 'Synthetic Demo').
            column_mapping: Optional additional mapping of input columns to normalized names.

        Returns:
            Normalized pandas DataFrame with columns matching NORMALIZED_COLUMNS.

        Raises:
            DataIngestionError: If df is empty or lacks minimum viable fields.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

        if df.empty:
            raise DataIngestionError("Cannot normalize an empty DataFrame.")

        working = df.copy()

        # Combine default mapping with any user-supplied mapping
        mappings = dict(DEFAULT_COLUMN_MAPPINGS)
        if column_mapping:
            mappings.update(column_mapping)

        # Rename known columns
        rename_dict = {
            orig: target
            for orig, target in mappings.items()
            if orig in working.columns and target not in working.columns
        }
        if rename_dict:
            working = working.rename(columns=rename_dict)

        # Minimum required fields to construct a meaningful social post record
        if "text" not in working.columns:
            raise DataIngestionError(
                "Input data lacks a 'text' (or equivalent 'content'/'body'/'message') column."
            )

        # If post_id is missing, auto-generate sequential IDs
        if "post_id" not in working.columns:
            working["post_id"] = [f"auto_{i + 1}" for i in range(len(working))]
        else:
            working["post_id"] = working["post_id"].astype(str).str.strip()

        # Platform assignment
        chosen_platform = platform or self.default_platform
        if "platform" not in working.columns or working["platform"].isna().all():
            working["platform"] = chosen_platform
        else:
            working["platform"] = (
                working["platform"]
                .fillna(chosen_platform)
                .astype(str)
                .str.strip()
                .replace("", chosen_platform)
            )

        # Text cleaning (basic strip)
        working["text"] = working["text"].astype(str).str.strip()

        # Timestamp normalization
        if "timestamp" not in working.columns:
            raise DataIngestionError(
                "Input data lacks a 'timestamp' (or equivalent 'created_at'/'date') column."
            )

        working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")

        # Provenance assignment
        chosen_provenance = provenance or self.default_provenance
        if chosen_provenance not in {PROVENANCE_REAL, PROVENANCE_IMPORTED, PROVENANCE_SYNTHETIC}:
            # Normalize to one of the valid constants if recognizable
            lowered = chosen_provenance.lower()
            if "synthetic" in lowered or "demo" in lowered:
                chosen_provenance = PROVENANCE_SYNTHETIC
            elif "real" in lowered:
                chosen_provenance = PROVENANCE_REAL
            else:
                chosen_provenance = PROVENANCE_IMPORTED
        working["provenance"] = chosen_provenance

        # Author column
        if "author" not in working.columns:
            working["author"] = None
        else:
            working["author"] = working["author"].apply(
                lambda x: str(x).strip() if pd.notna(x) and str(x).strip() != "" else None
            )

        # Brand, Product, Topic placeholders
        for field in ["brand", "product", "topic"]:
            if field not in working.columns:
                working[field] = None
            else:
                working[field] = working[field].apply(
                    lambda x: str(x).strip() if pd.notna(x) and str(x).strip() != "" else None
                )

        # Sentiment & Sentiment Score
        if "sentiment" not in working.columns:
            working["sentiment"] = None
        else:
            working["sentiment"] = working["sentiment"].apply(
                lambda x: str(x).strip().lower() if pd.notna(x) and str(x).strip() != "" else None
            )

        if "sentiment_score" not in working.columns:
            working["sentiment_score"] = None
        else:
            working["sentiment_score"] = pd.to_numeric(working["sentiment_score"], errors="coerce")

        # Engagement metrics
        if "engagement_metrics" not in working.columns:
            working["engagement_metrics"] = None

        # Filter invalid rows: non-empty text, parseable timestamp, non-empty post_id
        valid_mask = (
            working["text"].ne("")
            & working["timestamp"].notna()
            & working["post_id"].ne("")
            & working["post_id"].ne("nan")
        )
        working = working.loc[valid_mask].reset_index(drop=True)

        if working.empty:
            raise DataIngestionError(
                "No valid rows remain after schema normalization (all text or timestamps were invalid)."
            )

        # Deduplicate on post_id if duplicates exist
        if working["post_id"].duplicated().any():
            working = working.drop_duplicates(subset=["post_id"], keep="first").reset_index(drop=True)

        # Reindex strictly to standard 12 columns
        return working[NORMALIZED_COLUMNS]
