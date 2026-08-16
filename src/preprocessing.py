"""Text preprocessing utilities for sentiment analysis."""

from __future__ import annotations

import re

import pandas as pd

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MULTISPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """Clean a single text value for sentiment analysis."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    cleaned = str(text).strip().lower()
    cleaned = URL_PATTERN.sub(" ", cleaned)
    cleaned = MULTISPACE_PATTERN.sub(" ", cleaned).strip()
    return cleaned


def preprocess_dataframe(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """Return a copy of the DataFrame with a cleaned text column."""
    if text_column not in df.columns:
        raise ValueError(f"Text column not found: {text_column}")

    processed = df.copy()
    processed[text_column] = processed[text_column].apply(clean_text)
    processed = processed.loc[processed[text_column].ne("")].reset_index(drop=True)

    if processed.empty:
        raise ValueError("No rows remain after text preprocessing.")

    return processed
