"""Text preprocessing utilities for sentiment analysis."""

from __future__ import annotations

import html
import re

import pandas as pd

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"(?<!\w)@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{2,}")
MULTISPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """Clean a single text value for sentiment analysis.

    Handles:
    - HTML entity decoding (e.g. &amp; -> &)
    - URL removal (http/https and www links)
    - User handle removal (@username)
    - Hashtag symbol stripping (#topic -> topic)
    - Repeated character compression (e.g. sooooo -> soo)
    - Lowercasing and whitespace normalization
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    # Decode HTML entities first (e.g., &amp; -> &, &lt; -> <)
    cleaned = html.unescape(str(text))

    # Strip URLs
    cleaned = URL_PATTERN.sub(" ", cleaned)

    # Remove user handles (@username)
    cleaned = MENTION_PATTERN.sub(" ", cleaned)

    # Strip hashtag symbol while preserving the sentiment-bearing word
    cleaned = HASHTAG_PATTERN.sub(r"\1", cleaned)

    # Compress 3+ consecutive identical characters down to 2 (e.g., sooooo -> soo)
    cleaned = REPEATED_CHAR_PATTERN.sub(r"\1\1", cleaned)

    # Lowercase and normalize whitespace
    cleaned = cleaned.lower()
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
