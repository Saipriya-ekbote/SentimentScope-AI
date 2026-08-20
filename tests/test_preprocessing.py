"""Tests for text preprocessing."""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing import clean_text, preprocess_dataframe


def test_clean_text_lowercases_and_trims() -> None:
    assert clean_text("  Hello WORLD  ") == "hello world"


def test_clean_text_removes_urls() -> None:
    assert clean_text("Check this out https://example.com now") == "check this out now"


def test_clean_text_removes_www_urls() -> None:
    assert clean_text("Visit www.example.com now") == "visit now"


def test_clean_text_handles_missing_values() -> None:
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_clean_text_normalizes_whitespace() -> None:
    assert clean_text("too    many   spaces") == "too many spaces"


def test_preprocess_dataframe_removes_empty_text() -> None:
    df = pd.DataFrame(
        {
            "text": ["Hello", "   ", None],
            "sentiment": ["positive", "neutral", "negative"],
        }
    )
    processed = preprocess_dataframe(df)
    assert len(processed) == 1
    assert processed.iloc[0]["text"] == "hello"


def test_preprocess_dataframe_requires_text_column() -> None:
    with pytest.raises(ValueError, match="Text column not found"):
        preprocess_dataframe(pd.DataFrame({"sentiment": ["positive"]}))
