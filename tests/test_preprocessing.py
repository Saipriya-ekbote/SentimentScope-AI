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


def test_clean_text_decodes_html_entities() -> None:
    assert clean_text("Service was fast &amp; friendly &lt;3") == "service was fast & friendly <3"
    assert clean_text("It&#39;s a &quot;must have&quot; item!") == "it's a \"must have\" item!"


def test_clean_text_removes_user_handles() -> None:
    assert clean_text("@username loved this!") == "loved this!"
    assert clean_text("Hey @support_team please help @user123") == "hey please help"


def test_clean_text_preserves_email_addresses() -> None:
    assert clean_text("Contact support@example.com for help") == "contact support@example.com for help"


def test_clean_text_normalizes_hashtags() -> None:
    assert clean_text("#AmazingProduct #LoveIt") == "amazingproduct loveit"
    assert clean_text("This is #awesome!") == "this is awesome!"


def test_clean_text_compresses_repeated_characters() -> None:
    assert clean_text("sooooo good") == "soo good"
    assert clean_text("goooood morning") == "good morning"
    # Preserves valid double-letter English words
    assert clean_text("happy coffee week") == "happy coffee week"
    assert clean_text("noooooooo wayyyyy!!!") == "noo wayy!!"


def test_clean_text_combined_social_media_features() -> None:
    raw = "@customer_care This service is sooooo bad &amp; broken! Check www.status.com #AngryCustomer"
    expected = "this service is soo bad & broken! check angrycustomer"
    assert clean_text(raw) == expected


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
