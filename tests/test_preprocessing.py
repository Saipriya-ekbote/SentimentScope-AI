"""Tests for text preprocessing."""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing import clean_text, preprocess_dataframe, split_hashtag


# --- Legacy Tests (Preserved 100%) ---

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


# --- Phase 3 Enhanced Preprocessing Tests ---

def test_clean_text_converts_positive_emojis() -> None:
    result = clean_text("Loving this update 😍🔥❤️")
    assert "loving this update" in result
    assert "positive" in result
    # Emojis should be replaced with text tokens, not raw unicode symbols
    assert "😍" not in result
    assert "🔥" not in result
    assert "❤️" not in result


def test_clean_text_converts_negative_emojis() -> None:
    result = clean_text("Flight cancelled again 😡😭👎")
    assert "flight cancelled again" in result
    assert "negative" in result
    assert "😡" not in result
    assert "😭" not in result
    assert "👎" not in result


def test_clean_text_handles_unmapped_emojis_safely() -> None:
    # 🚀 and ☕ are not in the standard sentiment map
    result = clean_text("Launching our new cafe 🚀☕")
    assert "launching our new cafe" in result
    assert "🚀" not in result
    assert "☕" not in result


def test_split_hashtag_utility() -> None:
    assert split_hashtag("#AmazingProduct") == "amazing product"
    assert split_hashtag("#WorstServiceEver") == "worst service ever"
    assert split_hashtag("#simple") == "simple"
    assert split_hashtag("") == ""


def test_clean_text_with_split_hashtags_option() -> None:
    result = clean_text("Check out this #AmazingProduct!", split_hashtags=True)
    assert result == "check out this amazing product!"


def test_sentiment_preservation_positive() -> None:
    raw = "OMG!!! This product is AMAZING 😍🔥 #BestProduct"
    cleaned = clean_text(raw)
    # Validates that key positive sentiment indicators are preserved
    assert "omg!!" in cleaned
    assert "amazing" in cleaned
    assert "positive" in cleaned
    assert "bestproduct" in cleaned


def test_sentiment_preservation_negative() -> None:
    raw = "Terrible service 😡😡 #WorstService"
    cleaned = clean_text(raw)
    # Validates that key negative sentiment indicators are preserved
    assert "terrible service" in cleaned
    assert "negative" in cleaned
    assert "worstservice" in cleaned


def test_clean_text_excessive_punctuation_normalization() -> None:
    assert clean_text("AMAZING!!!!") == "amazing!!"
    assert clean_text("Really????") == "really??"
    assert clean_text("So bad.....") == "so bad.."


def test_clean_text_non_string_inputs() -> None:
    assert clean_text(12345) == "12345"
    assert clean_text(3.14) == "3.14"
    assert clean_text(True) == "true"


def test_clean_text_empty_and_whitespace_only() -> None:
    assert clean_text("") == ""
    assert clean_text("    ") == ""
    assert clean_text("\t\n") == ""


def test_preprocess_dataframe_target_column() -> None:
    df = pd.DataFrame(
        {
            "raw_text": ["Great food 😍", "Terrible 😡"],
            "sentiment": ["positive", "negative"],
        }
    )
    processed = preprocess_dataframe(df, text_column="raw_text", target_column="clean_text")
    assert "clean_text" in processed.columns
    assert "raw_text" in processed.columns
    assert "positive" in processed.iloc[0]["clean_text"]
    assert "negative" in processed.iloc[1]["clean_text"]
