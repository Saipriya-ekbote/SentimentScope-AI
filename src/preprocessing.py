"""Text preprocessing utilities for sentiment analysis.

Backward-compatibility facade delegating to `src.preprocessing.cleaner`.
"""

from __future__ import annotations

from src.preprocessing.cleaner import (
    CAMEL_CASE_PATTERN,
    DEFAULT_EMOJI_SENTIMENT_MAP,
    EMOJI_PATTERN,
    HASHTAG_PATTERN,
    MENTION_PATTERN,
    MULTISPACE_PATTERN,
    REPEATED_CHAR_PATTERN,
    URL_PATTERN,
    clean_text,
    preprocess_dataframe,
    replace_emojis,
    split_hashtag,
)

__all__ = [
    "clean_text",
    "preprocess_dataframe",
    "split_hashtag",
    "replace_emojis",
    "DEFAULT_EMOJI_SENTIMENT_MAP",
    "URL_PATTERN",
    "MENTION_PATTERN",
    "HASHTAG_PATTERN",
    "REPEATED_CHAR_PATTERN",
    "MULTISPACE_PATTERN",
    "CAMEL_CASE_PATTERN",
    "EMOJI_PATTERN",
]
