"""Text cleaning and normalization pipeline for social media text."""

from __future__ import annotations

import html
import re
from typing import Any

import pandas as pd

# Core regex patterns for social media text cleaning
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"(?<!\w)@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{2,}")
MULTISPACE_PATTERN = re.compile(r"\s+")
CAMEL_CASE_PATTERN = re.compile(r"([a-z])([A-Z])")
EMOJI_PATTERN = re.compile(
    r"[\U00010000-\U0010ffff]|[\u2600-\u27bf]|[\ufe00-\ufe0f]|[\u2300-\u23ff]|[\u2b50-\u2b55]"
)

# Common sentiment-bearing emoji mapping (includes multi-byte / variation selector variants)
DEFAULT_EMOJI_SENTIMENT_MAP: dict[str, str] = {
    # High positive
    "😍": "positive",
    "🥰": "positive",
    "❤️": "positive",
    "\u2764\ufe0f": "positive",
    "\u2764": "positive",
    "💖": "positive",
    "💕": "positive",
    "😊": "positive",
    "😃": "positive",
    "😄": "positive",
    "😁": "positive",
    "😀": "positive",
    "🎉": "positive",
    "🔥": "positive",
    "👍": "positive",
    "✨": "positive",
    "👏": "positive",
    "🙌": "positive",
    "💯": "positive",
    "🤩": "positive",
    "🥳": "positive",
    "👌": "positive",
    # High negative
    "😡": "negative",
    "😠": "negative",
    "🤬": "negative",
    "😢": "negative",
    "😭": "negative",
    "😞": "negative",
    "😔": "negative",
    "😒": "negative",
    "👎": "negative",
    "💔": "negative",
    "🤮": "negative",
    "🤢": "negative",
    "💩": "negative",
    "😤": "negative",
    "😩": "negative",
    "😫": "negative",
    "🙄": "negative",
}


def split_hashtag(tag: str) -> str:
    """Split a hashtag word into separate words based on camelCase/PascalCase.

    Examples:
        split_hashtag("#AmazingProduct") -> "amazing product"
        split_hashtag("#WorstServiceEver") -> "worst service ever"
        split_hashtag("simple") -> "simple"
    """
    cleaned = tag.lstrip("#").strip()
    if not cleaned:
        return ""
    split = CAMEL_CASE_PATTERN.sub(r"\1 \2", cleaned)
    return split.lower()


def replace_emojis(
    text: str,
    emoji_map: dict[str, str] | None = None,
    strip_unmapped: bool = True,
) -> str:
    """Convert common sentiment emojis into textual tokens and optionally strip unmapped emojis.

    Args:
        text: Input string.
        emoji_map: Dictionary mapping emoji characters to text descriptions.
        strip_unmapped: If True, removes remaining unmapped unicode emojis.

    Returns:
        String with emojis translated or stripped.
    """
    mapping = emoji_map if emoji_map is not None else DEFAULT_EMOJI_SENTIMENT_MAP
    if not text:
        return ""

    # Sort keys by length descending to match composite emojis with variation selectors first
    sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(k) for k in sorted_keys))
    translated = pattern.sub(lambda m: f" {mapping[m.group(0)]} ", text)

    if strip_unmapped:
        translated = EMOJI_PATTERN.sub(" ", translated)

    return translated


def clean_text(
    text: Any,
    convert_emojis: bool = True,
    split_hashtags: bool = False,
    strip_unmapped_emojis: bool = True,
    emoji_map: dict[str, str] | None = None,
) -> str:
    """Clean a single social media text value for sentiment analysis.

    Deterministic transformation pipeline:
    1. Null/invalid check (None, NaN, empty -> return "")
    2. HTML entity unescaping (e.g. &amp; -> &, &lt;3 -> <3)
    3. URL removal (http://, https://, www.)
    4. Mention removal (@username stripped, emails preserved via lookbehind)
    5. Hashtag processing (#topic preserved as topic, optional camelCase split)
    6. Emoji translation (sentiment-bearing emojis mapped to textual tokens)
    7. Repeated character compression (3+ identical consecutive chars -> 2 chars)
    8. Lowercasing and whitespace normalization

    Args:
        text: Raw text string or object.
        convert_emojis: If True, translates sentiment-bearing emojis to sentiment tokens.
        split_hashtags: If True, splits PascalCase/camelCase hashtags into separate words.
        strip_unmapped_emojis: If True, removes non-sentiment unmapped unicode emojis.
        emoji_map: Optional custom dictionary for emoji replacement.

    Returns:
        Cleaned, normalized string.
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    raw_str = str(text).strip()
    if not raw_str:
        return ""

    # Step 1: Decode HTML entities
    cleaned = html.unescape(raw_str)

    # Step 2: Strip URLs
    cleaned = URL_PATTERN.sub(" ", cleaned)

    # Step 3: Remove user handles (@username) while preserving email addresses
    cleaned = MENTION_PATTERN.sub(" ", cleaned)

    # Step 4: Hashtag processing
    if split_hashtags:
        cleaned = HASHTAG_PATTERN.sub(lambda m: split_hashtag(m.group(1)), cleaned)
    else:
        cleaned = HASHTAG_PATTERN.sub(r"\1", cleaned)

    # Step 5: Emoji handling
    if convert_emojis:
        cleaned = replace_emojis(
            cleaned,
            emoji_map=emoji_map,
            strip_unmapped=strip_unmapped_emojis,
        )

    # Step 6: Character elongation compression (3+ consecutive identical chars -> 2)
    cleaned = REPEATED_CHAR_PATTERN.sub(r"\1\1", cleaned)

    # Step 7: Lowercasing and whitespace normalization
    cleaned = cleaned.lower()
    cleaned = MULTISPACE_PATTERN.sub(" ", cleaned).strip()

    return cleaned


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    target_column: str | None = None,
    convert_emojis: bool = True,
    split_hashtags: bool = False,
) -> pd.DataFrame:
    """Preprocess a pandas DataFrame by cleaning its text column.

    Args:
        df: Input pandas DataFrame.
        text_column: Name of the source text column to clean (default 'text').
        target_column: Destination column name. If None, overwrites text_column.
        convert_emojis: Whether to convert emojis to text tokens.
        split_hashtags: Whether to split camelCase hashtags.

    Returns:
        pd.DataFrame with cleaned text and non-empty rows.

    Raises:
        ValueError: If text_column is not in df or no usable rows remain.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")

    if text_column not in df.columns:
        raise ValueError(f"Text column not found: {text_column}")

    dest_col = target_column or text_column
    processed = df.copy()

    processed[dest_col] = processed[text_column].apply(
        lambda t: clean_text(
            t,
            convert_emojis=convert_emojis,
            split_hashtags=split_hashtags,
        )
    )

    processed = processed.loc[processed[dest_col].ne("")].reset_index(drop=True)

    if processed.empty:
        raise ValueError("No rows remain after text preprocessing.")

    return processed
