"""Normalization and regex compilation utilities for entity detection."""

from __future__ import annotations

import re

# Whitespace normalization pattern
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_string(text: str | None) -> str:
    """Normalize input text: strip leading/trailing whitespace and collapse internal spaces."""
    if text is None:
        return ""
    return WHITESPACE_PATTERN.sub(" ", str(text).strip())


def build_phrase_regex(phrase: str) -> re.Pattern[str]:
    """Compile a case-insensitive regex pattern enforcing word boundaries.

    Handles:
    - Single words: r'\\bapple\\b'
    - Multi-word phrases: r'\\bbattery\\s+life\\b'
    - Special symbols: safely escaped while keeping leading/trailing word boundaries

    Args:
        phrase: Keyword or entity phrase to match.

    Returns:
        Compiled regular expression pattern.
    """
    clean_phrase = normalize_string(phrase)
    tokens = clean_phrase.split()
    escaped_tokens = [re.escape(token) for token in tokens]

    # Join multi-word tokens with flexible whitespace
    body = r"\s+".join(escaped_tokens)

    # Use word boundary if token starts/ends with alphanumeric characters
    prefix = r"\b" if re.match(r"^\w", clean_phrase) else ""
    suffix = r"\b" if re.search(r"\w$", clean_phrase) else ""

    pattern_str = f"{prefix}{body}{suffix}"
    return re.compile(pattern_str, flags=re.IGNORECASE)


class PatternRegistry:
    """Caches compiled regex patterns to ensure fast, deterministic matching."""

    def __init__(self) -> None:
        self._cache: dict[str, re.Pattern[str]] = {}

    def get(self, phrase: str) -> re.Pattern[str]:
        """Retrieve or compile regex for the given phrase."""
        if phrase not in self._cache:
            self._cache[phrase] = build_phrase_regex(phrase)
        return self._cache[phrase]


# Global pattern cache
GLOBAL_REGISTRY = PatternRegistry()
