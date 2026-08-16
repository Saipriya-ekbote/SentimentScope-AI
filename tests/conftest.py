"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_csv_path(tmp_path: Path) -> Path:
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "platform": ["Twitter", "Reddit", "YouTube"],
            "text": [
                "I love this product",
                "This is terrible",
                "The package arrived today",
            ],
            "timestamp": [
                "2026-08-01 10:00:00",
                "2026-08-01 11:00:00",
                "2026-08-01 12:00:00",
            ],
            "sentiment": ["positive", "negative", "neutral"],
        }
    ).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def training_df() -> pd.DataFrame:
    rows = []
    positive_texts = [
        "I love this",
        "Fantastic service",
        "Great product",
        "Highly recommend",
        "Excellent quality",
    ]
    negative_texts = [
        "This is awful",
        "Terrible experience",
        "I hate this",
        "Worst app ever",
        "Very disappointing",
    ]
    neutral_texts = [
        "The package arrived",
        "I downloaded the app",
        "There is an update",
        "The page loaded",
        "I created an account",
    ]

    for index, text in enumerate(positive_texts * 4):
        rows.append(
            {
                "text": f"{text} {index}",
                "sentiment": "positive",
                "timestamp": f"2026-08-01 {index % 24:02d}:00:00",
                "platform": "Twitter",
            }
        )
    for index, text in enumerate(negative_texts * 4):
        rows.append(
            {
                "text": f"{text} {index}",
                "sentiment": "negative",
                "timestamp": f"2026-08-02 {index % 24:02d}:00:00",
                "platform": "Reddit",
            }
        )
    for index, text in enumerate(neutral_texts * 4):
        rows.append(
            {
                "text": f"{text} {index}",
                "sentiment": "neutral",
                "timestamp": f"2026-08-03 {index % 24:02d}:00:00",
                "platform": "YouTube",
            }
        )

    return pd.DataFrame(rows)
