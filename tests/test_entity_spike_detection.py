"""Tests for Phase 6.2 entity-specific spike detection."""

from __future__ import annotations

import pandas as pd

from src.entity_spike_detection import (
    detect_all_entity_spikes,
    detect_entity_spikes,
)


def _make_entity_data() -> pd.DataFrame:
    """Create a deterministic dataset containing a Samsung spike."""

    timestamps = pd.date_range(
        "2026-01-01 00:00:00",
        periods=6,
        freq="h",
        tz="UTC",
    )

    negative_counts = [1, 1, 1, 1, 1, 10]

    rows = []

    post_id = 1

    for timestamp, negative_count in zip(timestamps, negative_counts):
        for _ in range(negative_count):
            rows.append(
                {
                    "post_id": str(post_id),
                    "timestamp": timestamp,
                    "text": "Samsung Galaxy S25 battery is terrible",
                    "sentiment": "negative",
                    "sentiment_score": -0.9,
                    "brand": "Samsung",
                    "product": "Galaxy S25",
                    "topic": "battery",
                }
            )
            post_id += 1

    return pd.DataFrame(rows)


def test_entity_spike_detection_returns_entity_information():
    df = _make_entity_data()

    spikes = detect_entity_spikes(
        df,
        dimension="brand",
        z_threshold=2.0,
        freq="hourly",
    )

    assert spikes
    assert all(spike["dimension"] == "brand" for spike in spikes)
    assert all(spike["entity"] == "Samsung" for spike in spikes)


def test_entity_spike_detects_negative_increase():
    df = _make_entity_data()

    spikes = detect_entity_spikes(
        df,
        dimension="brand",
        z_threshold=2.0,
        freq="hourly",
    )

    assert any(spike["observed_value"] == 10.0 for spike in spikes)


def test_product_spike_detection():
    df = _make_entity_data()

    spikes = detect_entity_spikes(
        df,
        dimension="product",
        z_threshold=2.0,
        freq="hourly",
    )

    assert spikes
    assert all(spike["entity"] == "Galaxy S25" for spike in spikes)


def test_topic_spike_detection():
    df = _make_entity_data()

    spikes = detect_entity_spikes(
        df,
        dimension="topic",
        z_threshold=2.0,
        freq="hourly",
    )

    assert spikes
    assert all(spike["entity"] == "battery" for spike in spikes)


def test_no_spike_for_stable_entity():
    df = _make_entity_data()

    # Make every hour have exactly one negative post.
    df = df.groupby("timestamp", as_index=False).head(1).copy()

    spikes = detect_entity_spikes(
        df,
        dimension="brand",
        z_threshold=2.0,
        freq="hourly",
    )

    assert spikes == []


def test_all_entity_dimensions():
    df = _make_entity_data()

    spikes = detect_all_entity_spikes(
        df,
        dimensions=["brand", "product", "topic"],
        z_threshold=2.0,
        freq="hourly",
    )

    dimensions = {spike["dimension"] for spike in spikes}

    assert "brand" in dimensions
    assert "product" in dimensions
    assert "topic" in dimensions