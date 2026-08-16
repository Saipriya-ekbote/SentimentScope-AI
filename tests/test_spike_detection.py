"""Tests for negative sentiment spike detection."""

from __future__ import annotations

import pandas as pd

from src.alerts import generate_alerts
from src.spike_detection import detect_negative_spikes


def test_no_spike_in_stable_series() -> None:
    series = pd.DataFrame(
        {
            "period": pd.date_range("2026-08-01", periods=5, freq="D"),
            "negative": [10, 11, 10, 9, 10],
        }
    )
    spikes = detect_negative_spikes(series, z_threshold=2.0)
    assert spikes == []


def test_detects_actual_spike() -> None:
    series = pd.DataFrame(
        {
            "period": pd.date_range("2026-08-01", periods=6, freq="D"),
            "negative": [8, 9, 10, 9, 45, 10],
        }
    )
    spikes = detect_negative_spikes(series, z_threshold=2.0, window=3)
    assert len(spikes) >= 1
    assert spikes[-1]["observed_value"] == 45.0


def test_insufficient_data_returns_no_spikes() -> None:
    series = pd.DataFrame(
        {
            "period": pd.date_range("2026-08-01", periods=2, freq="D"),
            "negative": [5, 20],
        }
    )
    assert detect_negative_spikes(series) == []


def test_generate_alerts_from_spikes() -> None:
    spikes = [
        {
            "timestamp": "2026-08-05",
            "observed_value": 45.0,
            "baseline_value": 12.0,
            "z_score": 3.5,
            "threshold": 2.0,
        }
    ]
    alerts = generate_alerts(spikes)
    assert len(alerts) == 1
    assert alerts[0]["alert_type"] == "NEGATIVE_SENTIMENT_SPIKE"
    assert alerts[0]["observed_value"] == 45.0
    assert alerts[0]["baseline_value"] == 12.0
