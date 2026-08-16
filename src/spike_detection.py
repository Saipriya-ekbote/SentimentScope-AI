"""Statistical spike detection for negative sentiment activity."""

from __future__ import annotations

from typing import Any

import pandas as pd

DEFAULT_Z_THRESHOLD = 2.0
MIN_OBSERVATIONS = 3


def detect_negative_spikes(
    time_series: pd.DataFrame,
    value_column: str = "negative",
    z_threshold: float = DEFAULT_Z_THRESHOLD,
    window: int = 3,
) -> list[dict[str, Any]]:
    """Detect unusual increases in negative sentiment using rolling z-scores."""
    if value_column not in time_series.columns:
        raise ValueError(f"Column not found: {value_column}")

    if time_series.empty:
        return []

    if len(time_series) < MIN_OBSERVATIONS:
        return []

    series = time_series.copy().sort_values("period").reset_index(drop=True)
    rolling_mean = (
        series[value_column].rolling(window=window, min_periods=window).mean().shift(1)
    )
    rolling_std = (
        series[value_column].rolling(window=window, min_periods=window).std().shift(1)
    )

    spikes: list[dict[str, Any]] = []
    for index, row in series.iterrows():
        observed = float(row[value_column])
        baseline = rolling_mean.iloc[index]
        std_dev = rolling_std.iloc[index]

        if pd.isna(baseline) or pd.isna(std_dev):
            continue

        if std_dev == 0:
            z_score = float("inf") if observed > baseline else 0.0
        else:
            z_score = (observed - baseline) / std_dev

        if z_score > z_threshold:
            spikes.append(
                {
                    "timestamp": row["period"],
                    "observed_value": observed,
                    "baseline_value": float(baseline),
                    "z_score": float(z_score),
                    "threshold": z_threshold,
                }
            )

    return spikes
