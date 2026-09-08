"""
src/time_series/builder.py
--------------------------
Phase 6 - Time-Series Builder.

Converts the global aggregated time-series into a richer structure with:
  - Complete time index (no gaps inside the observed range)
  - Causal rolling statistics (rolling mean, rolling std)
  - Momentum / change columns

Causal rolling windows
----------------------
Only data from t-window+1 ... t is used to compute the rolling value at
time t.  Future data is NEVER used.  This is enforced by using
``min_periods=1`` and the default ``closed="left"``-exclusive-right
pandas rolling behaviour (center=False by default).
"""

import numpy as np
import pandas as pd

from src.aggregation.aggregator import aggregate_sentiment_over_time


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_time_series(
    df: pd.DataFrame,
    freq: str = "hourly",
) -> pd.DataFrame:
    """Build a complete, gap-filled time-series from a normalised post DataFrame.

    Calls :func:`src.aggregation.aggregator.aggregate_sentiment_over_time`
    and then fills any gaps inside the observed range so that every time
    bucket is represented.

    Parameters
    ----------
    df:
        Normalised post DataFrame with a ``timestamp`` column.
    freq:
        Aggregation frequency. Accepts ``"hourly"``, ``"daily"``,
        ``"weekly"``, ``"monthly"`` or any valid pandas offset alias.

    Returns
    -------
    DataFrame with columns:
        timestamp, post_count, average_sentiment_score,
        positive, neutral, negative,
        positive_ratio, neutral_ratio, negative_ratio

    Notes
    -----
    * Gaps (buckets with no posts) have:
        - ``post_count = 0``
        - ``average_sentiment_score = NaN``
        - sentiment counts = 0
        - sentiment ratios = NaN
    """
    ts = aggregate_sentiment_over_time(df, freq=freq)
    if ts.empty:
        return ts

    # Ensure timestamp is datetime
    ts["timestamp"] = pd.to_datetime(ts["timestamp"], utc=True, errors="coerce")
    ts = ts.set_index("timestamp").sort_index()

    # Determine pandas offset string for reindex
    from src.aggregation.aggregator import _resolve_freq, _safe_resample_freq
    freq_alias = _resolve_freq(freq)
    freq_alias = _safe_resample_freq(ts, freq_alias)

    # Build complete date range between first and last observed timestamp
    start = ts.index.min()
    end = ts.index.max()
    full_index = pd.date_range(start=start, end=end, freq=freq_alias, tz="UTC")

    ts = ts.reindex(full_index)

    # Fill missing buckets correctly
    ts["post_count"] = ts["post_count"].fillna(0).astype(int)
    for col in ["positive", "neutral", "negative"]:
        if col in ts.columns:
            ts[col] = ts[col].fillna(0).astype(int)
    # average_sentiment_score, positive/neutral/negative ratios stay NaN

    ts.index.name = "timestamp"
    ts = ts.reset_index()
    return ts


def add_rolling_statistics(
    time_series: pd.DataFrame,
    value_column: str,
    window: int = 3,
) -> pd.DataFrame:
    """Add causal rolling mean and rolling standard deviation columns.

    The rolling window is **strictly causal**: the value at position *t* is
    computed using data from positions ``max(0, t-window+1)`` through ``t``
    only.  Future data is never used (``center=False``).

    Parameters
    ----------
    time_series:
        DataFrame produced by :func:`build_time_series` or
        :func:`~src.aggregation.aggregator.aggregate_sentiment_over_time`.
        Must have a ``timestamp`` column.
    value_column:
        Name of the numeric column to compute rolling stats on.
    window:
        Rolling window size (number of periods).  Must be >= 1.

    Returns
    -------
    A **copy** of *time_series* with two new columns appended:
        ``<value_column>_rolling_mean``
        ``<value_column>_rolling_std``

    Notes
    -----
    * NaN values in *value_column* are propagated into rolling calculations
      (they do not reset the window; ``min_periods=1`` is used so that a
      window with at least one non-NaN value still yields a result).
    * The returned DataFrame is sorted by ``timestamp``.
    * Modifying the returned DataFrame does NOT affect the input.

    Raises
    ------
    ValueError
        If *value_column* is not present in *time_series*.
    ValueError
        If *window* < 1.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window!r}.")

    if value_column not in time_series.columns:
        raise ValueError(
            f"Column '{value_column}' not found in time_series. "
            f"Available columns: {list(time_series.columns)}"
        )

    result = time_series.copy()
    result = result.sort_values("timestamp").reset_index(drop=True)

    series = pd.to_numeric(result[value_column], errors="coerce")

    # Causal rolling: center=False (default), min_periods=1
    rolling = series.rolling(window=window, min_periods=1, center=False)

    mean_col = f"{value_column}_rolling_mean"
    std_col = f"{value_column}_rolling_std"

    result[mean_col] = rolling.mean()
    result[std_col] = rolling.std(ddof=1)  # sample std; NaN for window < 2

    return result
