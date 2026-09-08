"""
src/aggregation/aggregator.py
-----------------------------
Phase 6 - Core aggregation functions.

All functions accept a normalised DataFrame (the 12-field schema from
Phase 2) and return either a summary dict or a tidy DataFrame.

Frequency aliases
-----------------
    "hourly"  -> "h"   (pandas offset alias)
    "daily"   -> "D"
    "weekly"  -> "W"
    "monthly" -> "ME"  (month-end, pandas >= 2.2) / "M" fallback

Missing time buckets
--------------------
    post_count             -> 0
    average_sentiment_score -> NaN   (NOT zero)
    positive/neutral/negative counts -> 0
    positive/neutral/negative ratios -> NaN
"""

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .metrics import compute_sentiment_counts, compute_ratios, compute_average_score

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_FREQ_MAP = {
    "hourly": "h",
    "daily": "D",
    "weekly": "W",
    "monthly": "ME",
}


def _resolve_freq(freq: str) -> str:
    """Translate human-friendly freq alias to a pandas offset string."""
    key = freq.lower().strip()
    if key in _FREQ_MAP:
        return _FREQ_MAP[key]
    # If caller already uses a pandas alias like "h", "D", etc., pass through.
    return freq


def _ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with ``timestamp`` parsed to datetime (UTC)."""
    df = df.copy()
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must contain a 'timestamp' column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def _build_summary_row(group_df: pd.DataFrame) -> Dict:
    """Build the standard stats dict for a group DataFrame."""
    total = len(group_df)
    counts = compute_sentiment_counts(group_df)
    ratios = compute_ratios(counts, total)
    avg_score = compute_average_score(group_df)
    return {
        "post_count": total,
        "average_sentiment_score": avg_score,
        **counts,
        **ratios,
    }


def _safe_resample_freq(df: pd.DataFrame, freq_alias: str) -> str:
    """Return freq alias, falling back from 'ME' to 'M' for older pandas."""
    if freq_alias == "ME":
        try:
            # Probe whether pandas accepts ME
            pd.tseries.frequencies.to_offset("ME")
            return "ME"
        except ValueError:
            return "M"
    return freq_alias


# ---------------------------------------------------------------------------
# 1. Overall aggregation
# ---------------------------------------------------------------------------


def aggregate_sentiment(df: pd.DataFrame) -> Dict:
    """Compute overall sentiment statistics across the entire DataFrame.

    Parameters
    ----------
    df:
        Normalised post DataFrame.

    Returns
    -------
    dict with keys:
        post_count, average_sentiment_score,
        positive, neutral, negative,
        positive_ratio, neutral_ratio, negative_ratio
    """
    return _build_summary_row(df)


# ---------------------------------------------------------------------------
# 2. Per-dimension aggregation helpers
# ---------------------------------------------------------------------------


def _aggregate_by_column(
    df: pd.DataFrame,
    column: str,
    include_missing: bool = False,
) -> pd.DataFrame:
    """Group *df* by *column* and compute per-group sentiment stats.

    Parameters
    ----------
    df:
        Normalised post DataFrame.
    column:
        Column name to group by (e.g. "brand", "platform").
    include_missing:
        When True, include rows where *column* is None / empty string.

    Returns
    -------
    DataFrame sorted by ``post_count`` descending, with columns:
        <column>, post_count, average_sentiment_score,
        positive, neutral, negative,
        positive_ratio, neutral_ratio, negative_ratio
    """
    if column not in df.columns:
        return pd.DataFrame(
            columns=[
                column,
                "post_count",
                "average_sentiment_score",
                "positive",
                "neutral",
                "negative",
                "positive_ratio",
                "neutral_ratio",
                "negative_ratio",
            ]
        )

    work = df.copy()
    if not include_missing:
        mask = work[column].notna() & (work[column].astype(str).str.strip() != "")
        work = work[mask]

    rows = []
    for value, group in work.groupby(column, dropna=not include_missing):
        row = {"group_value": value}
        row.update(_build_summary_row(group))
        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                column,
                "post_count",
                "average_sentiment_score",
                "positive",
                "neutral",
                "negative",
                "positive_ratio",
                "neutral_ratio",
                "negative_ratio",
            ]
        )

    result = pd.DataFrame(rows)
    result = result.rename(columns={"group_value": column})
    result = result.sort_values("post_count", ascending=False).reset_index(drop=True)
    return result


def aggregate_by_brand(
    df: pd.DataFrame, include_missing: bool = False
) -> pd.DataFrame:
    """Aggregate sentiment statistics grouped by ``brand``.

    Parameters
    ----------
    df:
        Normalised post DataFrame.
    include_missing:
        When True include posts where brand is undetected (None / empty).

    Returns
    -------
    DataFrame sorted by ``post_count`` descending.
    """
    return _aggregate_by_column(df, "brand", include_missing=include_missing)


def aggregate_by_product(
    df: pd.DataFrame, include_missing: bool = False
) -> pd.DataFrame:
    """Aggregate sentiment statistics grouped by ``product``."""
    return _aggregate_by_column(df, "product", include_missing=include_missing)


def aggregate_by_topic(
    df: pd.DataFrame, include_missing: bool = False
) -> pd.DataFrame:
    """Aggregate sentiment statistics grouped by ``topic``."""
    return _aggregate_by_column(df, "topic", include_missing=include_missing)


def aggregate_by_platform(
    df: pd.DataFrame, include_missing: bool = False
) -> pd.DataFrame:
    """Aggregate sentiment statistics grouped by ``platform``."""
    return _aggregate_by_column(df, "platform", include_missing=include_missing)


# ---------------------------------------------------------------------------
# 3. Time-series aggregation
# ---------------------------------------------------------------------------


def aggregate_sentiment_over_time(
    df: pd.DataFrame,
    freq: str = "hourly",
) -> pd.DataFrame:
    """Aggregate global sentiment statistics into time buckets.

    Parameters
    ----------
    df:
        Normalised post DataFrame with a ``timestamp`` column.
    freq:
        Aggregation frequency.  Accepts ``"hourly"``, ``"daily"``,
        ``"weekly"``, ``"monthly"`` **or** any valid pandas offset alias
        (e.g. ``"h"``, ``"D"``).

    Returns
    -------
    DataFrame indexed by ``timestamp`` (period start) with columns:
        post_count, average_sentiment_score,
        positive, neutral, negative,
        positive_ratio, neutral_ratio, negative_ratio

    Notes
    -----
    * Buckets with zero posts have ``post_count=0`` and
      ``average_sentiment_score=NaN`` (not zero).
    * Only buckets inside the observed time range are included.
    """
    df = _ensure_timestamp(df)
    freq_alias = _resolve_freq(freq)
    freq_alias = _safe_resample_freq(df, freq_alias)

    if df["timestamp"].isna().all() or df.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "post_count",
                "average_sentiment_score",
                "positive",
                "neutral",
                "negative",
                "positive_ratio",
                "neutral_ratio",
                "negative_ratio",
            ]
        )

    # Set timestamp as index for resampling
    ts_df = df.set_index("timestamp").sort_index()

    def _agg_group(g):
        total = len(g)
        counts = compute_sentiment_counts(g.reset_index())
        ratios = compute_ratios(counts, total)
        avg = compute_average_score(g.reset_index())
        return pd.Series(
            {
                "post_count": total,
                "average_sentiment_score": avg,
                "positive": counts["positive"],
                "neutral": counts["neutral"],
                "negative": counts["negative"],
                "positive_ratio": ratios["positive_ratio"],
                "neutral_ratio": ratios["neutral_ratio"],
                "negative_ratio": ratios["negative_ratio"],
            }
        )

    result = ts_df.resample(freq_alias).apply(_agg_group)

    # Fill zero-post buckets correctly:
    # post_count / counts -> 0, ratios and avg_score -> NaN
    result["post_count"] = result["post_count"].fillna(0).astype(int)
    for col in ["positive", "neutral", "negative"]:
        result[col] = result[col].fillna(0).astype(int)
    # average_sentiment_score and ratios remain NaN for empty buckets

    result = result.reset_index()
    result = result.rename(columns={"timestamp": "timestamp"})
    return result


# ---------------------------------------------------------------------------
# 4. Per-dimension time-series aggregation
# ---------------------------------------------------------------------------


def aggregate_by_dimension_over_time(
    df: pd.DataFrame,
    dimension: str = "brand",
    freq: str = "hourly",
) -> pd.DataFrame:
    """Aggregate sentiment statistics per dimension value over time.

    Parameters
    ----------
    df:
        Normalised post DataFrame.
    dimension:
        Column to pivot on (``"brand"``, ``"product"``, ``"topic"``,
        ``"platform"``, or any other string column).
    freq:
        Aggregation frequency (same options as
        :func:`aggregate_sentiment_over_time`).

    Returns
    -------
    Long-format DataFrame with columns:
        timestamp, <dimension>, post_count, average_sentiment_score,
        positive, neutral, negative,
        positive_ratio, neutral_ratio, negative_ratio

    Notes
    -----
    * Only dimension values that appear in the data are included.
    * Missing time buckets for a given dimension value are **not** filled
      (sparse representation); the time-series builder handles gap-filling
      when required.
    """
    df = _ensure_timestamp(df)
    freq_alias = _resolve_freq(freq)
    freq_alias = _safe_resample_freq(df, freq_alias)

    if dimension not in df.columns:
        raise ValueError(
            f"Column '{dimension}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    valid = df[df[dimension].notna() & (df[dimension].astype(str).str.strip() != "")]
    if valid.empty or valid["timestamp"].isna().all():
        return pd.DataFrame(
            columns=[
                "timestamp",
                dimension,
                "post_count",
                "average_sentiment_score",
                "positive",
                "neutral",
                "negative",
                "positive_ratio",
                "neutral_ratio",
                "negative_ratio",
            ]
        )

    rows = []
    for dim_value, dim_group in valid.groupby(dimension):
        ts_df = dim_group.set_index("timestamp").sort_index()

        def _agg_group(g, _dv=dim_value):
            total = len(g)
            counts = compute_sentiment_counts(g.reset_index())
            ratios = compute_ratios(counts, total)
            avg = compute_average_score(g.reset_index())
            return pd.Series(
                {
                    "post_count": total,
                    "average_sentiment_score": avg,
                    "positive": counts["positive"],
                    "neutral": counts["neutral"],
                    "negative": counts["negative"],
                    "positive_ratio": ratios["positive_ratio"],
                    "neutral_ratio": ratios["neutral_ratio"],
                    "negative_ratio": ratios["negative_ratio"],
                }
            )

        resampled = ts_df.resample(freq_alias).apply(_agg_group)
        resampled = resampled[resampled["post_count"] > 0]  # drop empty buckets
        resampled[dimension] = dim_value
        resampled = resampled.reset_index()
        rows.append(resampled)

    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp",
                dimension,
                "post_count",
                "average_sentiment_score",
                "positive",
                "neutral",
                "negative",
                "positive_ratio",
                "neutral_ratio",
                "negative_ratio",
            ]
        )

    result = pd.concat(rows, ignore_index=True)
    for col in ["post_count", "positive", "neutral", "negative"]:
        result[col] = result[col].fillna(0).astype(int)

    col_order = (
        ["timestamp", dimension, "post_count", "average_sentiment_score"]
        + ["positive", "neutral", "negative"]
        + ["positive_ratio", "neutral_ratio", "negative_ratio"]
    )
    result = result[col_order].sort_values(["timestamp", dimension]).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# 5. Multi-dimension aggregation
# ---------------------------------------------------------------------------


def aggregate_by_entities(
    df: pd.DataFrame,
    dimensions: Optional[List[str]] = None,
    freq: str = "hourly",
) -> Dict[str, pd.DataFrame]:
    """Run :func:`aggregate_by_dimension_over_time` for multiple dimensions.

    Parameters
    ----------
    df:
        Normalised post DataFrame.
    dimensions:
        List of column names to aggregate over.  Defaults to
        ``["brand", "product", "topic"]``.
    freq:
        Aggregation frequency (same options as
        :func:`aggregate_sentiment_over_time`).

    Returns
    -------
    dict mapping each dimension name to its aggregated DataFrame.
    """
    if dimensions is None:
        dimensions = ["brand", "product", "topic"]

    result: Dict[str, pd.DataFrame] = {}
    for dim in dimensions:
        try:
            result[dim] = aggregate_by_dimension_over_time(df, dimension=dim, freq=freq)
        except ValueError as exc:
            warnings.warn(str(exc), stacklevel=2)
            result[dim] = pd.DataFrame()
    return result
