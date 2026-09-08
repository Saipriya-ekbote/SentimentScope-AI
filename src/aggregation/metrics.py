"""
src/aggregation/metrics.py
--------------------------
Phase 6 - Reusable metric computation helpers.

These are pure functions that operate on DataFrames and return scalar
values or dicts. They are free of side-effects and can be called
independently from the aggregator.
"""

import numpy as np
import pandas as pd
from typing import Dict


def compute_sentiment_counts(df: pd.DataFrame) -> Dict[str, int]:
    """Return the count of each sentiment label in *df*.

    Parameters
    ----------
    df:
        DataFrame containing a ``sentiment`` column with values drawn from
        ``{"positive", "neutral", "negative"}``.

    Returns
    -------
    dict
        Keys: ``positive``, ``neutral``, ``negative`` (always present).
        Values: integer counts.
    """
    counts: Dict[str, int] = {"positive": 0, "neutral": 0, "negative": 0}
    if df.empty or "sentiment" not in df.columns:
        return counts
    vc = df["sentiment"].value_counts()
    for label in counts:
        counts[label] = int(vc.get(label, 0))
    return counts


def compute_ratios(counts: Dict[str, int], total: int) -> Dict[str, float]:
    """Convert raw sentiment counts to ratios in [0, 1].

    Parameters
    ----------
    counts:
        Dict produced by :func:`compute_sentiment_counts`.
    total:
        Total number of posts (denominator). When *total* is zero all
        ratios are returned as ``float("nan")``.

    Returns
    -------
    dict
        Keys: ``positive_ratio``, ``neutral_ratio``, ``negative_ratio``.
    """
    if total == 0:
        return {
            "positive_ratio": float("nan"),
            "neutral_ratio": float("nan"),
            "negative_ratio": float("nan"),
        }
    return {
        "positive_ratio": counts.get("positive", 0) / total,
        "neutral_ratio": counts.get("neutral", 0) / total,
        "negative_ratio": counts.get("negative", 0) / total,
    }


def compute_average_score(df: pd.DataFrame) -> float:
    """Return the mean ``sentiment_score`` ignoring NaN values.

    Parameters
    ----------
    df:
        DataFrame that may contain a ``sentiment_score`` column.

    Returns
    -------
    float
        Mean score, or ``float("nan")`` when the column is absent or all
        values are NaN.
    """
    if df.empty or "sentiment_score" not in df.columns:
        return float("nan")
    col = pd.to_numeric(df["sentiment_score"], errors="coerce")
    if col.isna().all():
        return float("nan")
    return float(col.mean(skipna=True))
