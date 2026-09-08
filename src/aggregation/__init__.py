"""
src/aggregation/__init__.py
---------------------------
Phase 6 - Sentiment Aggregation package.

Public API
----------
    aggregate_sentiment(df)
    aggregate_by_brand(df, include_missing=False)
    aggregate_by_product(df, include_missing=False)
    aggregate_by_topic(df, include_missing=False)
    aggregate_by_platform(df, include_missing=False)
    aggregate_sentiment_over_time(df, freq="hourly")
    aggregate_by_dimension_over_time(df, dimension="brand", freq="hourly")
    aggregate_by_entities(df, dimensions=["brand","product","topic"], freq="hourly")
"""

from .aggregator import (
    aggregate_sentiment,
    aggregate_by_brand,
    aggregate_by_product,
    aggregate_by_topic,
    aggregate_by_platform,
    aggregate_sentiment_over_time,
    aggregate_by_dimension_over_time,
    aggregate_by_entities,
)

from .metrics import (
    compute_sentiment_counts,
    compute_ratios,
    compute_average_score,
)

__all__ = [
    "aggregate_sentiment",
    "aggregate_by_brand",
    "aggregate_by_product",
    "aggregate_by_topic",
    "aggregate_by_platform",
    "aggregate_sentiment_over_time",
    "aggregate_by_dimension_over_time",
    "aggregate_by_entities",
    "compute_sentiment_counts",
    "compute_ratios",
    "compute_average_score",
]
