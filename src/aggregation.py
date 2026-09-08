"""
src/aggregation.py
------------------
Backward-compatibility facade for Phase 6.

All imports go through the modular src/aggregation/ package.
Do NOT add logic here; keep this file as a thin re-export.
"""

from src.aggregation.metrics import (  # noqa: F401
    compute_sentiment_counts,
    compute_ratios,
    compute_average_score,
)
from src.aggregation.aggregator import (  # noqa: F401
    aggregate_sentiment,
    aggregate_by_brand,
    aggregate_by_product,
    aggregate_by_topic,
    aggregate_by_platform,
    aggregate_sentiment_over_time,
    aggregate_by_dimension_over_time,
    aggregate_by_entities,
)

__all__ = [
    "compute_sentiment_counts",
    "compute_ratios",
    "compute_average_score",
    "aggregate_sentiment",
    "aggregate_by_brand",
    "aggregate_by_product",
    "aggregate_by_topic",
    "aggregate_by_platform",
    "aggregate_sentiment_over_time",
    "aggregate_by_dimension_over_time",
    "aggregate_by_entities",
]
