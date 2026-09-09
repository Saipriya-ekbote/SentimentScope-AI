"""
Entity-specific negative sentiment spike detection.

Phase 6.2
----------
Detect negative sentiment spikes separately for brands, products,
and topics.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.aggregation import aggregate_by_dimension_over_time
from src.spike_detection import detect_negative_spikes


DEFAULT_DIMENSIONS = ["brand", "product", "topic"]


def detect_entity_spikes(
    df: pd.DataFrame,
    dimension: str,
    z_threshold: float = 2.0,
    freq: str = "hourly",
) -> list[dict[str, Any]]:
    """
    Detect negative sentiment spikes for each entity in a dimension.

    Parameters
    ----------
    df:
        DataFrame containing timestamp, sentiment, sentiment_score,
        and the requested entity column.
    dimension:
        Entity dimension such as "brand", "product", or "topic".
    z_threshold:
        Z-score threshold used to identify a spike.
    freq:
        Time aggregation frequency.

    Returns
    -------
    list of dictionaries containing entity-specific spike information.
    """

    aggregated = aggregate_by_dimension_over_time(
        df,
        dimension=dimension,
        freq=freq,
    )

    if aggregated.empty:
        return []

    spikes: list[dict[str, Any]] = []

    for entity, entity_df in aggregated.groupby(dimension):
        # The existing spike detector expects a column named "period".
        series = entity_df.rename(columns={"timestamp": "period"}).reset_index(
            drop=True
        )

        entity_spikes = detect_negative_spikes(
            series,
            value_column="negative",
            z_threshold=z_threshold,
        )

        for spike in entity_spikes:
            spike["dimension"] = dimension
            spike["entity"] = entity

        spikes.extend(entity_spikes)

    return sorted(
        spikes,
        key=lambda item: item["timestamp"],
    )


def detect_all_entity_spikes(
    df: pd.DataFrame,
    dimensions: list[str] | None = None,
    z_threshold: float = 2.0,
    freq: str = "hourly",
) -> list[dict[str, Any]]:
    """
    Detect negative sentiment spikes across multiple entity dimensions.
    """

    if dimensions is None:
        dimensions = DEFAULT_DIMENSIONS.copy()

    all_spikes: list[dict[str, Any]] = []

    for dimension in dimensions:
        all_spikes.extend(
            detect_entity_spikes(
                df,
                dimension=dimension,
                z_threshold=z_threshold,
                freq=freq,
            )
        )

    return sorted(
        all_spikes,
        key=lambda item: item["timestamp"],
    )