"""
src/insights/analyzer.py
------------------------
Phase 7 - Explainable AI Insight Engine.

Consumes outputs from:
  - Sentiment classification & continuous scoring
  - Entity detection (brand, product, topic)
  - Time-series aggregation
  - Spike detection and alert generation

Produces structured, explainable business insights:
  - Macro sentiment trend classification (improving, stable, declining, volatile)
  - Entity and topic negative concentration analysis
  - Spike root-cause driver correlation
  - Prioritized structured insight objects
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.insights.severity import (
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    calculate_severity,
    priority_from_severity,
)


def analyze_sentiment_trend(time_series: pd.DataFrame) -> dict[str, Any]:
    """Classify overall sentiment trajectory from time-series metrics.

    Evaluates chronological net sentiment (positive_ratio - negative_ratio)
    over time to determine whether customer sentiment is improving, stable,
    declining, or volatile.

    Parameters
    ----------
    time_series : pd.DataFrame
        Aggregated time series with columns 'timestamp', 'positive_ratio',
        'negative_ratio', 'post_count', and optionally 'average_sentiment_score'.

    Returns
    -------
    dict[str, Any]
        Dictionary with:
          - 'direction': 'improving' | 'stable' | 'declining' | 'volatile' | 'insufficient_data'
          - 'label': Human-readable label (e.g. 'Improving', 'Declining')
          - 'net_change': Difference in net sentiment between recent and baseline periods
          - 'volatility': Standard deviation of net sentiment across periods
          - 'recent_net_sentiment': Average net sentiment in recent periods
          - 'baseline_net_sentiment': Average net sentiment in initial periods
          - 'total_periods': Count of non-empty periods evaluated
    """
    if time_series is None or time_series.empty:
        return {
            "direction": "insufficient_data",
            "label": "Insufficient Data",
            "net_change": 0.0,
            "volatility": 0.0,
            "recent_net_sentiment": 0.0,
            "baseline_net_sentiment": 0.0,
            "total_periods": 0,
            "description": "No time-series periods available to evaluate sentiment trend.",
        }

    # Defensive copy & sort chronologically to strictly prevent future data leakage
    ts = time_series.copy()
    if "timestamp" in ts.columns:
        ts["timestamp"] = pd.to_datetime(ts["timestamp"], utc=True, errors="coerce")
        ts = ts.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Filter out empty buckets where post_count == 0
    if "post_count" in ts.columns:
        ts = ts[ts["post_count"] > 0].reset_index(drop=True)

    if len(ts) < 2:
        return {
            "direction": "insufficient_data",
            "label": "Insufficient Data",
            "net_change": 0.0,
            "volatility": 0.0,
            "recent_net_sentiment": 0.0,
            "baseline_net_sentiment": 0.0,
            "total_periods": len(ts),
            "description": "At least two active time periods are required for trend classification.",
        }

    # Net sentiment = positive_ratio - negative_ratio, bounded in [-1.0, 1.0]
    if "positive_ratio" in ts.columns and "negative_ratio" in ts.columns:
        pos = ts["positive_ratio"].fillna(0.0).to_numpy(dtype=float)
        neg = ts["negative_ratio"].fillna(0.0).to_numpy(dtype=float)
        net_series = pos - neg
    elif "average_sentiment_score" in ts.columns:
        net_series = ts["average_sentiment_score"].fillna(0.0).to_numpy(dtype=float)
    else:
        return {
            "direction": "insufficient_data",
            "label": "Insufficient Data",
            "net_change": 0.0,
            "volatility": 0.0,
            "recent_net_sentiment": 0.0,
            "baseline_net_sentiment": 0.0,
            "total_periods": len(ts),
            "description": "Missing sentiment ratio columns for trend evaluation.",
        }

    n_points = len(net_series)
    volatility = float(np.std(net_series, ddof=1)) if n_points > 1 else 0.0

    # Split into chronological baseline (earlier half) and recent (later half)
    half = max(1, n_points // 2)
    baseline_val = float(np.mean(net_series[:half]))
    recent_val = float(np.mean(net_series[half:]))
    net_change = float(recent_val - baseline_val)

    # Measure volatility vs monotonic trend using detrended residual variance and sign alternations
    x = np.arange(n_points)
    if n_points >= 3:
        p = np.polyfit(x, net_series, 1)
        fitted = np.polyval(p, x)
        residual_std = float(np.std(net_series - fitted, ddof=1))
        diffs = np.diff(net_series)
        alternations = int(np.sum(diffs[:-1] * diffs[1:] < 0)) if len(diffs) > 1 else 0
    else:
        residual_std = volatility
        alternations = 0

    # Classification logic:
    # 1. Volatile: requires at least 3 periods with high residual variance (residual_std >= 0.20) or frequent alternations with volatility >= 0.25
    # 2. Improving: net sentiment shifted upward by >= +0.05
    # 3. Declining: net sentiment shifted downward by <= -0.05
    # 4. Stable: minor fluctuations within [-0.05, +0.05]
    if n_points >= 3 and ((residual_std >= 0.20 and volatility >= 0.25) or (alternations >= 2 and volatility >= 0.25)):
        direction = "volatile"
        label = "Highly Volatile"
        desc = (
            f"Sentiment is exhibiting significant volatility (std={volatility:.2f}) "
            "with frequent shifts between positive and negative sentiment across periods."
        )
    elif net_change <= -0.05:
        direction = "declining"
        label = "Declining"
        desc = (
            f"Net sentiment deteriorated by {abs(net_change) * 100:.1f}% "
            f"from baseline ({baseline_val:+.2f}) to recent periods ({recent_val:+.2f})."
        )
    elif net_change >= 0.05:
        direction = "improving"
        label = "Improving"
        desc = (
            f"Net sentiment improved by {net_change * 100:.1f}% "
            f"from baseline ({baseline_val:+.2f}) to recent periods ({recent_val:+.2f})."
        )
    else:
        direction = "stable"
        label = "Stable"
        desc = (
            f"Sentiment remains stable (net change of {net_change * 100:+.1f}%) "
            f"with low periodic volatility (std={volatility:.2f})."
        )

    return {
        "direction": direction,
        "label": label,
        "net_change": round(net_change, 4),
        "volatility": round(volatility, 4),
        "recent_net_sentiment": round(recent_val, 4),
        "baseline_net_sentiment": round(baseline_val, 4),
        "total_periods": n_points,
        "description": desc,
    }


def analyze_entity_concentration(df: pd.DataFrame) -> dict[str, Any]:
    """Analyze negative sentiment concentration across brands, products, and topics.

    Identifies which entities account for the largest proportion of negative
    sentiment and customer friction.

    Parameters
    ----------
    df : pd.DataFrame
        Normalized DataFrame with 'sentiment' and entity columns.

    Returns
    -------
    dict[str, Any]
        Summary containing:
          - 'total_negative_posts': Total negative post count
          - 'top_brands': Top affected brands ranked by negative posts
          - 'top_products': Top affected products ranked by negative posts
          - 'top_topics': Top topics ranked by negative posts
          - 'primary_focal_point': Most heavily impacted entity across all dimensions
    """
    if df is None or df.empty or "sentiment" not in df.columns:
        return {
            "total_negative_posts": 0,
            "top_brands": [],
            "top_products": [],
            "top_topics": [],
            "primary_focal_point": None,
        }

    neg_df = df[df["sentiment"] == "negative"]
    total_negative = len(neg_df)

    def _rank_dimension(dim_col: str, min_count: int = 1) -> list[dict[str, Any]]:
        if dim_col not in df.columns:
            return []

        # Filter out missing/blank values
        valid = df[df[dim_col].notna() & (df[dim_col].astype(str).str.strip() != "")]
        if valid.empty:
            return []

        stats = []
        for name, group in valid.groupby(dim_col):
            total_posts = len(group)
            neg_posts = len(group[group["sentiment"] == "negative"])
            if neg_posts < min_count:
                continue
            neg_ratio = neg_posts / total_posts if total_posts > 0 else 0.0
            share_of_all_neg = (neg_posts / total_negative * 100.0) if total_negative > 0 else 0.0

            stats.append(
                {
                    "name": str(name),
                    "dimension": dim_col,
                    "negative_count": neg_posts,
                    "total_count": total_posts,
                    "negative_ratio": round(neg_ratio, 4),
                    "share_of_all_negative": round(share_of_all_neg, 1),
                }
            )

        # Sort primarily by negative post count descending, then negative ratio
        return sorted(
            stats,
            key=lambda x: (x["negative_count"], x["negative_ratio"]),
            reverse=True,
        )

    top_brands = _rank_dimension("brand")
    top_products = _rank_dimension("product")
    top_topics = _rank_dimension("topic")

    primary_focal_point = None
    all_entities = top_brands + top_products + top_topics
    if all_entities:
        primary_focal_point = max(all_entities, key=lambda x: x["negative_count"])

    return {
        "total_negative_posts": total_negative,
        "top_brands": top_brands,
        "top_products": top_products,
        "top_topics": top_topics,
        "primary_focal_point": primary_focal_point,
    }


def correlate_spike_drivers(
    df: pd.DataFrame,
    alert: dict[str, Any],
) -> dict[str, Any]:
    """Identify the underlying topic and entity drivers for a detected alert.

    Inspects posts at the alert's timestamp to uncover the primary co-occurring
    topics or brands explaining *why* the spike occurred.

    Parameters
    ----------
    df : pd.DataFrame
        Normalized post DataFrame with timestamp, sentiment, brand, product, topic.
    alert : dict[str, Any]
        Alert dictionary containing 'timestamp', and optional 'dimension', 'entity'.

    Returns
    -------
    dict[str, Any]
        Dictionary with:
          - 'driver_topic': Primary co-occurring topic during spike
          - 'driver_topic_share': Percentage of negative spike posts on this topic
          - 'driver_brand': Primary co-occurring brand
          - 'driver_brand_share': Percentage of negative spike posts on this brand
          - 'total_spike_negative_posts': Number of negative posts at that period
    """
    if df is None or df.empty or "timestamp" not in df.columns or "sentiment" not in df.columns:
        return {
            "driver_topic": None,
            "driver_topic_share": 0.0,
            "driver_brand": None,
            "driver_brand_share": 0.0,
            "total_spike_negative_posts": 0,
        }

    alert_time = alert.get("timestamp")
    dimension = alert.get("dimension")
    entity = alert.get("entity")

    work_df = df.copy()
    work_df["parsed_time"] = pd.to_datetime(work_df["timestamp"], utc=True, errors="coerce")

    # Match posts to the alert period
    matching = work_df
    if alert_time is not None and not pd.isna(alert_time):
        alert_dt = pd.to_datetime(alert_time, utc=True, errors="coerce")
        if not pd.isna(alert_dt):
            # Check if timestamps match at date level or hour level
            same_day = work_df["parsed_time"].dt.date == alert_dt.date()
            if same_day.any():
                matching = work_df[same_day]

    # If this is an entity alert, isolate posts for that entity
    if dimension and entity and dimension in matching.columns:
        matching = matching[matching[dimension].astype(str).str.lower() == str(entity).lower()]

    neg_matching = matching[matching["sentiment"] == "negative"]
    spike_neg_count = len(neg_matching)

    driver_topic = None
    driver_topic_share = 0.0
    if "topic" in neg_matching.columns and spike_neg_count > 0:
        valid_topics = neg_matching[neg_matching["topic"].notna() & (neg_matching["topic"].astype(str).str.strip() != "")]
        if not valid_topics.empty:
            topic_counts = valid_topics["topic"].value_counts()
            top_topic = topic_counts.index[0]
            top_count = int(topic_counts.iloc[0])
            driver_topic = str(top_topic)
            driver_topic_share = round((top_count / spike_neg_count) * 100.0, 1)

    driver_brand = None
    driver_brand_share = 0.0
    if "brand" in neg_matching.columns and spike_neg_count > 0:
        valid_brands = neg_matching[neg_matching["brand"].notna() & (neg_matching["brand"].astype(str).str.strip() != "")]
        if not valid_brands.empty:
            brand_counts = valid_brands["brand"].value_counts()
            top_b = brand_counts.index[0]
            b_count = int(brand_counts.iloc[0])
            driver_brand = str(top_b)
            driver_brand_share = round((b_count / spike_neg_count) * 100.0, 1)

    return {
        "driver_topic": driver_topic,
        "driver_topic_share": driver_topic_share,
        "driver_brand": driver_brand,
        "driver_brand_share": driver_brand_share,
        "total_spike_negative_posts": spike_neg_count,
    }


def generate_insights(
    df: pd.DataFrame,
    time_series: pd.DataFrame,
    alerts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate structured, explainable AI insights from pipeline outputs.

    Transforms raw alerts, time-series metrics, and entity aggregations into
    prioritized, actionable business insights with empirical evidence.

    Parameters
    ----------
    df : pd.DataFrame
        Normalized post DataFrame.
    time_series : pd.DataFrame
        Aggregated time-series DataFrame.
    alerts : list[dict[str, Any]]
        List of detected global and entity spike alerts.

    Returns
    -------
    list[dict[str, Any]]
        Sorted list of structured insight records.
    """
    insights: list[dict[str, Any]] = []

    # 1. Macro Trend Analysis
    trend_result = analyze_sentiment_trend(time_series)
    concentration_result = analyze_entity_concentration(df)

    # If overall sentiment trend is declining, emit a strategic macro insight
    if trend_result["direction"] == "declining":
        insights.append(
            {
                "insight_id": "macro_sentiment_decline",
                "insight_type": "sentiment_decline",
                "severity": SEVERITY_HIGH,
                "priority": priority_from_severity(SEVERITY_HIGH),
                "entity_dimension": "global",
                "entity_name": "Overall Social Stream",
                "timestamp": str(time_series["timestamp"].iloc[-1]) if ("timestamp" in time_series.columns and not time_series.empty) else None,
                "primary_topic": concentration_result["top_topics"][0]["name"] if concentration_result["top_topics"] else None,
                "summary": "Overall social sentiment is deteriorating over time.",
                "evidence": {
                    "net_change": trend_result["net_change"],
                    "recent_net_sentiment": trend_result["recent_net_sentiment"],
                    "baseline_net_sentiment": trend_result["baseline_net_sentiment"],
                    "volatility": trend_result["volatility"],
                    "total_periods": trend_result["total_periods"],
                    "top_driver": concentration_result["primary_focal_point"],
                },
                "explanation": (
                    f"Net sentiment has decreased by {abs(trend_result['net_change']) * 100:.1f}% "
                    f"compared to the earlier baseline. "
                    + (
                        f"Friction is concentrated around {concentration_result['primary_focal_point']['dimension']} "
                        f"'{concentration_result['primary_focal_point']['name']}'."
                        if concentration_result["primary_focal_point"]
                        else "Friction is distributed across multiple channels."
                    )
                ),
            }
        )

    # 2. Convert Spike Alerts into Explainable Operational Insights
    for idx, alert in enumerate(alerts):
        observed = float(alert.get("observed_value", 0.0))
        baseline = float(alert.get("baseline_value", 0.0))
        z_score = float(alert.get("z_score", 0.0))
        increase_percent = float(alert.get("increase_percent", 0.0))
        timestamp = alert.get("timestamp")
        dimension = alert.get("dimension")
        entity = alert.get("entity")

        # Correlate underlying root causes at spike time
        driver_info = correlate_spike_drivers(df, alert)

        # Multi-factor severity evaluation: respect alert severity if set, or calculate
        if "severity" in alert and alert["severity"] in (SEVERITY_CRITICAL, SEVERITY_HIGH, SEVERITY_MEDIUM, SEVERITY_LOW):
            severity = alert["severity"]
        else:
            severity = calculate_severity(
                z_score=z_score,
                observed_count=observed,
                increase_percent=increase_percent,
                negative_ratio=1.0,
            )
        priority = priority_from_severity(severity)

        # Compose entity label and focal point
        if dimension and entity:
            entity_dim = str(dimension)
            entity_nm = str(entity)
            summary = f"Negative sentiment spike detected for {entity_dim.title()} '{entity_nm}'."
        else:
            entity_dim = "global"
            entity_nm = "Overall Stream"
            summary = "Macro surge in negative social sentiment detected."

        primary_topic = driver_info.get("driver_topic")
        topic_share = driver_info.get("driver_topic_share", 0.0)

        # Build natural explanation grounded in actual data
        explanation_parts = [
            (
                f"Negative sentiment surged {increase_percent:.1f}% above the expected baseline "
                f"({observed:.0f} observed vs. {baseline:.1f} baseline, z-score: {z_score:.2f})."
            )
        ]
        if primary_topic and topic_share > 0:
            explanation_parts.append(
                f"The spike is primarily driven by complaints concerning '{primary_topic}', "
                f"accounting for {topic_share:.1f}% of negative mentions during this period."
            )
        elif driver_info.get("driver_brand"):
            explanation_parts.append(
                f"The surge is concentrated around brand '{driver_info['driver_brand']}' "
                f"({driver_info.get('driver_brand_share', 0.0):.1f}% of spike negative posts)."
            )

        insights.append(
            {
                "insight_id": f"spike_alert_{idx}_{entity_dim}_{entity_nm}",
                "insight_type": "spike_escalation",
                "severity": severity,
                "priority": priority,
                "entity_dimension": entity_dim,
                "entity_name": entity_nm,
                "timestamp": str(timestamp) if timestamp is not None else None,
                "primary_topic": primary_topic,
                "summary": summary,
                "evidence": {
                    "observed_negative": observed,
                    "baseline_negative": baseline,
                    "increase_percent": increase_percent,
                    "z_score": z_score,
                    "primary_topic": primary_topic,
                    "topic_share": topic_share,
                    "co_occurring_brand": driver_info.get("driver_brand"),
                    "co_occurring_brand_share": driver_info.get("driver_brand_share", 0.0),
                },
                "explanation": " ".join(explanation_parts),
            }
        )

    # 3. High Negative Concentration Insights (Disproportionate entity friction)
    focal = concentration_result.get("primary_focal_point")
    total_neg = concentration_result.get("total_negative_posts", 0)
    if focal and total_neg >= 10 and focal.get("share_of_all_negative", 0.0) >= 25.0:
        focal_dim = focal["dimension"]
        focal_name = focal["name"]
        focal_share = focal["share_of_all_negative"]
        focal_count = focal["negative_count"]

        # Check if already covered by an identical spike alert
        already_alerted = any(
            ins.get("entity_name") == focal_name and ins.get("entity_dimension") == focal_dim
            for ins in insights
        )
        if not already_alerted:
            insights.append(
                {
                    "insight_id": f"concentration_{focal_dim}_{focal_name}",
                    "insight_type": "negative_concentration",
                    "severity": SEVERITY_MEDIUM if focal_share < 40.0 else SEVERITY_HIGH,
                    "priority": priority_from_severity(SEVERITY_MEDIUM if focal_share < 40.0 else SEVERITY_HIGH),
                    "entity_dimension": focal_dim,
                    "entity_name": focal_name,
                    "timestamp": None,
                    "primary_topic": None,
                    "summary": f"Disproportionate negative sentiment concentrated in {focal_dim.title()} '{focal_name}'.",
                    "evidence": {
                        "negative_count": focal_count,
                        "share_of_all_negative": focal_share,
                        "negative_ratio": focal["negative_ratio"],
                        "total_negative": total_neg,
                    },
                    "explanation": (
                        f"{focal_dim.title()} '{focal_name}' accounts for {focal_share:.1f}% "
                        f"({focal_count} of {total_neg}) of all negative posts in the dataset, "
                        f"with an internal negative sentiment ratio of {focal['negative_ratio'] * 100:.1f}%."
                    ),
                }
            )

    # Sort insights strictly by priority (1: CRITICAL first, then 2: HIGH, etc.)
    insights.sort(key=lambda item: item["priority"])

    return insights
