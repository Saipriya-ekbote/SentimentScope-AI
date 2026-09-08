"""
tests/test_aggregation.py
-------------------------
Phase 6 tests for src/aggregation/ package.

All tests use synthetic DataFrames so they run offline with no model files.
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.aggregation import (
    aggregate_sentiment,
    aggregate_by_brand,
    aggregate_by_product,
    aggregate_by_topic,
    aggregate_by_platform,
    aggregate_sentiment_over_time,
    aggregate_by_dimension_over_time,
    aggregate_by_entities,
    compute_sentiment_counts,
    compute_ratios,
    compute_average_score,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_df(n_pos=4, n_neu=3, n_neg=3, brand="ACME", platform="twitter"):
    """Build a simple synthetic normalised DataFrame."""
    rows = []
    base = pd.Timestamp("2024-01-01 10:00:00", tz="UTC")
    sentiments = (["positive"] * n_pos + ["neutral"] * n_neu + ["negative"] * n_neg)
    scores = ([0.8] * n_pos + [0.0] * n_neu + [-0.6] * n_neg)
    for i, (s, sc) in enumerate(zip(sentiments, scores)):
        rows.append(
            {
                "post_id": str(i),
                "platform": platform,
                "timestamp": base + pd.Timedelta(hours=i),
                "author": f"user_{i}",
                "text": f"post {i}",
                "brand": brand,
                "product": "WidgetPro" if i % 2 == 0 else "WidgetLite",
                "topic": "quality" if i % 3 == 0 else "support",
                "sentiment": s,
                "sentiment_score": sc,
                "engagement_metrics": "{}",
                "provenance": "csv",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_df():
    return _make_df()


@pytest.fixture
def multi_brand_df():
    """DataFrame with multiple brands and platforms."""
    df1 = _make_df(n_pos=5, n_neu=2, n_neg=3, brand="ACME", platform="twitter")
    df2 = _make_df(n_pos=2, n_neu=4, n_neg=4, brand="TechCorp", platform="reddit")
    return pd.concat([df1, df2], ignore_index=True)


# ---------------------------------------------------------------------------
# metrics.py tests
# ---------------------------------------------------------------------------


class TestComputeSentimentCounts:
    def test_basic(self, sample_df):
        counts = compute_sentiment_counts(sample_df)
        assert counts["positive"] == 4
        assert counts["neutral"] == 3
        assert counts["negative"] == 3

    def test_empty_df(self):
        counts = compute_sentiment_counts(pd.DataFrame())
        assert counts == {"positive": 0, "neutral": 0, "negative": 0}

    def test_missing_column(self):
        df = pd.DataFrame({"text": ["a", "b"]})
        counts = compute_sentiment_counts(df)
        assert counts == {"positive": 0, "neutral": 0, "negative": 0}

    def test_all_positive(self):
        df = pd.DataFrame({"sentiment": ["positive"] * 5})
        counts = compute_sentiment_counts(df)
        assert counts["positive"] == 5
        assert counts["neutral"] == 0
        assert counts["negative"] == 0


class TestComputeRatios:
    def test_basic(self):
        counts = {"positive": 5, "neutral": 3, "negative": 2}
        ratios = compute_ratios(counts, total=10)
        assert math.isclose(ratios["positive_ratio"], 0.5)
        assert math.isclose(ratios["neutral_ratio"], 0.3)
        assert math.isclose(ratios["negative_ratio"], 0.2)

    def test_zero_total(self):
        ratios = compute_ratios({"positive": 0, "neutral": 0, "negative": 0}, total=0)
        assert math.isnan(ratios["positive_ratio"])
        assert math.isnan(ratios["neutral_ratio"])
        assert math.isnan(ratios["negative_ratio"])


class TestComputeAverageScore:
    def test_basic(self, sample_df):
        avg = compute_average_score(sample_df)
        # (4*0.8 + 3*0.0 + 3*(-0.6)) / 10 = (3.2 - 1.8) / 10 = 0.14
        assert math.isclose(avg, 0.14, rel_tol=1e-5)

    def test_empty(self):
        assert math.isnan(compute_average_score(pd.DataFrame()))

    def test_all_nan(self):
        df = pd.DataFrame({"sentiment_score": [float("nan"), float("nan")]})
        assert math.isnan(compute_average_score(df))


# ---------------------------------------------------------------------------
# aggregate_sentiment tests
# ---------------------------------------------------------------------------


class TestAggregateSentiment:
    def test_basic(self, sample_df):
        result = aggregate_sentiment(sample_df)
        assert result["post_count"] == 10
        assert result["positive"] == 4
        assert result["neutral"] == 3
        assert result["negative"] == 3

    def test_ratios_sum_to_one(self, sample_df):
        result = aggregate_sentiment(sample_df)
        total = result["positive_ratio"] + result["neutral_ratio"] + result["negative_ratio"]
        assert math.isclose(total, 1.0, rel_tol=1e-9)

    def test_empty_df(self):
        result = aggregate_sentiment(pd.DataFrame(columns=["sentiment", "sentiment_score"]))
        assert result["post_count"] == 0

    def test_score_in_range(self, sample_df):
        result = aggregate_sentiment(sample_df)
        assert -1.0 <= result["average_sentiment_score"] <= 1.0


# ---------------------------------------------------------------------------
# aggregate_by_* tests
# ---------------------------------------------------------------------------


class TestAggregateByBrand:
    def test_returns_dataframe(self, multi_brand_df):
        result = aggregate_by_brand(multi_brand_df)
        assert isinstance(result, pd.DataFrame)
        assert "brand" in result.columns
        assert "post_count" in result.columns

    def test_sorted_descending(self, multi_brand_df):
        result = aggregate_by_brand(multi_brand_df)
        counts = result["post_count"].tolist()
        assert counts == sorted(counts, reverse=True)

    def test_exclude_missing_by_default(self):
        df = _make_df()
        df.loc[0, "brand"] = None
        result = aggregate_by_brand(df, include_missing=False)
        total = result["post_count"].sum()
        assert total == 9  # one row excluded

    def test_include_missing(self):
        df = _make_df()
        df.loc[0, "brand"] = None
        result = aggregate_by_brand(df, include_missing=True)
        total = result["post_count"].sum()
        assert total == 10

    def test_two_brands_present(self, multi_brand_df):
        result = aggregate_by_brand(multi_brand_df)
        assert len(result) == 2
        brands = set(result["brand"].tolist())
        assert brands == {"ACME", "TechCorp"}


class TestAggregateByProduct:
    def test_basic(self, sample_df):
        result = aggregate_by_product(sample_df)
        assert "product" in result.columns
        assert result["post_count"].sum() == 10


class TestAggregateByTopic:
    def test_basic(self, sample_df):
        result = aggregate_by_topic(sample_df)
        assert "topic" in result.columns


class TestAggregateByPlatform:
    def test_basic(self, multi_brand_df):
        result = aggregate_by_platform(multi_brand_df)
        assert "platform" in result.columns
        assert set(result["platform"].tolist()) == {"twitter", "reddit"}

    def test_counts_match(self, multi_brand_df):
        result = aggregate_by_platform(multi_brand_df)
        assert result["post_count"].sum() == len(multi_brand_df)


# ---------------------------------------------------------------------------
# aggregate_sentiment_over_time tests
# ---------------------------------------------------------------------------


class TestAggregateSentimentOverTime:
    def test_returns_dataframe(self, sample_df):
        result = aggregate_sentiment_over_time(sample_df, freq="hourly")
        assert isinstance(result, pd.DataFrame)
        assert "timestamp" in result.columns
        assert "post_count" in result.columns

    def test_post_counts_sum(self, sample_df):
        result = aggregate_sentiment_over_time(sample_df, freq="daily")
        # All posts are on 2024-01-01, so sum should equal total rows
        assert result["post_count"].sum() == 10

    def test_missing_buckets_have_zero_count(self):
        """Two posts separated by a large gap; hourly should produce zeros."""
        df = pd.DataFrame(
            {
                "timestamp": [
                    pd.Timestamp("2024-01-01 10:00:00", tz="UTC"),
                    pd.Timestamp("2024-01-01 13:00:00", tz="UTC"),
                ],
                "sentiment": ["positive", "negative"],
                "sentiment_score": [0.8, -0.7],
            }
        )
        result = aggregate_sentiment_over_time(df, freq="hourly")
        # Hours 10, 11, 12, 13 should exist
        assert len(result) >= 4
        zero_rows = result[result["post_count"] == 0]
        assert len(zero_rows) >= 2  # at least hours 11 and 12

    def test_missing_buckets_avg_score_is_nan(self):
        """Empty buckets must have NaN average_sentiment_score, not zero."""
        df = pd.DataFrame(
            {
                "timestamp": [
                    pd.Timestamp("2024-01-01 10:00:00", tz="UTC"),
                    pd.Timestamp("2024-01-01 13:00:00", tz="UTC"),
                ],
                "sentiment": ["positive", "negative"],
                "sentiment_score": [0.8, -0.7],
            }
        )
        result = aggregate_sentiment_over_time(df, freq="hourly")
        empty = result[result["post_count"] == 0]
        # average_sentiment_score for zero-count rows should be NaN
        assert empty["average_sentiment_score"].isna().all()

    def test_freq_daily(self, sample_df):
        result = aggregate_sentiment_over_time(sample_df, freq="daily")
        assert len(result) >= 1

    def test_freq_alias_passthrough(self, sample_df):
        """Pandas offset alias 'D' should work directly."""
        result = aggregate_sentiment_over_time(sample_df, freq="D")
        assert len(result) >= 1

    def test_empty_df(self):
        df = pd.DataFrame(
            columns=["timestamp", "sentiment", "sentiment_score"]
        )
        result = aggregate_sentiment_over_time(df, freq="hourly")
        assert isinstance(result, pd.DataFrame)
        assert result.empty or result["post_count"].sum() == 0


# ---------------------------------------------------------------------------
# aggregate_by_dimension_over_time tests
# ---------------------------------------------------------------------------


class TestAggregateByDimensionOverTime:
    def test_returns_dataframe(self, multi_brand_df):
        result = aggregate_by_dimension_over_time(
            multi_brand_df, dimension="brand", freq="daily"
        )
        assert isinstance(result, pd.DataFrame)
        assert "brand" in result.columns
        assert "timestamp" in result.columns

    def test_known_brands_present(self, multi_brand_df):
        result = aggregate_by_dimension_over_time(
            multi_brand_df, dimension="brand", freq="daily"
        )
        brands = set(result["brand"].tolist())
        assert "ACME" in brands
        assert "TechCorp" in brands

    def test_invalid_dimension_raises(self, sample_df):
        with pytest.raises(ValueError, match="not found"):
            aggregate_by_dimension_over_time(sample_df, dimension="nonexistent_col")

    def test_counts_per_brand(self, multi_brand_df):
        result = aggregate_by_dimension_over_time(
            multi_brand_df, dimension="brand", freq="daily"
        )
        acme_total = result[result["brand"] == "ACME"]["post_count"].sum()
        tech_total = result[result["brand"] == "TechCorp"]["post_count"].sum()
        assert acme_total == 10
        assert tech_total == 10


# ---------------------------------------------------------------------------
# aggregate_by_entities tests
# ---------------------------------------------------------------------------


class TestAggregateByEntities:
    def test_returns_dict(self, multi_brand_df):
        result = aggregate_by_entities(multi_brand_df, freq="daily")
        assert isinstance(result, dict)

    def test_default_dimensions(self, multi_brand_df):
        result = aggregate_by_entities(multi_brand_df, freq="daily")
        assert "brand" in result
        assert "product" in result
        assert "topic" in result

    def test_custom_dimensions(self, multi_brand_df):
        result = aggregate_by_entities(
            multi_brand_df, dimensions=["brand", "platform"], freq="daily"
        )
        assert set(result.keys()) == {"brand", "platform"}

    def test_values_are_dataframes(self, multi_brand_df):
        result = aggregate_by_entities(multi_brand_df, freq="daily")
        for key, val in result.items():
            assert isinstance(val, pd.DataFrame), f"{key} should be a DataFrame"

    def test_missing_dimension_warns(self, sample_df):
        """A missing column should warn but not raise."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = aggregate_by_entities(
                sample_df, dimensions=["brand", "nonexistent"], freq="daily"
            )
        assert "nonexistent" in result
        assert len(w) >= 1
