"""
tests/test_time_series.py
-------------------------
Phase 6 tests for src/time_series/ package.
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.time_series import build_time_series, add_rolling_statistics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_sparse_df():
    """Two posts separated by a 3-hour gap (no posts at hours 11, 12)."""
    return pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-01-01 10:00:00", tz="UTC"),
                pd.Timestamp("2024-01-01 13:00:00", tz="UTC"),
            ],
            "sentiment": ["positive", "negative"],
            "sentiment_score": [0.8, -0.7],
        }
    )


def _make_dense_df():
    """5 consecutive hourly posts."""
    base = pd.Timestamp("2024-01-01 09:00:00", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": [base + pd.Timedelta(hours=i) for i in range(5)],
            "sentiment": ["positive", "positive", "neutral", "negative", "positive"],
            "sentiment_score": [0.9, 0.7, 0.1, -0.5, 0.6],
        }
    )


# ---------------------------------------------------------------------------
# build_time_series tests
# ---------------------------------------------------------------------------


class TestBuildTimeSeries:
    def test_returns_dataframe(self):
        result = build_time_series(_make_dense_df(), freq="hourly")
        assert isinstance(result, pd.DataFrame)

    def test_timestamp_column_present(self):
        result = build_time_series(_make_dense_df(), freq="hourly")
        assert "timestamp" in result.columns

    def test_no_gaps_in_dense(self):
        """Dense data should produce exactly 5 rows, no gaps."""
        result = build_time_series(_make_dense_df(), freq="hourly")
        assert len(result) == 5

    def test_gaps_filled(self):
        """Sparse data should have gap rows with count=0."""
        result = build_time_series(_make_sparse_df(), freq="hourly")
        # hours 10, 11, 12, 13 = 4 rows
        assert len(result) == 4

    def test_gap_rows_have_zero_count(self):
        result = build_time_series(_make_sparse_df(), freq="hourly")
        empty = result[result["post_count"] == 0]
        assert len(empty) == 2  # hours 11 and 12

    def test_gap_rows_avg_score_is_nan(self):
        """Empty buckets must have NaN average_sentiment_score, not zero."""
        result = build_time_series(_make_sparse_df(), freq="hourly")
        empty = result[result["post_count"] == 0]
        assert empty["average_sentiment_score"].isna().all()

    def test_total_post_count(self):
        result = build_time_series(_make_sparse_df(), freq="hourly")
        assert result["post_count"].sum() == 2

    def test_sorted_by_timestamp(self):
        result = build_time_series(_make_dense_df(), freq="hourly")
        ts = pd.to_datetime(result["timestamp"])
        assert ts.is_monotonic_increasing

    def test_daily_freq(self):
        result = build_time_series(_make_dense_df(), freq="daily")
        # All 5 posts within 5 hours on same day -> 1 row
        assert len(result) == 1
        assert result.iloc[0]["post_count"] == 5

    def test_empty_df(self):
        df = pd.DataFrame(
            columns=["timestamp", "sentiment", "sentiment_score"]
        )
        result = build_time_series(df, freq="hourly")
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# add_rolling_statistics tests
# ---------------------------------------------------------------------------


class TestAddRollingStatistics:
    @pytest.fixture
    def ts(self):
        """A simple 6-row time-series for rolling tests."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01", periods=6, freq="h", tz="UTC"
                ),
                "post_count": [1, 2, 3, 0, 4, 5],
                "average_sentiment_score": [0.5, 0.3, 0.8, float("nan"), 0.2, 0.6],
            }
        )

    def test_new_columns_added(self, ts):
        result = add_rolling_statistics(ts, "post_count", window=3)
        assert "post_count_rolling_mean" in result.columns
        assert "post_count_rolling_std" in result.columns

    def test_original_not_modified(self, ts):
        original_cols = list(ts.columns)
        _ = add_rolling_statistics(ts, "post_count", window=3)
        assert list(ts.columns) == original_cols

    def test_rolling_mean_is_causal(self, ts):
        result = add_rolling_statistics(ts, "post_count", window=3)
        means = result["post_count_rolling_mean"].tolist()
        # Row 0: window=[1]         -> mean=1.0
        # Row 1: window=[1,2]       -> mean=1.5
        # Row 2: window=[1,2,3]     -> mean=2.0
        assert math.isclose(means[0], 1.0, rel_tol=1e-9)
        assert math.isclose(means[1], 1.5, rel_tol=1e-9)
        assert math.isclose(means[2], 2.0, rel_tol=1e-9)

    def test_rolling_mean_future_not_used(self, ts):
        """Value at t should NOT include data from t+1 or later."""
        result = add_rolling_statistics(ts, "post_count", window=3)
        means = result["post_count_rolling_mean"].tolist()
        # Row 2 uses rows 0-2 (values 1,2,3) -> mean=2.0
        # If future data were used it would include row 3 (0), giving 1.5
        assert math.isclose(means[2], 2.0, rel_tol=1e-9)

    def test_window_1_mean_equals_original(self, ts):
        result = add_rolling_statistics(ts, "post_count", window=1)
        means = result["post_count_rolling_mean"].tolist()
        expected = [float(v) for v in ts["post_count"].tolist()]
        for m, e in zip(means, expected):
            assert math.isclose(m, e, rel_tol=1e-9)

    def test_std_first_row_is_nan(self, ts):
        """With window=3, std of single element should be NaN (ddof=1)."""
        result = add_rolling_statistics(ts, "post_count", window=3)
        # Row 0 has min_periods=1 so mean is computed, but std needs >= 2 points
        assert math.isnan(result["post_count_rolling_std"].iloc[0])

    def test_nan_value_column_propagated(self, ts):
        """NaN in source column should not crash and should appear in output."""
        result = add_rolling_statistics(ts, "average_sentiment_score", window=2)
        assert "average_sentiment_score_rolling_mean" in result.columns

    def test_invalid_column_raises(self, ts):
        with pytest.raises(ValueError, match="not found"):
            add_rolling_statistics(ts, "nonexistent_column")

    def test_window_zero_raises(self, ts):
        with pytest.raises(ValueError, match="window must be >= 1"):
            add_rolling_statistics(ts, "post_count", window=0)



def test_facade_import():
    """Time series functions must be importable via facade."""
    from src.time_series import build_time_series, add_rolling_statistics
    assert callable(build_time_series)
    assert callable(add_rolling_statistics)
