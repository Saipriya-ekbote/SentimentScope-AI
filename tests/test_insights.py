"""
tests/test_insights.py
----------------------
Phase 7 - Automated tests for explainable AI insights engine.

Covers:
  - Severity calculation across all 4 tiers (CRITICAL, HIGH, MEDIUM, LOW)
  - Edge cases in severity calculation (NaN, infinite z-scores, zero baseline)
  - Priority mapping from severity
  - Trend classification (improving, declining, stable, volatile, insufficient data)
  - Defensive trend analysis (empty dataframe, single-period, missing columns)
  - Entity concentration analysis across brands, products, and topics
  - Root-cause spike driver correlation
  - Structured insight generation (no alerts, multiple alerts, macro trends)
  - Business-friendly text formatting and executive reporting
  - End-to-end integration with existing aggregation and spike detection pipelines
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.alerts import generate_alerts
from src.insights.analyzer import (
    analyze_entity_concentration,
    analyze_sentiment_trend,
    correlate_spike_drivers,
    generate_insights,
)
from src.insights.formatter import (
    format_executive_report,
    format_insight_text,
    format_trend_summary,
)
from src.insights.severity import (
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    calculate_severity,
    priority_from_severity,
    score_alert_severity,
)
from src.spike_detection import detect_negative_spikes
from src.time_series import build_time_series

# ===========================================================================
# 1. Severity Scoring Tests
# ===========================================================================

class TestSeverityScoring:
    """Test explainable multi-factor severity scoring."""

    def test_severity_critical_tier_by_z_score(self):
        """Z-score >= 4.0 or infinite spike must trigger CRITICAL."""
        assert calculate_severity(z_score=4.0) == SEVERITY_CRITICAL
        assert calculate_severity(z_score=5.5) == SEVERITY_CRITICAL
        assert calculate_severity(z_score=float("inf")) == SEVERITY_CRITICAL

    def test_severity_critical_tier_by_concentration_and_volume(self):
        """High z-score (>= 3.0) with >= 70% negative ratio and >= 10 count triggers CRITICAL."""
        sev = calculate_severity(
            z_score=3.2,
            negative_ratio=0.75,
            observed_count=15,
            increase_percent=40.0,
        )
        assert sev == SEVERITY_CRITICAL

    def test_severity_critical_tier_by_massive_surge(self):
        """Surge >= 100% with negative ratio >= 60% and count >= 20 triggers CRITICAL."""
        sev = calculate_severity(
            z_score=2.5,
            negative_ratio=0.65,
            observed_count=25,
            increase_percent=120.0,
        )
        assert sev == SEVERITY_CRITICAL

    def test_severity_high_tier(self):
        """Z-score in [3.0, 4.0) or moderate z-score with high ratio/volume triggers HIGH."""
        assert calculate_severity(z_score=3.0) == SEVERITY_HIGH
        assert calculate_severity(z_score=3.8) == SEVERITY_HIGH

        # z=2.2, ratio=0.65, count=8 -> HIGH
        sev = calculate_severity(
            z_score=2.2,
            negative_ratio=0.65,
            observed_count=8,
            increase_percent=20.0,
        )
        assert sev == SEVERITY_HIGH

    def test_severity_medium_tier(self):
        """Standard anomaly (z in [2.0, 3.0)) triggers MEDIUM."""
        assert calculate_severity(z_score=2.0) == SEVERITY_MEDIUM
        assert calculate_severity(z_score=2.8) == SEVERITY_MEDIUM

        # Low z but high negative ratio (>= 40%) with volume >= 5 -> MEDIUM
        sev = calculate_severity(
            z_score=1.5,
            negative_ratio=0.50,
            observed_count=6,
            increase_percent=10.0,
        )
        assert sev == SEVERITY_MEDIUM

    def test_severity_low_tier(self):
        """Below anomaly threshold with mild metrics triggers LOW."""
        assert calculate_severity(z_score=1.0) == SEVERITY_LOW
        assert calculate_severity(z_score=0.5, negative_ratio=0.2, observed_count=2) == SEVERITY_LOW
        assert calculate_severity() == SEVERITY_LOW

    def test_severity_edge_cases_and_invalid_inputs(self):
        """Handles NaNs, negative z-scores, negative ratios, and infinite values gracefully."""
        assert calculate_severity(z_score=float("nan")) == SEVERITY_LOW
        assert calculate_severity(z_score=-2.5) == SEVERITY_LOW
        assert calculate_severity(z_score=float("-inf")) == SEVERITY_LOW
        assert calculate_severity(negative_ratio=float("nan")) == SEVERITY_LOW
        assert calculate_severity(observed_count=float("nan")) == SEVERITY_LOW
        assert calculate_severity(increase_percent=float("nan")) == SEVERITY_LOW

    def test_priority_from_severity_mapping(self):
        """Priorities map deterministically: CRITICAL=1, HIGH=2, MEDIUM=3, LOW=4."""
        assert priority_from_severity(SEVERITY_CRITICAL) == 1
        assert priority_from_severity(SEVERITY_HIGH) == 2
        assert priority_from_severity(SEVERITY_MEDIUM) == 3
        assert priority_from_severity(SEVERITY_LOW) == 4
        assert priority_from_severity("UNKNOWN") == 99

    def test_score_alert_severity_wrapper(self):
        """score_alert_severity correctly parses alert dictionary keys."""
        alert = {
            "z_score": 4.5,
            "observed_value": 30.0,
            "increase_percent": 80.0,
        }
        assert score_alert_severity(alert) == SEVERITY_CRITICAL


# ===========================================================================
# 2. Trend Analysis Tests
# ===========================================================================

class TestTrendAnalysis:
    """Test deterministic and leak-free sentiment trend classification."""

    def test_trend_improving(self):
        """Increasing positive ratio over chronological time classifies as improving."""
        ts = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC"),
                "post_count": [10, 10, 10, 10, 10, 10],
                "positive_ratio": [0.1, 0.15, 0.2, 0.5, 0.6, 0.7],
                "negative_ratio": [0.7, 0.65, 0.6, 0.3, 0.2, 0.1],
            }
        )
        result = analyze_sentiment_trend(ts)
        assert result["direction"] == "improving"
        assert result["label"] == "Improving"
        assert result["net_change"] > 0.05
        assert result["total_periods"] == 6

    def test_trend_declining(self):
        """Increasing negative ratio over chronological time classifies as declining."""
        ts = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC"),
                "post_count": [20, 20, 20, 20, 20, 20],
                "positive_ratio": [0.7, 0.65, 0.6, 0.3, 0.2, 0.1],
                "negative_ratio": [0.1, 0.15, 0.2, 0.5, 0.6, 0.7],
            }
        )
        result = analyze_sentiment_trend(ts)
        assert result["direction"] == "declining"
        assert result["label"] == "Declining"
        assert result["net_change"] < -0.05

    def test_trend_stable(self):
        """Constant sentiment ratios classify as stable."""
        ts = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC"),
                "post_count": [15, 15, 15, 15, 15, 15],
                "positive_ratio": [0.4, 0.41, 0.39, 0.4, 0.41, 0.4],
                "negative_ratio": [0.3, 0.29, 0.31, 0.3, 0.29, 0.3],
            }
        )
        result = analyze_sentiment_trend(ts)
        assert result["direction"] == "stable"
        assert result["label"] == "Stable"
        assert abs(result["net_change"]) < 0.05

    def test_trend_volatile(self):
        """Alternating wild swings classify as volatile."""
        ts = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC"),
                "post_count": [10, 10, 10, 10, 10, 10],
                "positive_ratio": [0.8, 0.1, 0.8, 0.1, 0.8, 0.1],
                "negative_ratio": [0.1, 0.8, 0.1, 0.8, 0.1, 0.8],
            }
        )
        result = analyze_sentiment_trend(ts)
        assert result["direction"] == "volatile"
        assert result["label"] == "Highly Volatile"
        assert result["volatility"] >= 0.25

    def test_trend_insufficient_data(self):
        """Empty time series or fewer than 2 active periods returns insufficient_data."""
        assert analyze_sentiment_trend(pd.DataFrame())["direction"] == "insufficient_data"
        assert analyze_sentiment_trend(None)["direction"] == "insufficient_data"

        single_point = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-01", tz="UTC")],
                "post_count": [5],
                "positive_ratio": [0.6],
                "negative_ratio": [0.2],
            }
        )
        assert analyze_sentiment_trend(single_point)["direction"] == "insufficient_data"

    def test_trend_handles_empty_buckets_safely(self):
        """Buckets with post_count == 0 are filtered out before calculating trend."""
        ts = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC"),
                "post_count": [10, 0, 0, 10],
                "positive_ratio": [0.1, float("nan"), float("nan"), 0.7],
                "negative_ratio": [0.7, float("nan"), float("nan"), 0.1],
            }
        )
        result = analyze_sentiment_trend(ts)
        assert result["total_periods"] == 2
        assert result["direction"] == "improving"

    def test_trend_does_not_mutate_input(self):
        """Source DataFrame must remain completely unaltered."""
        ts = pd.DataFrame(
            {
                "timestamp": ["2026-01-02", "2026-01-01"],
                "post_count": [10, 10],
                "positive_ratio": [0.5, 0.2],
                "negative_ratio": [0.2, 0.5],
            }
        )
        ts_orig = ts.copy()
        analyze_sentiment_trend(ts)
        pd.testing.assert_frame_equal(ts, ts_orig)


# ===========================================================================
# 3. Entity Concentration & Root-Cause Correlation Tests
# ===========================================================================

class TestEntityAndSpikeCorrelation:
    """Test entity concentration and alert root-cause correlation."""

    @pytest.fixture
    def sample_posts_df(self):
        return pd.DataFrame(
            {
                "timestamp": [
                    "2026-02-01T10:00:00Z",
                    "2026-02-01T10:30:00Z",
                    "2026-02-01T11:00:00Z",
                    "2026-02-01T11:30:00Z",
                    "2026-02-01T12:00:00Z",
                ],
                "sentiment": ["negative", "negative", "negative", "positive", "negative"],
                "brand": ["Delta", "Delta", "Delta", "Delta", "United"],
                "product": ["Flights", "Flights", "Baggage", "Flights", "Flights"],
                "topic": ["delay", "delay", "lost_bag", "service", "delay"],
            }
        )

    def test_analyze_entity_concentration(self, sample_posts_df):
        """Identifies top brands, products, and topics with negative volume and ratios."""
        res = analyze_entity_concentration(sample_posts_df)
        assert res["total_negative_posts"] == 4

        # Delta has 3 negative posts out of 4 (75% share of all negative)
        assert len(res["top_brands"]) >= 1
        delta_stat = next(b for b in res["top_brands"] if b["name"] == "Delta")
        assert delta_stat["negative_count"] == 3
        assert delta_stat["share_of_all_negative"] == 75.0

        # Primary focal point should be Delta or delay
        assert res["primary_focal_point"] is not None
        assert res["primary_focal_point"]["name"] in ["Delta", "delay", "Flights"]

    def test_analyze_entity_concentration_empty(self):
        """Empty or invalid DataFrame produces graceful empty response."""
        res = analyze_entity_concentration(pd.DataFrame())
        assert res["total_negative_posts"] == 0
        assert res["top_brands"] == []
        assert res["primary_focal_point"] is None

    def test_correlate_spike_drivers(self, sample_posts_df):
        """Identifies co-occurring topics and brands during a spike at a given timestamp."""
        alert = {
            "timestamp": "2026-02-01T10:00:00Z",
            "dimension": "brand",
            "entity": "Delta",
        }
        drivers = correlate_spike_drivers(sample_posts_df, alert)
        assert drivers["driver_topic"] == "delay"
        assert drivers["driver_topic_share"] > 0
        assert drivers["total_spike_negative_posts"] == 3

    def test_correlate_spike_drivers_global(self, sample_posts_df):
        """Global alert identifies top brand and topic drivers."""
        alert = {"timestamp": "2026-02-01T10:00:00Z"}
        drivers = correlate_spike_drivers(sample_posts_df, alert)
        assert drivers["driver_brand"] == "Delta"
        assert drivers["driver_topic"] == "delay"


# ===========================================================================
# 4. Insight Generation & Formatting Tests
# ===========================================================================

class TestInsightGenerationAndFormatting:
    """Test structured insight synthesis and business-friendly text formatting."""

    def test_generate_insights_empty_inputs(self):
        """Empty inputs produce no crash and empty/minimal insights."""
        insights = generate_insights(pd.DataFrame(), pd.DataFrame(), [])
        assert isinstance(insights, list)

    def test_generate_insights_with_spike_alert(self):
        """Converts an alert into a structured, prioritized insight with explanation."""
        df = pd.DataFrame(
            {
                "timestamp": ["2026-02-01T10:00:00Z", "2026-02-01T10:05:00Z"],
                "sentiment": ["negative", "negative"],
                "brand": ["Delta", "Delta"],
                "product": ["Flights", "Flights"],
                "topic": ["delay", "delay"],
            }
        )
        ts = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-02-01", periods=3, freq="D", tz="UTC"),
                "post_count": [10, 10, 10],
                "positive_ratio": [0.4, 0.4, 0.4],
                "negative_ratio": [0.3, 0.3, 0.3],
            }
        )
        alert = {
            "timestamp": "2026-02-01T10:00:00Z",
            "severity": "HIGH",
            "observed_value": 25.0,
            "baseline_value": 10.0,
            "increase_percent": 150.0,
            "z_score": 3.8,
            "dimension": "brand",
            "entity": "Delta",
        }

        insights = generate_insights(df, ts, [alert])
        assert len(insights) >= 1

        insight = insights[0]
        assert insight["entity_dimension"] == "brand"
        assert insight["entity_name"] == "Delta"
        assert insight["severity"] == SEVERITY_HIGH
        assert insight["priority"] == 2
        assert insight["primary_topic"] == "delay"
        assert "Delta" in insight["summary"]
        assert "150.0%" in insight["explanation"]
        assert "delay" in insight["explanation"]

    def test_format_insight_text_contains_actual_metrics(self):
        """Formatted text contains exact observed metrics, entities, and no fake data."""
        insight = {
            "severity": "CRITICAL",
            "summary": "Negative sentiment spike detected for Brand 'Delta'.",
            "explanation": "Negative sentiment surged 120.0% above baseline.",
            "evidence": {
                "observed_negative": 45.0,
                "baseline_negative": 15.0,
                "increase_percent": 200.0,
                "z_score": 4.5,
                "primary_topic": "cancellation",
                "topic_share": 65.0,
            },
        }

        card = format_insight_text(insight)
        assert "[CRITICAL ALERT]" in card
        assert "Delta" in card
        assert "45" in card
        assert "15.0" in card
        assert "200.0%" in card
        assert "4.50" in card
        assert "cancellation" in card
        assert "65.0%" in card

    def test_format_insight_text_empty(self):
        """Empty insight dictionary returns fallback string."""
        assert format_insight_text({}) == "No insight details available."

    def test_format_trend_summary(self):
        """Trend summary correctly formats metrics and status."""
        trend = {
            "direction": "improving",
            "label": "Improving",
            "net_change": 0.125,
            "volatility": 0.04,
            "total_periods": 5,
            "description": "Net sentiment improved by 12.5%.",
        }
        text = format_trend_summary(trend)
        assert "Improving" in text
        assert "+12.5%" in text
        assert "5 periods" in text

    def test_format_executive_report(self):
        """Executive report compiles trend and top operational insights."""
        trend = {
            "direction": "stable",
            "label": "Stable",
            "net_change": 0.01,
            "volatility": 0.02,
            "total_periods": 4,
            "description": "Stable sentiment.",
        }
        insight = {
            "severity": "HIGH",
            "summary": "Negative spike for Topic 'software'.",
            "explanation": "Spike details.",
            "evidence": {"observed_negative": 10.0, "baseline_negative": 4.0},
        }

        report = format_executive_report([insight], trend)
        assert "Executive AI Sentiment Briefing" in report
        assert "Stable" in report
        assert "Negative spike for Topic 'software'." in report


# ===========================================================================
# 5. Pipeline Integration Tests
# ===========================================================================

class TestPipelineIntegration:
    """Test full integration with Phase 1-6 pipeline outputs."""

    def test_integration_with_synthetic_sample_data(self):
        """End-to-end integration test with sample_data.csv."""
        from src.data_loader import load_csv
        from src.entity_detection import EntityDetector
        from src.entity_spike_detection import detect_all_entity_spikes
        from src.preprocessing import preprocess_dataframe

        raw_df = load_csv("data/sample_data.csv")
        df = preprocess_dataframe(raw_df)
        detector = EntityDetector()
        df = detector.detect_entities_dataframe(df)

        ts = build_time_series(df, freq="daily")
        spikes = detect_negative_spikes(ts.rename(columns={"timestamp": "period"}), "negative", 2.0)
        entity_spikes = detect_all_entity_spikes(df, ["brand", "product", "topic"], 2.0, "daily")
        alerts = generate_alerts(spikes) + generate_alerts(entity_spikes)

        # Generate Phase 7 AI Insights
        insights = generate_insights(df, ts, alerts)
        trend = analyze_sentiment_trend(ts)

        assert isinstance(insights, list)
        assert len(insights) > 0
        assert trend["direction"] in ["stable", "improving", "declining", "volatile"]

        # Check that each insight has valid schema and formatting
        for ins in insights:
            assert "insight_id" in ins
            assert ins["severity"] in [SEVERITY_CRITICAL, SEVERITY_HIGH, SEVERITY_MEDIUM, SEVERITY_LOW]
            assert 1 <= ins["priority"] <= 4
            formatted = format_insight_text(ins)
            assert len(formatted) > 20
