"""
src/insights/__init__.py
------------------------
Phase 7 - Explainable AI Insight Layer Package.
"""

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

__all__ = [
    "SEVERITY_CRITICAL",
    "SEVERITY_HIGH",
    "SEVERITY_LOW",
    "SEVERITY_MEDIUM",
    "analyze_entity_concentration",
    "analyze_sentiment_trend",
    "calculate_severity",
    "correlate_spike_drivers",
    "format_executive_report",
    "format_insight_text",
    "format_trend_summary",
    "generate_insights",
    "priority_from_severity",
    "score_alert_severity",
]
