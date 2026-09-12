"""
src/insights.py
---------------
Backward-compatibility facade for Phase 7.

All imports go through the modular `src.insights` package.
Do NOT add logic here; keep this file as a thin re-export.
"""

from src.insights import (
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    analyze_entity_concentration,
    analyze_sentiment_trend,
    calculate_severity,
    correlate_spike_drivers,
    format_executive_report,
    format_insight_text,
    format_trend_summary,
    generate_insights,
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
