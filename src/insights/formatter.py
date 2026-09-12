"""
src/insights/formatter.py
-------------------------
Phase 7 - Business-Friendly Insight Explanations & Formatters.

Converts structured insight dictionaries and statistical evidence into clear,
concise, and executive-ready human explanations without hallucinating or
hard-coding synthetic metrics.
"""

from __future__ import annotations

from typing import Any


def format_insight_text(insight: dict[str, Any]) -> str:
    """Format a single structured insight into a concise, business-friendly card.

    Parameters
    ----------
    insight : dict[str, Any]
        Structured insight dictionary containing 'severity', 'summary',
        'evidence', 'explanation', 'entity_name', 'entity_dimension'.

    Returns
    -------
    str
        Formatted multi-line text or markdown.
    """
    if not insight:
        return "No insight details available."

    severity = str(insight.get("severity", "MEDIUM")).upper()
    summary = insight.get("summary", "Unusual sentiment pattern detected.")
    explanation = insight.get("explanation", "")
    evidence = insight.get("evidence", {})

    lines: list[str] = [
        f"[{severity} ALERT]",
        summary,
        "",
        "Evidence:",
    ]

    # Build evidence bullets based on actual empirical metrics present
    if "observed_negative" in evidence:
        obs = evidence["observed_negative"]
        base = evidence.get("baseline_negative", 0.0)
        lines.append(f"• Negative post volume: {obs:.0f} (Baseline: {base:.1f})")

    if "increase_percent" in evidence:
        inc = evidence["increase_percent"]
        lines.append(f"• Surge above baseline: +{inc:.1f}%")

    if "z_score" in evidence:
        z = evidence["z_score"]
        lines.append(f"• Anomaly significance: Z-score {z:.2f}")

    if evidence.get("primary_topic"):
        top = evidence["primary_topic"]
        share = evidence.get("topic_share", 0.0)
        share_str = f" ({share:.1f}% of spike negative posts)" if share > 0 else ""
        lines.append(f"• Primary co-occurring topic: {top}{share_str}")

    if evidence.get("co_occurring_brand"):
        br = evidence["co_occurring_brand"]
        br_share = evidence.get("co_occurring_brand_share", 0.0)
        share_str = f" ({br_share:.1f}% of negative volume)" if br_share > 0 else ""
        lines.append(f"• Associated brand: {br}{share_str}")

    if "share_of_all_negative" in evidence:
        lines.append(f"• Negative post concentration: {evidence['share_of_all_negative']:.1f}% of all complaints")

    if "net_change" in evidence:
        lines.append(f"• Net sentiment shift: {evidence['net_change'] * 100:+.1f}%")

    lines.append(f"• Severity tier: {severity}")

    if explanation:
        lines.extend(["", "Interpretation:", explanation])

    return "\n".join(lines)


def format_trend_summary(trend_result: dict[str, Any]) -> str:
    """Format macro trend classification into an executive summary paragraph.

    Parameters
    ----------
    trend_result : dict[str, Any]
        Output from `analyze_sentiment_trend`.

    Returns
    -------
    str
        Human-readable summary of the overall sentiment trajectory.
    """
    if not trend_result:
        return "Sentiment trend data is unavailable."

    direction = trend_result.get("direction", "insufficient_data")
    label = trend_result.get("label", "Unknown")
    net_change = trend_result.get("net_change", 0.0)
    volatility = trend_result.get("volatility", 0.0)
    periods = trend_result.get("total_periods", 0)

    if direction == "insufficient_data":
        return f"Trend Status: {label}. Insufficient temporal periods ({periods}) to establish a trajectory."

    lines = [
        f"**Overall Sentiment Trajectory: {label}**",
        f"- Net Sentiment Change: {net_change * 100:+.1f}%",
        f"- Period-to-Period Volatility: σ = {volatility:.2f}",
        f"- Time Intervals Analyzed: {periods} periods",
        "",
        trend_result.get("description", ""),
    ]

    return "\n".join(lines)


def format_executive_report(
    insights: list[dict[str, Any]],
    trend_result: dict[str, Any],
) -> str:
    """Compile an executive management report across all detected insights and trends.

    Parameters
    ----------
    insights : list[dict[str, Any]]
        List of structured insights.
    trend_result : dict[str, Any]
        Macro trend result dictionary.

    Returns
    -------
    str
        Markdown formatted executive report.
    """
    sections = [
        "## Executive AI Sentiment Briefing",
        "",
        format_trend_summary(trend_result),
        "",
        "### Key Operational Insights",
    ]

    if not insights:
        sections.append("No critical sentiment anomalies or concentrated risks detected.")
        return "\n".join(sections)

    for i, ins in enumerate(insights[:5], start=1):
        sections.append(f"#### {i}. {ins.get('summary', 'Alert')}")
        sections.append(format_insight_text(ins))
        sections.append("")

    return "\n".join(sections)
