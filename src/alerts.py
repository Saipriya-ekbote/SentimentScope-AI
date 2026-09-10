"""Alert generation for detected sentiment spikes."""

from __future__ import annotations

from typing import Any


def _severity_from_z_score(z_score: float) -> str:
    """Convert a z-score into an alert severity."""
    if z_score >= 4:
        return "CRITICAL"
    if z_score >= 3:
        return "HIGH"
    return "MEDIUM"


def _priority_from_severity(severity: str) -> int:
    """Convert alert severity into a numeric priority."""
    priorities = {
        "CRITICAL": 1,
        "HIGH": 2,
        "MEDIUM": 3,
    }

    return priorities.get(severity, 99)


def _calculate_increase_percent(
    observed_value: float,
    baseline_value: float,
) -> float:
    """Calculate percentage increase above the baseline."""
    if baseline_value <= 0:
        return 0.0

    return ((observed_value - baseline_value) / baseline_value) * 100


def _entity_label(dimension: str) -> str:
    """Convert an entity dimension into a human-readable label."""
    return dimension.replace("_", " ").title()


def _deduplicate_alerts(
    alerts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove duplicate alerts while preserving alert order."""
    unique_alerts: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    for alert in alerts:
        key = (
            alert["timestamp"],
            alert["alert_type"],
            alert.get("dimension"),
            alert.get("entity"),
        )

        if key in seen:
            continue

        seen.add(key)
        unique_alerts.append(alert)

    return unique_alerts


def generate_alerts(
    spikes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Convert spike detections into structured alert records.

    Supports:
    - Global spikes from Phase 6.1.
    - Entity-specific spikes from Phase 6.2.
    - Severity and priority from Phase 6.3.
    - Percentage increase from Phase 6.3.
    - Alert deduplication from Phase 6.3.
    """

    alerts: list[dict[str, Any]] = []

    for spike in spikes:
        z_score = float(spike["z_score"])
        observed_value = float(spike["observed_value"])
        baseline_value = float(spike["baseline_value"])

        severity = _severity_from_z_score(z_score)
        priority = _priority_from_severity(severity)

        increase_percent = _calculate_increase_percent(
            observed_value,
            baseline_value,
        )

        alert = {
            "alert_type": "NEGATIVE_SENTIMENT_SPIKE",
            "timestamp": spike["timestamp"],
            "severity": severity,
            "priority": priority,
            "observed_value": observed_value,
            "baseline_value": baseline_value,
            "increase_percent": increase_percent,
            "threshold": spike["threshold"],
            "z_score": z_score,
        }

        # Add entity information when this is an entity-specific spike.
        if "entity" in spike and "dimension" in spike:
            dimension = str(spike["dimension"])
            entity = str(spike["entity"])
            label = _entity_label(dimension)

            alert["dimension"] = dimension
            alert["entity"] = entity

            alert["message"] = (
                f"Negative sentiment for {label} '{entity}' has increased "
                f"{increase_percent:.1f}% above the expected baseline."
            )

        else:
            # Preserve Phase 6.1 global alert behavior.
            alert["message"] = (
                "Negative sentiment has increased "
                f"{increase_percent:.1f}% above the expected baseline."
            )

        alerts.append(alert)

    # Remove duplicate alerts.
    alerts = _deduplicate_alerts(alerts)

    # Highest-priority alerts appear first.
    alerts.sort(
        key=lambda alert: (
            alert["priority"],
            alert["timestamp"],
        )
    )

    return alerts