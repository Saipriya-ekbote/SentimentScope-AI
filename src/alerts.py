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


def _entity_label(dimension: str) -> str:
    """Convert an entity dimension into a human-readable label."""
    return dimension.replace("_", " ").title()


def generate_alerts(spikes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert spike detections into structured alert records.

    Supports both:
    - Global spikes from Phase 6.1.
    - Entity-specific spikes from Phase 6.2.
    """

    alerts: list[dict[str, Any]] = []

    for spike in spikes:
        z_score = float(spike["z_score"])

        alert = {
            "alert_type": "NEGATIVE_SENTIMENT_SPIKE",
            "timestamp": spike["timestamp"],
            "severity": _severity_from_z_score(z_score),
            "observed_value": spike["observed_value"],
            "baseline_value": spike["baseline_value"],
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
                "significantly above the expected baseline."
            )
        else:
            # Preserve Phase 6.1 global alert behavior.
            alert["message"] = (
                "Negative sentiment has increased significantly above the "
                "expected baseline."
            )

        alerts.append(alert)

    return alerts