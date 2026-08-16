"""Alert generation for detected sentiment spikes."""

from __future__ import annotations

from typing import Any


def _severity_from_z_score(z_score: float) -> str:
    if z_score >= 4:
        return "CRITICAL"
    if z_score >= 3:
        return "HIGH"
    return "MEDIUM"


def generate_alerts(spikes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert spike detections into structured alert records."""
    alerts: list[dict[str, Any]] = []
    for spike in spikes:
        z_score = float(spike["z_score"])
        alerts.append(
            {
                "alert_type": "NEGATIVE_SENTIMENT_SPIKE",
                "timestamp": spike["timestamp"],
                "severity": _severity_from_z_score(z_score),
                "observed_value": spike["observed_value"],
                "baseline_value": spike["baseline_value"],
                "threshold": spike["threshold"],
                "z_score": z_score,
                "message": (
                    "Negative sentiment has increased significantly above the "
                    "expected baseline."
                ),
            }
        )
    return alerts
