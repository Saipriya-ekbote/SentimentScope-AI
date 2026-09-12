"""
src/insights/severity.py
------------------------
Phase 7 - Multi-factor explainable severity scoring.

Provides deterministic, rule-based severity grading (CRITICAL, HIGH, MEDIUM, LOW)
for sentiment spikes, entity anomalies, and trend deterioration based on
measurable statistical evidence:
  - Spike z-score / magnitude
  - Negative sentiment ratio
  - Observed post volume
  - Relative increase percentage over baseline
"""

from __future__ import annotations

import math
from typing import Any

SEVERITY_CRITICAL = "CRITICAL"
SEVERITY_HIGH = "HIGH"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_LOW = "LOW"

VALID_SEVERITIES = (
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_MEDIUM,
    SEVERITY_LOW,
)

PRIORITY_MAP: dict[str, int] = {
    SEVERITY_CRITICAL: 1,
    SEVERITY_HIGH: 2,
    SEVERITY_MEDIUM: 3,
    SEVERITY_LOW: 4,
}


def priority_from_severity(severity: str) -> int:
    """Convert an insight severity string into a numerical priority (1-4).

    Parameters
    ----------
    severity : str
        One of 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'.

    Returns
    -------
    int
        1 for CRITICAL, 2 for HIGH, 3 for MEDIUM, 4 for LOW, 99 for unknown.
    """
    return PRIORITY_MAP.get(str(severity).upper(), 99)


def calculate_severity(
    z_score: float = 0.0,
    negative_ratio: float = 0.0,
    observed_count: float = 0.0,
    increase_percent: float = 0.0,
) -> str:
    """Calculate an explainable severity level based on empirical metrics.

    Parameters
    ----------
    z_score : float
        Statistical anomaly z-score relative to the rolling baseline.
    negative_ratio : float
        Proportion of negative posts (0.0 to 1.0).
    observed_count : float
        Volume of observed negative posts.
    increase_percent : float
        Percentage increase above baseline (e.g. 50.0 for +50%).

    Returns
    -------
    str
        One of 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'.
    """
    # Sanitize inputs
    z = 0.0 if (math.isnan(z_score) or (math.isinf(z_score) and z_score < 0)) else z_score
    is_inf = math.isinf(z) and z > 0
    neg_ratio = max(0.0, min(1.0, 0.0 if math.isnan(negative_ratio) else float(negative_ratio)))
    count = max(0.0, 0.0 if math.isnan(observed_count) else float(observed_count))
    inc_pct = max(0.0, 0.0 if math.isnan(increase_percent) else float(increase_percent))

    # CRITICAL:
    # 1. Extreme z-score (z >= 4.0 or infinite spike)
    # 2. High z-score (z >= 3.0) with very heavy negative concentration (>= 70%) and volume (>= 10)
    # 3. Massive volume surge (>= 100% increase) with strong negative concentration (>= 60%) and high count (>= 20)
    if is_inf or z >= 4.0:
        return SEVERITY_CRITICAL
    if z >= 3.0 and neg_ratio >= 0.70 and count >= 10:
        return SEVERITY_CRITICAL
    if inc_pct >= 100.0 and neg_ratio >= 0.60 and count >= 20:
        return SEVERITY_CRITICAL

    # HIGH:
    # 1. Significant z-score (z in [3.0, 4.0))
    # 2. Moderate z-score (z >= 2.0) with high negative ratio (>= 60%) and volume (>= 5)
    # 3. Noticeable surge (>= 50% increase) with predominantly negative posts (>= 50%) and volume (>= 5)
    if z >= 3.0:
        return SEVERITY_HIGH
    if z >= 2.0 and neg_ratio >= 0.60 and count >= 5:
        return SEVERITY_HIGH
    if inc_pct >= 50.0 and neg_ratio >= 0.50 and count >= 5:
        return SEVERITY_HIGH

    # MEDIUM:
    # 1. Standard anomaly threshold (z in [2.0, 3.0))
    # 2. Notable negative concentration (>= 40%) with meaningful volume (>= 5)
    # 3. Moderate surge (>= 25%) with negative ratio >= 40%
    if z >= 2.0:
        return SEVERITY_MEDIUM
    if neg_ratio >= 0.40 and count >= 5:
        return SEVERITY_MEDIUM
    if inc_pct >= 25.0 and neg_ratio >= 0.40:
        return SEVERITY_MEDIUM

    # LOW:
    # Minor fluctuations, sub-threshold volume, or predominantly neutral/positive activity
    return SEVERITY_LOW


def score_alert_severity(alert: dict[str, Any], negative_ratio: float = 0.0) -> str:
    """Score the multi-factor severity of an existing alert record.

    Parameters
    ----------
    alert : dict[str, Any]
        Alert dictionary containing 'z_score', 'observed_value', 'increase_percent'.
    negative_ratio : float, optional
        Observed negative sentiment ratio if available, default 0.0.

    Returns
    -------
    str
        One of 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'.
    """
    z_score = float(alert.get("z_score", 0.0))
    observed = float(alert.get("observed_value", 0.0))
    increase = float(alert.get("increase_percent", 0.0))

    return calculate_severity(
        z_score=z_score,
        negative_ratio=negative_ratio,
        observed_count=observed,
        increase_percent=increase,
    )
