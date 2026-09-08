"""
src/time_series/__init__.py
---------------------------
Phase 6 - Time-Series Builder package.

Public API
----------
    build_time_series(df, freq="hourly")
    add_rolling_statistics(time_series, value_column, window=3)
"""

from .builder import (
    build_time_series,
    add_rolling_statistics,
)

__all__ = [
    "build_time_series",
    "add_rolling_statistics",
]
