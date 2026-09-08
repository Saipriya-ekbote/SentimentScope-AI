"""
src/time_series.py
------------------
Backward-compatibility facade for Phase 6.

All imports go through the modular src/time_series/ package.
Do NOT add logic here; keep this file as a thin re-export.
"""

from src.time_series.builder import (  # noqa: F401
    build_time_series,
    add_rolling_statistics,
)

__all__ = [
    "build_time_series",
    "add_rolling_statistics",
]
