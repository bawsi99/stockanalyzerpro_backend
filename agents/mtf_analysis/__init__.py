#!/usr/bin/env python3
"""
MTF Agents package exports
"""

from .orchestrator import MTFOrchestrator
from .intraday import IntradayMTFProcessor

# Optional processors (may not exist depending on deployment stage)
try:  # pragma: no cover - optional import
    from .swing import SwingMTFProcessor  # type: ignore
except Exception:  # noqa: E722
    SwingMTFProcessor = None

try:  # pragma: no cover - optional import
    from .position import PositionMTFProcessor  # type: ignore
except Exception:  # noqa: E722
    PositionMTFProcessor = None

__all__ = [
    "MTFOrchestrator",
    "IntradayMTFProcessor",
    "SwingMTFProcessor",
    "PositionMTFProcessor",
]
