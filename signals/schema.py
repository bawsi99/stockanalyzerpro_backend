from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

Bias = Literal["bullish", "bearish", "neutral"]


@dataclass
class SignalReason:
    indicator: str
    description: str
    weight: float
    bias: Bias


@dataclass
class TimeframeScore:
    timeframe: str
    score: float  # normalized in [-1, 1]
    confidence: float  # [0, 1]
    bias: Bias
    reasons: List[SignalReason] = field(default_factory=list)


@dataclass
class SignalsSummary:
    consensus_score: float  # [-1, 1]
    consensus_bias: Bias
    confidence: float  # [0, 1]
    per_timeframe: List[TimeframeScore] = field(default_factory=list)
    regime: Optional[Dict[str, str]] = None


