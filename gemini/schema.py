from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional


class EntryStrategy(BaseModel):
    type: str = Field(default="breakout")
    entry_range: Optional[List[float]] = None
    entry_conditions: Optional[List[str]] = None
    confidence: Optional[float] = None


class TimeframeStrategy(BaseModel):
    horizon_days: Optional[int] = None
    bias: Optional[str] = None
    entry_strategy: Optional[EntryStrategy] = None
    exit_strategy: Optional[dict] = None
    position_sizing: Optional[dict] = None
    rationale: Optional[str] = None


class MarketOutlook(BaseModel):
    primary_trend: dict = Field(default_factory=dict)
    secondary_trend: Optional[dict] = None
    key_drivers: Optional[List[dict]] = None


class RiskManagement(BaseModel):
    key_risks: Optional[List[dict]] = None
    stop_loss_levels: Optional[List[dict]] = None
    position_management: Optional[dict] = None


class AIAnalysisSchema(BaseModel):
    trend: Optional[str] = None
    confidence_pct: Optional[float] = None
    market_outlook: Optional[MarketOutlook] = None
    trading_strategy: Optional[dict] = None
    risk_management: Optional[RiskManagement] = None
    critical_levels: Optional[dict] = None
    monitoring_plan: Optional[dict] = None
    data_quality_assessment: Optional[dict] = None
    key_takeaways: Optional[List[str]] = None
    indicator_summary: Optional[str] = None
    chart_insights: Optional[str] = None


def coerce_ai_analysis(payload: dict) -> AIAnalysisSchema:
    """Coerce arbitrary LLM JSON to the AIAnalysisSchema or raise ValidationError."""
    return AIAnalysisSchema.model_validate(payload)


