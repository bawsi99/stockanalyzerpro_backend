#!/usr/bin/env python3
"""
Risk Analysis Agents Package

Comprehensive risk analysis system with specialized agents for different risk categories:
- Market Risk: Systemic risks, correlation breakdowns, regime changes
- Volatility Risk: Volatility clustering, regime changes, spillovers
- Liquidity Risk: Execution risks, market depth, bid-ask spreads
- Technical Risk: Signal reliability, indicator divergences, pattern failures

The orchestrator coordinates all agents for unified risk assessment.
"""

# Import individual risk agents
from .market_risk import MarketRiskProcessor
from .volatility_risk import VolatilityRiskProcessor
from .liquidity_risk import LiquidityRiskProcessor
from .technical_risk import TechnicalRiskProcessor
# Optional synthesis agent (LLM wrapper)
try:  # pragma: no cover - optional import
    from .synthesis import RiskSynthesisProcessor  # type: ignore
except Exception:  # noqa: E722
    RiskSynthesisProcessor = None

# Import orchestrator and result classes
from .orchestrator import (
    RiskAgentsOrchestrator,
    RiskAgentResult,
    RiskAnalysisResult,
    risk_orchestrator
)

__all__ = [
    # Individual Risk Agents
    'MarketRiskProcessor',
    'VolatilityRiskProcessor', 
    'LiquidityRiskProcessor',
    'TechnicalRiskProcessor',
    'RiskSynthesisProcessor',
    
    # Orchestrator and Results
    'RiskAgentsOrchestrator',
    'RiskAgentResult',
    'RiskAnalysisResult',
    'risk_orchestrator'
]
