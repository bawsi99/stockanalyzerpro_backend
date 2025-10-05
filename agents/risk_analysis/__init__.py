#!/usr/bin/env python3
"""
Risk Analysis Agents Package

Modern risk analysis system with:
- Quantitative Risk Processor: Advanced metrics, stress testing, scenario analysis
- Risk LLM Agent: Natural language risk synthesis and insights

Integrated with the analysis service for comprehensive risk assessment.
"""

# Import core risk analysis components
from .quantitative_risk.processor import QuantitativeRiskProcessor
from .risk_llm_agent import RiskLLMAgent, risk_llm_agent

__all__ = [
    # Core Risk Analysis Components
    'QuantitativeRiskProcessor',
    'RiskLLMAgent',
    'risk_llm_agent'
]
