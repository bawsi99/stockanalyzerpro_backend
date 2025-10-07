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
from .risk_llm_agent import RiskLLMAgent, get_risk_llm_agent

# Lazy initialization for backwards compatibility
def _get_risk_llm_agent():
    return get_risk_llm_agent()

# Create a property that returns the agent when accessed
class _RiskAgentProperty:
    def __getattr__(self, name):
        if name == 'risk_llm_agent':
            return get_risk_llm_agent()
        raise AttributeError(f"module has no attribute '{name}'")

risk_llm_agent = _get_risk_llm_agent  # Function to get agent

__all__ = [
    # Core Risk Analysis Components
    'QuantitativeRiskProcessor',
    'RiskLLMAgent',
    'get_risk_llm_agent',
    'risk_llm_agent'
]
