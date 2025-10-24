"""
Backend Agents Module

This module contains all AI agents used for specialized analysis tasks.
Currently includes distributed volume analysis agents, pattern analysis agents,
indicator analysis agents, and risk analysis agents.

Agents:
- volume_anomaly: Detects unusual volume spikes and patterns
- institutional_activity: Analyzes institutional trading patterns  
- volume_confirmation: Confirms price movements with volume analysis
- support_resistance: Volume-based support/resistance analysis
- volume_momentum: Volume trend and momentum analysis
- pattern agents: Technical pattern recognition and analysis
- indicator agents: Comprehensive technical indicator analysis 
- risk analysis agents: Multi-faceted risk assessment system
"""

from .volume import VolumeAgentsOrchestrator, VolumeAgentIntegrationManager
# from .patterns import PatternAgentsOrchestrator, patterns_orchestrator
from .indicators import IndicatorAgentsOrchestrator, indicators_orchestrator
# Risk analysis now handled directly by analysis service
# from .risk_analysis import RiskAgentsOrchestrator, risk_orchestrator

__all__ = [
    'VolumeAgentsOrchestrator',
    'VolumeAgentIntegrationManager',
    # 'PatternAgentsOrchestrator', 
    # 'patterns_orchestrator',
    'IndicatorAgentsOrchestrator',
    'indicators_orchestrator'
    # Risk analysis components are imported directly by analysis service
]
