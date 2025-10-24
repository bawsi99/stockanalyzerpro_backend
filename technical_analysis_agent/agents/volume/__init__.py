"""
Volume Agents Package

This package contains all volume-based analysis agents for the StockAnalyzer Pro system.
It includes 5 specialized agents that work together to provide comprehensive volume analysis.

Agents:
- Volume Anomaly: Statistical volume spike detection
- Institutional Activity: Large order flow and institutional patterns  
- Volume Confirmation: Price-volume relationship validation
- Support/Resistance: Volume-based level identification
- Volume Momentum: Volume trend and momentum analysis
"""

from .volume_agents import (
    VolumeAgentsOrchestrator,
    VolumeAgentIntegrationManager,
    VolumeAgentResult,
    AggregatedVolumeAnalysis,
    volume_agents_logger
)

__all__ = [
    'VolumeAgentsOrchestrator',
    'VolumeAgentIntegrationManager', 
    'VolumeAgentResult',
    'AggregatedVolumeAnalysis',
    'volume_agents_logger'
]