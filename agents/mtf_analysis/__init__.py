#!/usr/bin/env python3
"""
MTF Analysis Agents Module

This module provides the complete MTF (Multi-Timeframe) analysis system following
the agents pattern. It includes:

- Core MTF processor for fundamental analysis
- Specialized agents for different trading styles (intraday, swing, position)
- Integration manager for health monitoring and fallback mechanisms
- Orchestrator for coordinating all MTF analysis components

Usage:
    from agents.mtf_analysis import mtf_agent_integration_manager
    
    success, mtf_analysis = await mtf_agent_integration_manager.get_comprehensive_mtf_analysis(
        symbol="RELIANCE",
        exchange="NSE"
    )
"""

# Import core components
from .core.processor import (
    CoreMTFProcessor, 
    MTFTimeframeConfig, 
    MTFTimeframeAnalysis, 
    MTFCrossTimeframeValidation,
    MTFAnalysisResult
)

# Import specialized processors
from .intraday.processor import IntradayMTFProcessor
from .swing.processor import SwingMTFProcessor
from .position.processor import PositionMTFProcessor

# Import orchestrator and its data classes
from .mtf_agents import (
    MTFAgentsOrchestrator,
    MTFAgentResult,
    AggregatedMTFAnalysis,
    mtf_agents_orchestrator
)

# Import integration manager (main interface)
from .integration_manager import (
    MTFAgentIntegrationManager,
    MTFAgentsHealthMetrics,
    mtf_agent_integration_manager  # Global instance
)

# Keep backward compatibility with old orchestrator
try:
    from .orchestrator import MTFOrchestrator
except ImportError:
    # If old orchestrator doesn't exist, create alias
    MTFOrchestrator = MTFAgentsOrchestrator

# Export main interface and key classes
__all__ = [
    # Main interface
    'mtf_agent_integration_manager',
    
    # Integration manager
    'MTFAgentIntegrationManager',
    'MTFAgentsHealthMetrics',
    
    # Orchestrator
    'MTFAgentsOrchestrator',
    'mtf_agents_orchestrator',
    'MTFAgentResult',
    'AggregatedMTFAnalysis',
    
    # Core processor
    'CoreMTFProcessor',
    'MTFTimeframeConfig',
    'MTFTimeframeAnalysis', 
    'MTFCrossTimeframeValidation',
    'MTFAnalysisResult',
    
    # Specialized processors
    'IntradayMTFProcessor',
    'SwingMTFProcessor', 
    'PositionMTFProcessor',
    
    # Backward compatibility
    'MTFOrchestrator'
]

# Module-level convenience functions
async def get_mtf_analysis(symbol: str, exchange: str = "NSE", include_agents: list = None):
    """
    Convenience function to get comprehensive MTF analysis.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name (default: NSE)
        include_agents: Optional list of specific agents to include
        
    Returns:
        Tuple[bool, Dict[str, Any]]: (success, analysis_result)
    """
    return await mtf_agent_integration_manager.get_comprehensive_mtf_analysis(
        symbol=symbol,
        exchange=exchange,
        include_agents=include_agents
    )

def get_mtf_health_metrics():
    """
    Get health metrics for the MTF agents system.
    
    Returns:
        Dict[str, Any]: Health metrics including success rates, performance data
    """
    return mtf_agent_integration_manager.get_performance_metrics()

def is_mtf_system_healthy():
    """
    Check if the MTF agents system is healthy.
    
    Returns:
        bool: True if system is healthy, False otherwise
    """
    return mtf_agent_integration_manager.is_mtf_agents_healthy()
