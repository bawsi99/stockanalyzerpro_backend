#!/usr/bin/env python3
"""
MTF Agents Integration Manager

Provides a robust, fault-tolerant integration layer between the main analysis system
and the distributed MTF agents orchestrator. Following the same pattern as volume and
indicator agents integration managers, this provides:

- Health monitoring and validation
- Graceful fallback mechanisms  
- Performance metrics tracking
- Error recovery and logging
- Standardized interface for the main system

The integration manager ensures that MTF agent failures don't break the entire
analysis pipeline and provides consistent results even when individual agents fail.
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd

from .core.processor import CoreMTFProcessor, MTFAnalysisResult
from .mtf_agents import MTFAgentsOrchestrator, AggregatedMTFAnalysis
# MTFVisualizer import removed - chart generation disabled (redundant with text data)

logger = logging.getLogger(__name__)

@dataclass
class MTFAgentsHealthMetrics:
    """Health metrics for the MTF agents system"""
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0
    total_successful_runs: int = 0
    total_failed_runs: int = 0
    average_processing_time: float = 0.0
    last_error_message: Optional[str] = None
    core_processor_failures: int = 0
    agents_orchestrator_failures: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        total = self.total_successful_runs + self.total_failed_runs
        if total == 0:
            return 1.0
        return self.total_successful_runs / total
    
    @property
    def is_healthy(self) -> bool:
        """Determine if the system is healthy"""
        # System is unhealthy if we have too many consecutive failures
        # or if success rate is too low with sufficient samples
        if self.consecutive_failures >= 3:
            return False
        
        total_runs = self.total_successful_runs + self.total_failed_runs
        if total_runs >= 5 and self.success_rate < 0.6:
            return False
            
        return True

class MTFAgentIntegrationManager:
    """
    Integration manager for distributed MTF agents system.
    
    Provides a standardized interface to the MTF agents orchestrator with:
    - Health monitoring
    - Fallback mechanisms  
    - Performance tracking
    - Error recovery
    """
    
    def __init__(self):
        self.health_metrics = MTFAgentsHealthMetrics()
        self.core_processor = CoreMTFProcessor()
        self.agents_orchestrator = MTFAgentsOrchestrator()
        logger.info("MTFAgentIntegrationManager initialized")
    
    def is_mtf_agents_healthy(self) -> bool:
        """Check if the MTF agents system is healthy"""
        return self.health_metrics.is_healthy
    
    def get_health_reason(self) -> str:
        """Get a human-readable reason for the current health status"""
        if self.is_mtf_agents_healthy():
            return f"Healthy (Success rate: {self.health_metrics.success_rate:.1%})"
        
        reasons = []
        if self.health_metrics.consecutive_failures >= 3:
            reasons.append(f"Too many consecutive failures ({self.health_metrics.consecutive_failures})")
        
        total_runs = self.health_metrics.total_successful_runs + self.health_metrics.total_failed_runs
        if total_runs >= 5 and self.health_metrics.success_rate < 0.6:
            reasons.append(f"Low success rate ({self.health_metrics.success_rate:.1%})")
        
        return "; ".join(reasons) if reasons else "Unknown health issue"
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        return {
            'health_status': 'healthy' if self.is_mtf_agents_healthy() else 'unhealthy',
            'health_reason': self.get_health_reason(),
            'success_rate': self.health_metrics.success_rate,
            'consecutive_failures': self.health_metrics.consecutive_failures,
            'total_successful_runs': self.health_metrics.total_successful_runs,
            'total_failed_runs': self.health_metrics.total_failed_runs,
            'average_processing_time': self.health_metrics.average_processing_time,
            'last_error_message': self.health_metrics.last_error_message,
            'last_success_time': self.health_metrics.last_success_time,
            'core_processor_failures': self.health_metrics.core_processor_failures,
            'agents_orchestrator_failures': self.health_metrics.agents_orchestrator_failures
        }
    
    async def get_comprehensive_mtf_analysis(
        self, 
        symbol: str, 
        exchange: str = "NSE",
        include_agents: Optional[list] = None,
        generate_chart: bool = False
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Get comprehensive MTF analysis from the MTF agents system.
        
        This combines core MTF analysis with specialized agent insights.
        
        Args:
            symbol: Stock symbol to analyze
            exchange: Exchange name (default: NSE)
            include_agents: Optional list of specific agents to include
            generate_chart: Whether to generate MTF visualization chart (default: False)
                          Note: Chart generation is disabled by default as the visual bar chart
                          is redundant with the comprehensive text data already provided.
        
        Returns:
            Tuple[bool, Dict[str, Any]]: (success, mtf_analysis_dict)
            
            The mtf_analysis_dict includes a 'chart' key with image bytes if generate_chart=True
        """
        logger.info(f"[MTF_AGENTS] Starting comprehensive MTF analysis for {symbol}")
        start_time = time.time()
        
        try:
            # Check if system is healthy enough to attempt analysis
            if not self.is_mtf_agents_healthy():
                logger.warning(f"[MTF_AGENTS] System unhealthy: {self.get_health_reason()}")
                return await self._fallback_to_core_only(symbol, exchange)
            
            # Run the MTF agents orchestrator
            agent_results = await self.agents_orchestrator.analyze_comprehensive_mtf(
                symbol=symbol,
                exchange=exchange,
                include_agents=include_agents
            )
            
            processing_time = time.time() - start_time
            
            # Check if the analysis was successful
            if not agent_results.success:
                logger.warning(f"[MTF_AGENTS] MTF agents orchestrator failed for {symbol}: {agent_results.error_message}")
                self._record_failure(f"MTF agents orchestrator failed: {agent_results.error_message}", processing_time)
                self.health_metrics.agents_orchestrator_failures += 1
                return await self._fallback_to_core_only(symbol, exchange)
            
            # Convert to the expected format for the main system
            mtf_analysis = self._format_agents_results_for_main_system(
                agent_results,
                symbol=symbol,
                exchange=exchange
            )
            
            # Chart generation completely removed (redundant with comprehensive text data)
            # The bar chart visualization only showed the same trend/confidence data
            # that's already available in the structured text format, which LLMs can
            # process more effectively.
            # 
            # If visualization is needed in the future, it should show actual candlestick
            # charts with price action across timeframes, not just bar charts of text data.
            # The generate_chart parameter is kept for API compatibility but ignored.
            if generate_chart:
                logger.warning(
                    f"[MTF_AGENTS] Chart generation requested but disabled. "
                    f"The MTF bar chart was redundant with text data. "
                    f"See backend/agents/mtf_analysis/CHART_REMOVAL_SUMMARY.md for details."
                )
            
            # Record success
            self._record_success(processing_time)
            
            logger.info(f"[MTF_AGENTS] Successfully completed comprehensive MTF analysis for {symbol}")
            
            return True, mtf_analysis
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_message = str(e)
            logger.error(f"[MTF_AGENTS] Analysis failed for {symbol}: {error_message}")
            
            self._record_failure(error_message, processing_time)
            
            # Return fallback result
            return await self._fallback_to_core_only(symbol, exchange)
    
    async def _fallback_to_core_only(self, symbol: str, exchange: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Fallback to using only the core MTF processor when agents fail.
        """
        logger.info(f"[MTF_AGENTS] Falling back to core MTF processor for {symbol}")
        
        try:
            core_result = await self.core_processor.analyze_comprehensive_mtf(symbol, exchange)
            
            if core_result.success:
                # Convert core result to main system format
                fallback_analysis = self._format_core_result_for_main_system(core_result)
                logger.info(f"[MTF_AGENTS] Core processor fallback successful for {symbol}")
                return True, fallback_analysis
            else:
                logger.error(f"[MTF_AGENTS] Core processor fallback failed for {symbol}: {core_result.error_message}")
                self.health_metrics.core_processor_failures += 1
                return False, self._create_minimal_fallback_analysis(symbol, exchange, core_result.error_message)
                
        except Exception as e:
            logger.error(f"[MTF_AGENTS] Core processor fallback exception for {symbol}: {e}")
            self.health_metrics.core_processor_failures += 1
            return False, self._create_minimal_fallback_analysis(symbol, exchange, str(e))
    
    def _format_agents_results_for_main_system(
        self, 
        agent_results: AggregatedMTFAnalysis, 
        symbol: str, 
        exchange: str
    ) -> Dict[str, Any]:
        """
        Convert MTF agents results into the format expected by the main system.
        This matches the structure returned by the original EnhancedMultiTimeframeAnalyzer.
        """
        try:
            # Extract core analysis from agents results
            core_analysis = agent_results.core_analysis
            
            # Build the main system format
            formatted_result = {
                'success': True,
                'symbol': symbol,
                'exchange': exchange,
                'analysis_timestamp': core_analysis.get('analysis_timestamp', ''),
                'timeframe_analyses': core_analysis.get('timeframe_analyses', {}),
                'cross_timeframe_validation': core_analysis.get('cross_timeframe_validation', {}),
                'summary': core_analysis.get('summary', {}),
                
                # Add agent-specific enhancements
                'agent_insights': {
                    'total_agents_run': agent_results.total_agents_run,
                    'successful_agents': agent_results.successful_agents,
                    'failed_agents': agent_results.failed_agents,
                    'processing_time': agent_results.total_processing_time,
                    'confidence_score': agent_results.overall_confidence,
                    'agent_results': {
                        agent_name: {
                            'success': result.success,
                            'confidence': result.confidence_score,
                            'processing_time': result.processing_time,
                            'key_insights': result.analysis_data.get('summary', {}) if result.success else None
                        }
                        for agent_name, result in agent_results.individual_results.items()
                    }
                },
                
                # Unified analysis combining all agents
                'unified_analysis': agent_results.unified_analysis,
                'consensus_signals': agent_results.consensus_signals,
                'trading_recommendations': agent_results.trading_recommendations,
                
                # Quality metrics
                'analysis_quality': {
                    'data_coverage': len(core_analysis.get('timeframe_analyses', {})),
                    'signal_alignment': core_analysis.get('summary', {}).get('signal_alignment', 'unknown'),
                    'confidence_level': core_analysis.get('summary', {}).get('confidence', 0.0),
                    'risk_level': core_analysis.get('summary', {}).get('risk_level', 'unknown')
                }
            }
            
            return formatted_result
            
        except Exception as e:
            logger.error(f"[MTF_AGENTS] Error formatting agents results: {e}")
            # Return core analysis as fallback
            return agent_results.core_analysis or {}
    
    def _format_core_result_for_main_system(self, core_result: MTFAnalysisResult) -> Dict[str, Any]:
        """
        Convert core MTF processor results into the format expected by the main system.
        """
        return {
            'success': core_result.success,
            'symbol': core_result.symbol,
            'exchange': core_result.exchange,
            'analysis_timestamp': core_result.analysis_timestamp,
            'timeframe_analyses': core_result.timeframe_analyses,
            'cross_timeframe_validation': core_result.cross_timeframe_validation,
            'summary': core_result.summary,
            
            # Indicate this is a fallback result
            'agent_insights': {
                'total_agents_run': 0,
                'successful_agents': 0,
                'failed_agents': 0,
                'processing_time': core_result.processing_time,
                'confidence_score': core_result.confidence_score,
                'fallback_used': True,
                'fallback_reason': 'MTF agents system unhealthy or failed'
            },
            
            # Quality metrics
            'analysis_quality': {
                'data_coverage': len(core_result.timeframe_analyses),
                'signal_alignment': core_result.summary.get('signal_alignment', 'unknown'),
                'confidence_level': core_result.summary.get('confidence', 0.0),
                'risk_level': core_result.summary.get('risk_level', 'unknown'),
                'fallback_analysis': True
            }
        }
    
    def _create_minimal_fallback_analysis(self, symbol: str, exchange: str, error_message: str) -> Dict[str, Any]:
        """
        Create a minimal fallback analysis when everything fails.
        """
        return {
            'success': False,
            'symbol': symbol,
            'exchange': exchange,
            'analysis_timestamp': time.time(),
            'error': error_message,
            'timeframe_analyses': {},
            'cross_timeframe_validation': {
                'consensus_trend': 'neutral',
                'signal_strength': 0.0,
                'confidence_score': 0.0,
                'supporting_timeframes': [],
                'conflicting_timeframes': [],
                'neutral_timeframes': [],
                'divergence_detected': False,
                'key_conflicts': []
            },
            'summary': {
                'overall_signal': 'neutral',
                'confidence': 0.0,
                'timeframes_analyzed': 0,
                'signal_alignment': 'unknown',
                'risk_level': 'high',
                'recommendation': 'Unable to analyze - please try again later'
            },
            'agent_insights': {
                'total_agents_run': 0,
                'successful_agents': 0,
                'failed_agents': 0,
                'processing_time': 0.0,
                'confidence_score': 0.0,
                'fallback_used': True,
                'fallback_reason': f'Complete system failure: {error_message}'
            },
            'analysis_quality': {
                'data_coverage': 0,
                'signal_alignment': 'failed',
                'confidence_level': 0.0,
                'risk_level': 'high',
                'complete_failure': True
            }
        }
    
    def _record_success(self, processing_time: float):
        """Record a successful analysis"""
        self.health_metrics.total_successful_runs += 1
        self.health_metrics.consecutive_failures = 0
        self.health_metrics.last_success_time = time.time()
        
        # Update average processing time
        total_runs = self.health_metrics.total_successful_runs + self.health_metrics.total_failed_runs
        if total_runs == 1:
            self.health_metrics.average_processing_time = processing_time
        else:
            # Running average
            self.health_metrics.average_processing_time = (
                (self.health_metrics.average_processing_time * (total_runs - 1) + processing_time) / total_runs
            )
    
    def _record_failure(self, error_message: str, processing_time: float):
        """Record a failed analysis"""
        self.health_metrics.total_failed_runs += 1
        self.health_metrics.consecutive_failures += 1
        self.health_metrics.last_error_message = error_message
        
        # Update average processing time (even for failures)
        total_runs = self.health_metrics.total_successful_runs + self.health_metrics.total_failed_runs
        if total_runs == 1:
            self.health_metrics.average_processing_time = processing_time
        else:
            # Running average
            self.health_metrics.average_processing_time = (
                (self.health_metrics.average_processing_time * (total_runs - 1) + processing_time) / total_runs
            )

# Global instance
mtf_agent_integration_manager = MTFAgentIntegrationManager()