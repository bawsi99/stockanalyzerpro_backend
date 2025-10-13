#!/usr/bin/env python3
"""
Pattern Agents Orchestrator and Integration Manager

This module provides the central orchestration and integration layer for all pattern analysis agents.
It manages simultaneous execution, result aggregation, and provides a unified interface to the main system.

Pattern Agents:
- market_structure_agent: Analyzes swing points, BOS/CHOCH events, trend structure
- cross_validation_agent: Detects patterns and performs cross-validation analysis
"""

import logging
import asyncio
import time
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
import pandas as pd
import numpy as np
import traceback

# Import pattern agents
from agents.patterns.market_structure_agent.agent import MarketStructureAgent
from agents.patterns.cross_validation_agent.agent import CrossValidationAgent

# Set up logging
logger = logging.getLogger(__name__)

class PatternAgentsLogger:
    """
    Comprehensive logging system for pattern agents with structured logging and metrics
    """
    
    def __init__(self, base_logger=None):
        self.base_logger = base_logger or logger
        self.operation_id = 0
        
    def get_next_operation_id(self) -> str:
        """Generate unique operation ID for tracking"""
        self.operation_id += 1
        return f"PA_{int(time.time())}_{self.operation_id:04d}"
    
    def log_operation_start(self, operation_type: str, symbol: str, agent_names: List[str] = None, **kwargs) -> str:
        """Log the start of a pattern agents operation"""
        operation_id = self.get_next_operation_id()
        
        log_data = {
            'operation_id': operation_id,
            'operation_type': operation_type,
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'agents': agent_names or [],
            'status': 'started'
        }
        log_data.update(kwargs)
        
        self.base_logger.info(f"[PATTERN_AGENTS] Operation started: {json.dumps(log_data)}")
        return operation_id
    
    def log_operation_complete(self, operation_id: str, success: bool, processing_time: float, 
                             result_summary: Dict[str, Any] = None, **kwargs):
        """Log the completion of a pattern agents operation"""
        log_data = {
            'operation_id': operation_id,
            'status': 'completed',
            'success': success,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'result_summary': result_summary or {}
        }
        log_data.update(kwargs)
        
        level = logging.INFO if success else logging.WARNING
        self.base_logger.log(level, f"[PATTERN_AGENTS] Operation completed: {json.dumps(log_data)}")
    
    def log_agent_execution(self, operation_id: str, agent_name: str, success: bool, 
                          processing_time: float, error_message: str = None, 
                          confidence: float = None, **kwargs):
        """Log individual agent execution results"""
        log_data = {
            'operation_id': operation_id,
            'agent_name': agent_name,
            'success': success,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'confidence': confidence,
            'error_message': error_message
        }
        log_data.update(kwargs)
        
        level = logging.INFO if success else logging.ERROR
        self.base_logger.log(level, f"[PATTERN_AGENTS] Agent execution: {json.dumps(log_data)}")
    
    def log_partial_success(self, operation_id: str, successful_agents: List[str], 
                          failed_agents: List[str], fallback_activated: bool = False):
        """Log partial success scenarios"""
        log_data = {
            'operation_id': operation_id,
            'event_type': 'partial_success',
            'successful_agents': successful_agents,
            'failed_agents': failed_agents,
            'success_rate': len(successful_agents) / (len(successful_agents) + len(failed_agents)),
            'fallback_activated': fallback_activated,
            'timestamp': datetime.now().isoformat()
        }
        
        self.base_logger.warning(f"[PATTERN_AGENTS] Partial success: {json.dumps(log_data)}")

# Global logger instance
pattern_agents_logger = PatternAgentsLogger(logger)

@dataclass
class PatternAgentResult:
    """Individual pattern agent analysis result"""
    agent_name: str
    success: bool
    processing_time: float
    chart_image: Optional[bytes] = None
    analysis_data: Dict[str, Any] = field(default_factory=dict)
    llm_analysis: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    confidence_score: Optional[float] = None

@dataclass
class AggregatedPatternAnalysis:
    """Aggregated results from all pattern agents"""
    individual_results: Dict[str, PatternAgentResult] = field(default_factory=dict)
    unified_analysis: Dict[str, Any] = field(default_factory=dict)
    total_processing_time: float = 0.0
    successful_agents: int = 0
    failed_agents: int = 0
    overall_confidence: float = 0.0
    consensus_signals: Dict[str, Any] = field(default_factory=dict)
    pattern_conflicts: List[Dict[str, Any]] = field(default_factory=list)

class PatternAgentsOrchestrator:
    """
    Central orchestrator for all pattern analysis agents
    
    Manages simultaneous execution of pattern agents:
    - market_structure: Swing points, BOS/CHOCH events, trend structure analysis
    - cross_validation: Pattern detection and cross-validation analysis
    """
    
    def __init__(self):
        # Initialize pattern agents
        try:
            self.agents = {
                'market_structure': MarketStructureAgent(),
                'cross_validation': CrossValidationAgent()
            }
            
            print("✅ All pattern agents initialized")
            print(f"   - market_structure: {self.agents['market_structure'].agent_version}")
            print(f"   - cross_validation: {self.agents['cross_validation'].version}")
            
        except Exception as e:
            print(f"❌ Failed to initialize pattern agents: {e}")
            raise
        
        # Agent capabilities mapping
        self.capabilities = {
            'market_structure': {
                'swing_point_analysis': True,
                'bos_choch_detection': True,
                'trend_structure_analysis': True,
                'support_resistance_analysis': True,
                'fractal_analysis': True,
                'llm_enhanced_insights': True
            },
            'cross_validation': {
                'pattern_detection': True,
                'cross_validation': True,
                'multi_method_validation': True,
                'pattern_confidence_assessment': True,
                'validation_charts': True,
                'llm_validation_insights': True
            }
        }
    
    async def analyze_all_patterns(self, 
                                  stock_data: pd.DataFrame, 
                                  symbol: str,
                                  context: str = "",
                                  include_charts: bool = True,
                                  include_llm_analysis: bool = True) -> AggregatedPatternAnalysis:
        """
        Run all pattern agents concurrently and aggregate results.
        
        Args:
            stock_data: DataFrame with OHLCV data
            symbol: Stock symbol
            context: Additional context for analysis
            include_charts: Whether to generate charts
            include_llm_analysis: Whether to include LLM analysis
            
        Returns:
            Aggregated pattern analysis results
        """
        start_time = time.time()
        operation_id = pattern_agents_logger.log_operation_start(
            'analyze_all_patterns', symbol, list(self.agents.keys())
        )
        
        print(f"🔍 [PATTERN_AGENTS] Starting comprehensive pattern analysis for {symbol}")
        print(f"    Include Charts: {include_charts}")
        print(f"    Include LLM: {include_llm_analysis}")
        print(f"    Data Points: {len(stock_data)}")
        
        # Run agents concurrently
        tasks = []
        
        # Market Structure Agent
        tasks.append(asyncio.create_task(
            self._run_market_structure_agent(stock_data, symbol, context),
            name=f"market_structure_{symbol}"
        ))
        
        # Cross-Validation Agent (includes pattern detection)
        tasks.append(asyncio.create_task(
            self._run_cross_validation_agent(
                stock_data, symbol, include_charts, include_llm_analysis
            ),
            name=f"cross_validation_{symbol}"
        ))
        
        # Execute all agents
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        individual_results = {}
        successful_agents = []
        failed_agents = []
        
        agent_names = ['market_structure', 'cross_validation']
        
        for i, (agent_name, result) in enumerate(zip(agent_names, results)):
            if isinstance(result, Exception):
                # Agent failed
                pattern_agents_logger.log_agent_execution(
                    operation_id, agent_name, False, 0.0, str(result)
                )
                individual_results[agent_name] = PatternAgentResult(
                    agent_name=agent_name,
                    success=False,
                    processing_time=0.0,
                    error_message=str(result)
                )
                failed_agents.append(agent_name)
            else:
                # Agent succeeded
                success = result.get('success', False)
                processing_time = result.get('processing_time', 0.0)
                confidence = result.get('confidence_score', 0.0)
                
                pattern_agents_logger.log_agent_execution(
                    operation_id, agent_name, success, processing_time, 
                    confidence=confidence
                )
                
                individual_results[agent_name] = PatternAgentResult(
                    agent_name=agent_name,
                    success=success,
                    processing_time=processing_time,
                    chart_image=result.get('chart_image'),
                    analysis_data=result.get('technical_analysis') or result.get('cross_validation', {}),
                    llm_analysis=result.get('llm_analysis'),
                    confidence_score=confidence,
                    error_message=result.get('error') if not success else None
                )
                
                if success:
                    successful_agents.append(agent_name)
                else:
                    failed_agents.append(agent_name)
        
        # Aggregate results
        total_time = time.time() - start_time
        aggregated = self._aggregate_results(
            individual_results, total_time, successful_agents, failed_agents
        )
        
        # Log completion
        pattern_agents_logger.log_operation_complete(
            operation_id, len(successful_agents) > 0, total_time,
            {
                'successful_agents': len(successful_agents),
                'failed_agents': len(failed_agents),
                'overall_confidence': aggregated.overall_confidence
            }
        )
        
        if failed_agents:
            pattern_agents_logger.log_partial_success(
                operation_id, successful_agents, failed_agents
            )
        
        print(f"✅ [PATTERN_AGENTS] Analysis completed for {symbol} in {total_time:.2f}s")
        print(f"    Successful: {len(successful_agents)}/{len(agent_names)} agents")
        print(f"    Overall Confidence: {aggregated.overall_confidence:.2f}")
        
        return aggregated
    
    async def _run_market_structure_agent(self, 
                                         stock_data: pd.DataFrame, 
                                         symbol: str, 
                                         context: str) -> Dict[str, Any]:
        """Run market structure agent"""
        try:
            return await self.agents['market_structure'].analyze_complete(
                stock_data, symbol, context
            )
        except Exception as e:
            logger.error(f"Market structure agent failed for {symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _run_cross_validation_agent(self, 
                                         stock_data: pd.DataFrame, 
                                         symbol: str,
                                         include_charts: bool,
                                         include_llm_analysis: bool) -> Dict[str, Any]:
        """Run cross-validation agent (includes pattern detection)"""
        try:
            return await self.agents['cross_validation'].analyze_and_validate_patterns(
                stock_data=stock_data,
                symbol=symbol,
                include_charts=include_charts,
                include_llm_analysis=include_llm_analysis
            )
        except Exception as e:
            logger.error(f"Cross-validation agent failed for {symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _aggregate_results(self, 
                          individual_results: Dict[str, PatternAgentResult],
                          total_time: float,
                          successful_agents: List[str],
                          failed_agents: List[str]) -> AggregatedPatternAnalysis:
        """Aggregate results from individual agents"""
        
        # Calculate overall confidence
        confidences = [
            result.confidence_score for result in individual_results.values()
            if result.success and result.confidence_score is not None
        ]
        overall_confidence = np.mean(confidences) if confidences else 0.0
        
        # Build consensus signals
        consensus_signals = self._build_consensus_signals(individual_results)
        
        # Identify pattern conflicts
        pattern_conflicts = self._identify_pattern_conflicts(individual_results)
        
        # Build unified analysis summary
        unified_analysis = self._build_unified_analysis(individual_results)
        
        return AggregatedPatternAnalysis(
            individual_results=individual_results,
            unified_analysis=unified_analysis,
            total_processing_time=total_time,
            successful_agents=len(successful_agents),
            failed_agents=len(failed_agents),
            overall_confidence=overall_confidence,
            consensus_signals=consensus_signals,
            pattern_conflicts=pattern_conflicts
        )
    
    def _build_consensus_signals(self, 
                               individual_results: Dict[str, PatternAgentResult]) -> Dict[str, Any]:
        """Build consensus signals from successful agents"""
        consensus = {}
        
        # Extract signals from each successful agent
        market_structure_result = individual_results.get('market_structure')
        cross_validation_result = individual_results.get('cross_validation')
        
        # Market structure signals
        if market_structure_result and market_structure_result.success:
            ms_data = market_structure_result.analysis_data
            consensus['trend_direction'] = ms_data.get('trend_analysis', {}).get('primary_trend')
            consensus['structure_quality'] = ms_data.get('structure_quality', {})
            consensus['key_levels'] = ms_data.get('key_levels', {})
            consensus['bos_choch_signals'] = ms_data.get('bos_choch_events', [])
        
        # Pattern validation signals
        if cross_validation_result and cross_validation_result.success:
            cv_data = cross_validation_result.analysis_data
            consensus['detected_patterns'] = cv_data.get('detected_patterns', [])
            consensus['validation_confidence'] = cv_data.get('overall_validation_confidence', 0)
            consensus['pattern_levels'] = cv_data.get('key_levels', {})
        
        return consensus
    
    def _identify_pattern_conflicts(self, 
                                  individual_results: Dict[str, PatternAgentResult]) -> List[Dict[str, Any]]:
        """Identify conflicts between different agent analyses"""
        conflicts = []
        
        # Example: Check for trend conflicts between agents
        # This can be expanded based on specific conflict detection logic
        
        return conflicts
    
    def _build_unified_analysis(self, 
                              individual_results: Dict[str, PatternAgentResult]) -> Dict[str, Any]:
        """Build unified analysis summary"""
        unified = {
            'analysis_type': 'comprehensive_pattern_analysis',
            'agents_executed': list(individual_results.keys()),
            'successful_agents': [k for k, v in individual_results.items() if v.success],
            'failed_agents': [k for k, v in individual_results.items() if not v.success],
            'timestamp': datetime.now().isoformat()
        }
        
        # Add summary from successful agents
        for agent_name, result in individual_results.items():
            if result.success:
                unified[f'{agent_name}_summary'] = {
                    'confidence': result.confidence_score,
                    'processing_time': result.processing_time,
                    'has_chart': result.chart_image is not None,
                    'has_llm_analysis': result.llm_analysis is not None
                }
        
        return unified

class PatternAgentIntegrationManager:
    """
    Integration manager for pattern agents - provides high-level API for the analysis service
    """
    
    def __init__(self):
        self.orchestrator = PatternAgentsOrchestrator()
    
    async def get_comprehensive_pattern_analysis(self, 
                                               stock_data: pd.DataFrame, 
                                               symbol: str,
                                               context: str = "",
                                               include_charts: bool = True,
                                               include_llm_analysis: bool = True) -> Dict[str, Any]:
        """
        Get comprehensive pattern analysis from all agents.
        
        This is the main method used by the analysis service.
        """
        try:
            # Run comprehensive analysis
            aggregated_results = await self.orchestrator.analyze_all_patterns(
                stock_data=stock_data,
                symbol=symbol,
                context=context,
                include_charts=include_charts,
                include_llm_analysis=include_llm_analysis
            )
            
            # Convert to serializable format
            return {
                'success': aggregated_results.successful_agents > 0,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'processing_time': aggregated_results.total_processing_time,
                'overall_confidence': aggregated_results.overall_confidence,
                
                # Individual results
                'market_structure_analysis': self._format_agent_result(
                    aggregated_results.individual_results.get('market_structure')
                ),
                'cross_validation_analysis': self._format_agent_result(
                    aggregated_results.individual_results.get('cross_validation')
                ),
                
                # Aggregated insights
                'consensus_signals': aggregated_results.consensus_signals,
                'pattern_conflicts': aggregated_results.pattern_conflicts,
                'unified_analysis': aggregated_results.unified_analysis,
                
                # Summary statistics
                'agents_summary': {
                    'total_agents': len(aggregated_results.individual_results),
                    'successful_agents': aggregated_results.successful_agents,
                    'failed_agents': aggregated_results.failed_agents,
                    'success_rate': aggregated_results.successful_agents / len(aggregated_results.individual_results) if aggregated_results.individual_results else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Pattern analysis integration failed for {symbol}: {e}")
            return {
                'success': False,
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _format_agent_result(self, result: Optional[PatternAgentResult]) -> Dict[str, Any]:
        """Format individual agent result for API response"""
        if not result:
            return {'success': False, 'error': 'Agent not executed'}
        
        formatted = {
            'success': result.success,
            'processing_time': result.processing_time,
            'confidence_score': result.confidence_score,
            'has_chart': result.chart_image is not None,
            'has_llm_analysis': result.llm_analysis is not None
        }
        
        if result.success:
            formatted['analysis_data'] = result.analysis_data
            if result.llm_analysis:
                formatted['llm_analysis'] = result.llm_analysis
        else:
            formatted['error'] = result.error_message
        
        # Note: chart_image not included in API response due to size
        # Charts should be handled separately via dedicated endpoints
        
        return formatted

# Convenience functions for backward compatibility and easy imports
async def get_pattern_analysis(stock_data: pd.DataFrame, 
                             symbol: str,
                             **kwargs) -> Dict[str, Any]:
    """Convenience function for getting pattern analysis"""
    manager = PatternAgentIntegrationManager()
    return await manager.get_comprehensive_pattern_analysis(
        stock_data, symbol, **kwargs
    )

# Export key classes and functions
__all__ = [
    'PatternAgentsOrchestrator',
    'PatternAgentIntegrationManager', 
    'PatternAgentResult',
    'AggregatedPatternAnalysis',
    'get_pattern_analysis'
]