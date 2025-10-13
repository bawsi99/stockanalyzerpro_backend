"""
Pattern Analysis Agents Orchestrator and Integration Manager

This module provides the central orchestration and integration layer for all pattern analysis agents.
It manages simultaneous execution, result aggregation, and provides a unified interface to the main system.
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

# Import all pattern agents
from .reversal import ReversalPatternsProcessor, ReversalPatternsCharts
from .continuation import ContinuationPatternsProcessor, ContinuationPatternsCharts
# Removed: TechnicalOverviewProcessor - redundant with indicators agent system

# Import pattern recognition agent
from .pattern_recognition import PatternRecognitionProcessor, PatternRecognitionCharts

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
    
    def log_error_with_context(self, operation_id: str, error: Exception, context: Dict[str, Any]):
        """Log errors with full context for debugging"""
        log_data = {
            'operation_id': operation_id,
            'event_type': 'error',
            'error_type': type(error).__name__,
            'error_message': str(error),
            'error_traceback': traceback.format_exc(),
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        
        self.base_logger.error(f"[PATTERN_AGENTS] Error with context: {json.dumps(log_data)}")

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
    prompt_text: Optional[str] = None
    llm_response: Optional[str] = None
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
    conflicting_signals: List[Dict[str, Any]] = field(default_factory=list)

class PatternAgentsOrchestrator:
    """
    Central orchestrator for all pattern analysis agents
    
    Manages simultaneous execution of 3 specialized pattern agents:
    - reversal: Divergences, double tops/bottoms reversal patterns
    - continuation: Triangles, flags, channels continuation patterns  
    - pattern_recognition: Advanced pattern identification with multi-stage LLM analysis
    
    Note: Technical overview functionality moved to dedicated indicators system
    """
    
    def __init__(self, gemini_client=None):
        self.gemini_client = gemini_client
        
        # Initialize all agent processors (technical_overview removed - redundant with indicators system)
        self.reversal = ReversalPatternsProcessor()
        self.continuation = ContinuationPatternsProcessor()
        self.pattern_recognition = PatternRecognitionProcessor(llm_client=gemini_client)  # Pass LLM client for multi-stage processing
        
        # Initialize all chart generators
        self.reversal_charts = ReversalPatternsCharts()
        self.continuation_charts = ContinuationPatternsCharts()
        self.recognition_charts = PatternRecognitionCharts()
        
        # Agent configuration (technical_overview removed - use dedicated indicators system instead)
        self.agent_config = {
            'reversal': {
                'enabled': True,
                'weight': 0.33,  # Increased weight since we removed technical_overview
                'timeout': 30,
                'processor': self.reversal,
                'charts': self.reversal_charts,
                'prompt_template': 'optimized_reversal_patterns'
            },
            'continuation': {
                'enabled': True,
                'weight': 0.33,  # Increased weight since we removed technical_overview
                'timeout': 30,
                'processor': self.continuation,
                'charts': self.continuation_charts,
                'prompt_template': 'optimized_continuation_levels'
            },
            'pattern_recognition': {
                'enabled': True,
                'weight': 0.34,  # Increased weight since we removed technical_overview (includes multi-stage LLM)
                'timeout': 30,
                'processor': self.pattern_recognition,
                'charts': self.recognition_charts,
                'prompt_template': 'optimized_pattern_analysis'
            }
        }
    
    async def analyze_patterns_comprehensive(self, symbol: str, stock_data: pd.DataFrame, 
                                           indicators: Dict[str, Any] = None, 
                                           context: str = "", 
                                           chart_images: Dict[str, bytes] = None) -> AggregatedPatternAnalysis:
        """
        Execute all pattern analysis agents simultaneously and aggregate results
        
        Args:
            symbol: Stock symbol being analyzed
            stock_data: OHLCV price data
            indicators: Technical indicators data
            context: Additional analysis context
            chart_images: Pre-generated chart images for analysis
            
        Returns:
            AggregatedPatternAnalysis containing all agent results
        """
        # Log operation start
        agent_names = list(self.agent_config.keys())
        operation_id = pattern_agents_logger.log_operation_start(
            'patterns_comprehensive_analysis', 
            symbol, 
            agent_names
        )
        
        start_time = time.time()
        
        try:
            # Create tasks for all enabled agents
            tasks = {}
            for agent_name, config in self.agent_config.items():
                if config.get('enabled', True):
                    # Create task for each agent
                    task = self._execute_pattern_agent(
                        operation_id,
                        agent_name,
                        config,
                        symbol,
                        stock_data,
                        indicators,
                        context,
                        chart_images
                    )
                    tasks[agent_name] = task
            
            # Execute all agents simultaneously
            print(f"[PATTERN_AGENTS] Executing {len(tasks)} pattern agents simultaneously...")
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            
            # Process results
            individual_results = {}
            successful_agents = 0
            failed_agents = 0
            
            for i, agent_name in enumerate(tasks.keys()):
                result = results[i]
                if isinstance(result, Exception):
                    # Handle agent failure
                    individual_results[agent_name] = PatternAgentResult(
                        agent_name=agent_name,
                        success=False,
                        processing_time=0.0,
                        error_message=str(result)
                    )
                    failed_agents += 1
                    pattern_agents_logger.log_agent_execution(
                        operation_id, agent_name, False, 0.0, str(result)
                    )
                else:
                    individual_results[agent_name] = result
                    if result.success:
                        successful_agents += 1
                    else:
                        failed_agents += 1
                    
                    pattern_agents_logger.log_agent_execution(
                        operation_id, agent_name, result.success, 
                        result.processing_time, result.error_message,
                        result.confidence_score
                    )
            
            # Aggregate results
            total_processing_time = time.time() - start_time
            aggregated_analysis = self._aggregate_pattern_analysis(individual_results)
            
            # Create final result
            final_result = AggregatedPatternAnalysis(
                individual_results=individual_results,
                unified_analysis=aggregated_analysis,
                total_processing_time=total_processing_time,
                successful_agents=successful_agents,
                failed_agents=failed_agents,
                overall_confidence=self._calculate_overall_confidence(individual_results),
                consensus_signals=self._identify_consensus_signals(individual_results),
                conflicting_signals=self._identify_conflicting_signals(individual_results)
            )
            
            # Log completion
            pattern_agents_logger.log_operation_complete(
                operation_id, successful_agents > 0, total_processing_time,
                {
                    'successful_agents': successful_agents,
                    'failed_agents': failed_agents,
                    'overall_confidence': final_result.overall_confidence
                }
            )
            
            print(f"[PATTERN_AGENTS] Pattern analysis complete: {successful_agents}/{successful_agents+failed_agents} agents succeeded")
            return final_result
            
        except Exception as e:
            total_processing_time = time.time() - start_time
            pattern_agents_logger.log_error_with_context(
                operation_id, e, {'symbol': symbol, 'processing_time': total_processing_time}
            )
            
            # Return failure result
            return AggregatedPatternAnalysis(
                total_processing_time=total_processing_time,
                failed_agents=len(self.agent_config),
                unified_analysis={'error': str(e), 'fallback_used': True}
            )
    
    async def _execute_pattern_agent(self, operation_id: str, agent_name: str, config: Dict,
                                   symbol: str, stock_data: pd.DataFrame,
                                   indicators: Dict[str, Any], context: str,
                                   chart_images: Dict[str, bytes] = None) -> PatternAgentResult:
        """Execute a single pattern agent"""
        start_time = time.time()
        
        try:
            processor = config['processor']
            charts = config['charts']
            
            # Skip chart generation to avoid blocking errors
            chart_image = None
            
            # Execute agent analysis
            if hasattr(processor, 'analyze_async'):
                analysis_data = await processor.analyze_async(
                    stock_data, indicators, context, chart_image
                )
            else:
                analysis_data = processor.analyze(
                    stock_data, indicators, context, chart_image
                )
            
            processing_time = time.time() - start_time
            
            return PatternAgentResult(
                agent_name=agent_name,
                success=True,
                processing_time=processing_time,
                chart_image=chart_image,
                analysis_data=analysis_data,
                confidence_score=analysis_data.get('confidence_score', 0.5)
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return PatternAgentResult(
                agent_name=agent_name,
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def _aggregate_pattern_analysis(self, individual_results: Dict[str, PatternAgentResult]) -> Dict[str, Any]:
        """Aggregate individual pattern agent results into unified analysis"""
        unified = {
            'pattern_summary': {},
            'signal_consensus': {},
            'confidence_metrics': {},
            'key_levels': {},
            'trading_recommendations': {}
        }
        
        # Aggregate pattern findings
        all_patterns = []
        all_signals = []
        confidence_scores = []
        
        for agent_name, result in individual_results.items():
            if result.success and result.analysis_data:
                data = result.analysis_data
                
                # Extract patterns - handle different data structures
                if 'patterns' in data:
                    patterns = data['patterns']
                    if isinstance(patterns, list):
                        all_patterns.extend(patterns)
                
                # Also check for reversal patterns
                if 'reversal_patterns' in data:
                    rev_patterns = data['reversal_patterns']
                    if isinstance(rev_patterns, dict):
                        for pattern_type, pattern_list in rev_patterns.items():
                            if isinstance(pattern_list, list):
                                all_patterns.extend(pattern_list)
                
                # Also check for continuation patterns
                if 'continuation_patterns' in data:
                    cont_patterns = data['continuation_patterns']
                    if isinstance(cont_patterns, dict):
                        for pattern_type, pattern_list in cont_patterns.items():
                            if isinstance(pattern_list, list):
                                all_patterns.extend(pattern_list)
                
                # Extract signals
                if 'signals' in data:
                    all_signals.extend(data['signals'])
                
                # Extract confidence
                if result.confidence_score:
                    confidence_scores.append(result.confidence_score)
        
        # Build unified summary - safely handle pattern extraction
        pattern_types = []
        high_confidence_patterns = []
        
        for pattern in all_patterns:
            if isinstance(pattern, dict):
                pattern_type = pattern.get('type', 'unknown')
                pattern_types.append(pattern_type)
                if pattern.get('confidence', 0) > 0.7:
                    high_confidence_patterns.append(pattern)
            elif isinstance(pattern, str):
                pattern_types.append(pattern)
        
        unified['pattern_summary'] = {
            'total_patterns_identified': len(all_patterns),
            'pattern_types': list(set(pattern_types)),
            'high_confidence_patterns': high_confidence_patterns
        }
        
        unified['confidence_metrics'] = {
            'average_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
            'confidence_range': [min(confidence_scores), max(confidence_scores)] if confidence_scores else [0, 0],
            'agent_success_rate': len([r for r in individual_results.values() if r.success]) / len(individual_results)
        }
        
        return unified
    
    def _calculate_overall_confidence(self, individual_results: Dict[str, PatternAgentResult]) -> float:
        """Calculate overall confidence from individual agent confidences"""
        successful_results = [r for r in individual_results.values() if r.success and r.confidence_score]
        if not successful_results:
            return 0.0
        
        # Weight by agent configuration
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for result in successful_results:
            weight = self.agent_config.get(result.agent_name, {}).get('weight', 0.25)
            weighted_confidence += result.confidence_score * weight
            total_weight += weight
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def _identify_consensus_signals(self, individual_results: Dict[str, PatternAgentResult]) -> Dict[str, Any]:
        """Identify consensus signals across pattern agents"""
        signals = {}
        signal_counts = {}
        
        for agent_name, result in individual_results.items():
            if result.success and result.analysis_data:
                data = result.analysis_data
                if 'primary_signal' in data:
                    signal = data['primary_signal']
                    signal_counts[signal] = signal_counts.get(signal, 0) + 1
                    if signal not in signals:
                        signals[signal] = []
                    signals[signal].append(agent_name)
        
        # Find consensus (signals supported by majority)
        total_agents = len([r for r in individual_results.values() if r.success])
        consensus_threshold = total_agents * 0.5
        
        consensus_signals = {
            signal: {
                'supporting_agents': agents,
                'support_count': count,
                'consensus_strength': count / total_agents if total_agents > 0 else 0
            }
            for signal, agents in signals.items()
            for count in [signal_counts[signal]]
            if count >= consensus_threshold
        }
        
        return consensus_signals
    
    def _identify_conflicting_signals(self, individual_results: Dict[str, PatternAgentResult]) -> List[Dict[str, Any]]:
        """Identify conflicting signals between pattern agents"""
        conflicts = []
        
        # Extract all signals with their sources
        all_signals = []
        for agent_name, result in individual_results.items():
            if result.success and result.analysis_data:
                data = result.analysis_data
                if 'primary_signal' in data:
                    all_signals.append({
                        'agent': agent_name,
                        'signal': data['primary_signal'],
                        'confidence': result.confidence_score or 0.5
                    })
        
        # Find conflicting signals (opposing directions)
        bullish_signals = [s for s in all_signals if 'bullish' in s['signal'].lower()]
        bearish_signals = [s for s in all_signals if 'bearish' in s['signal'].lower()]
        
        if bullish_signals and bearish_signals:
            conflicts.append({
                'conflict_type': 'directional_conflict',
                'bullish_agents': [s['agent'] for s in bullish_signals],
                'bearish_agents': [s['agent'] for s in bearish_signals],
                'severity': 'high' if len(bullish_signals) == len(bearish_signals) else 'medium'
            })
        
        return conflicts


# Singleton instance for global use
patterns_orchestrator = PatternAgentsOrchestrator()