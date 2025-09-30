#!/usr/bin/env python3
"""
MTF Agents Orchestrator and Integration Manager

This module provides the central orchestration and integration layer for all MTF analysis agents.
It manages simultaneous execution, result aggregation, and provides a unified interface to the main system.

Following the same pattern as volume_agents.py, this orchestrator handles:
- Core MTF processor integration
- Specialized MTF agents (intraday, swing, position)
- Result aggregation and consensus building
- Error handling and fallback mechanisms
- Performance monitoring and logging
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

# Import core MTF processor and specialized agents
from .core.processor import CoreMTFProcessor, MTFAnalysisResult
from .intraday.processor import IntradayMTFProcessor
from .swing.processor import SwingMTFProcessor  
from .position.processor import PositionMTFProcessor

# Set up logging
logger = logging.getLogger(__name__)

class MTFAgentsLogger:
    """
    Comprehensive logging system for MTF agents with structured logging and metrics
    """
    
    def __init__(self, base_logger=None):
        self.base_logger = base_logger or logger
        self.operation_id = 0
        
    def get_next_operation_id(self) -> str:
        """Generate unique operation ID for tracking"""
        self.operation_id += 1
        return f"MTF_{int(time.time())}_{self.operation_id:04d}"
    
    def log_operation_start(self, operation_type: str, symbol: str, agent_names: List[str] = None, **kwargs) -> str:
        """Log the start of an MTF agents operation"""
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
        
        self.base_logger.info(f"[MTF_AGENTS] Operation started: {json.dumps(log_data)}")
        return operation_id
    
    def log_operation_complete(self, operation_id: str, success: bool, processing_time: float, 
                             result_summary: Dict[str, Any] = None, **kwargs):
        """Log the completion of an MTF agents operation"""
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
        self.base_logger.log(level, f"[MTF_AGENTS] Operation completed: {json.dumps(log_data)}")
    
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
        self.base_logger.log(level, f"[MTF_AGENTS] Agent execution: {json.dumps(log_data)}")

# Global logger instance
mtf_agents_logger = MTFAgentsLogger(logger)

@dataclass
class MTFAgentResult:
    """Individual MTF agent analysis result"""
    agent_name: str
    success: bool
    processing_time: float
    analysis_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    confidence_score: Optional[float] = None

@dataclass
class AggregatedMTFAnalysis:
    """Aggregated results from all MTF agents"""
    success: bool = False
    individual_results: Dict[str, MTFAgentResult] = field(default_factory=dict)
    core_analysis: Dict[str, Any] = field(default_factory=dict)
    unified_analysis: Dict[str, Any] = field(default_factory=dict)
    total_processing_time: float = 0.0
    total_agents_run: int = 0
    successful_agents: int = 0
    failed_agents: int = 0
    overall_confidence: float = 0.0
    consensus_signals: Dict[str, Any] = field(default_factory=dict)
    trading_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None

class MTFAgentsOrchestrator:
    """
    Central orchestrator for all MTF analysis agents
    
    Manages simultaneous execution of core MTF analysis and specialized agents:
    - core: Core multi-timeframe technical analysis 
    - intraday: Scalping and short-term trading signals
    - swing: Medium-term swing trading analysis
    - position: Long-term position trading insights
    """
    
    def __init__(self):
        # Initialize core processor and specialized agents
        self.core_processor = CoreMTFProcessor()
        self.intraday_processor = IntradayMTFProcessor()
        self.swing_processor = SwingMTFProcessor()
        self.position_processor = PositionMTFProcessor()
        
        # Agent configuration
        # NOTE: Placeholder agents (intraday, swing, position) are currently DISABLED
        # as they are not yet integrated. Only core MTF processor is being used.
        self.agent_config = {
            'intraday': {
                'enabled': False,  # DISABLED - placeholder not integrated
                'weight': 0.25,
                'timeout': 120.0,
                'processor': self.intraday_processor,
                'description': 'Intraday scalping and short-term trading analysis'
            },
            'swing': {
                'enabled': False,  # DISABLED - placeholder not integrated
                'weight': 0.35,
                'timeout': 150.0,
                'processor': self.swing_processor,
                'description': 'Medium-term swing trading analysis'
            },
            'position': {
                'enabled': False,  # DISABLED - placeholder not integrated
                'weight': 0.40,
                'timeout': 180.0,
                'processor': self.position_processor,
                'description': 'Long-term position trading insights'
            }
        }
        
        logger.info("MTFAgentsOrchestrator initialized with core processor (specialized agents disabled)")
    
    async def analyze_comprehensive_mtf(
        self, 
        symbol: str, 
        exchange: str = "NSE",
        include_agents: Optional[List[str]] = None
    ) -> AggregatedMTFAnalysis:
        """
        Perform comprehensive MTF analysis using core processor and specialized agents.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            include_agents: Optional list of agent names to include
            
        Returns:
            AggregatedMTFAnalysis: Complete analysis results
        """
        operation_id = mtf_agents_logger.log_operation_start(
            "comprehensive_mtf_analysis",
            symbol,
            agent_names=include_agents or list(self.agent_config.keys()),
            exchange=exchange
        )
        
        start_time = time.time()
        
        try:
            # Step 1: Run core MTF analysis (this is essential)
            logger.info(f"[{operation_id}] Starting core MTF analysis for {symbol}")
            core_result = await self._execute_core_analysis(symbol, exchange)
            
            if not core_result.success:
                # If core fails, we can't proceed
                error_msg = f"Core MTF analysis failed: {core_result.error_message}"
                logger.error(f"[{operation_id}] {error_msg}")
                
                processing_time = time.time() - start_time
                mtf_agents_logger.log_operation_complete(
                    operation_id, False, processing_time,
                    result_summary={'error': error_msg}
                )
                
                return AggregatedMTFAnalysis(
                    success=False,
                    total_processing_time=processing_time,
                    error_message=error_msg
                )
            
            # Step 2: Run specialized agents in parallel
            selected_agents = include_agents or list(self.agent_config.keys())
            enabled_agents = [
                agent for agent in selected_agents 
                if agent in self.agent_config and self.agent_config[agent]['enabled']
            ]
            
            logger.info(f"[{operation_id}] Running {len(enabled_agents)} specialized agents: {enabled_agents}")
            
            # Prepare data for agents (extracted from core analysis)
            mtf_data = await self._prepare_mtf_data_for_agents(symbol, exchange, core_result)
            
            # Execute agents in parallel
            agent_results = await self._execute_agents_parallel(
                operation_id, enabled_agents, mtf_data, symbol, exchange
            )
            
            # Step 3: Aggregate and unify results
            aggregated_result = await self._aggregate_results(
                operation_id, core_result, agent_results, symbol, exchange
            )
            
            processing_time = time.time() - start_time
            aggregated_result.total_processing_time = processing_time
            
            # Log completion
            mtf_agents_logger.log_operation_complete(
                operation_id, 
                aggregated_result.success, 
                processing_time,
                result_summary={
                    'successful_agents': aggregated_result.successful_agents,
                    'failed_agents': aggregated_result.failed_agents,
                    'overall_confidence': aggregated_result.overall_confidence
                }
            )
            
            logger.info(f"[{operation_id}] Comprehensive MTF analysis completed for {symbol}")
            return aggregated_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Comprehensive MTF analysis failed: {str(e)}"
            logger.error(f"[{operation_id}] {error_msg}")
            logger.error(f"[{operation_id}] Traceback: {traceback.format_exc()}")
            
            mtf_agents_logger.log_operation_complete(
                operation_id, False, processing_time,
                result_summary={'error': error_msg}
            )
            
            return AggregatedMTFAnalysis(
                success=False,
                total_processing_time=processing_time,
                error_message=error_msg
            )
    
    async def _execute_core_analysis(self, symbol: str, exchange: str) -> MTFAnalysisResult:
        """Execute core MTF analysis"""
        try:
            return await self.core_processor.analyze_comprehensive_mtf(symbol, exchange)
        except Exception as e:
            logger.error(f"Core MTF analysis failed for {symbol}: {e}")
            return MTFAnalysisResult(
                success=False,
                symbol=symbol,
                exchange=exchange,
                error_message=str(e)
            )
    
    async def _prepare_mtf_data_for_agents(
        self, 
        symbol: str, 
        exchange: str, 
        core_result: MTFAnalysisResult
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepare MTF data for specialized agents.
        Extract timeframe data from core analysis or fetch fresh data.
        """
        try:
            # For now, we'll let each agent fetch its own data
            # In the future, we could optimize by sharing data from core analysis
            return {}
        except Exception as e:
            logger.warning(f"Failed to prepare MTF data for agents: {e}")
            return {}
    
    async def _execute_agents_parallel(
        self, 
        operation_id: str, 
        agent_names: List[str], 
        mtf_data: Dict[str, pd.DataFrame],
        symbol: str,
        exchange: str
    ) -> Dict[str, MTFAgentResult]:
        """Execute multiple MTF agents in parallel"""
        
        async def _execute_single_agent(agent_name: str) -> Tuple[str, MTFAgentResult]:
            """Execute a single agent with error handling and logging"""
            agent_start = time.time()
            
            try:
                config = self.agent_config[agent_name]
                processor = config['processor']
                timeout = config.get('timeout')
                
                logger.info(f"[{operation_id}] Starting agent: {agent_name}")
                
                # Execute agent with timeout
                if timeout:
                    analysis_result = await asyncio.wait_for(
                        processor.analyze_async(
                            mtf_data=mtf_data,
                            indicators={},  # Will be computed by each agent
                            context=f"symbol={symbol} exchange={exchange}"
                        ),
                        timeout=timeout
                    )
                else:
                    analysis_result = await processor.analyze_async(
                        mtf_data=mtf_data,
                        indicators={},
                        context=f"symbol={symbol} exchange={exchange}"
                    )
                
                processing_time = time.time() - agent_start
                
                # Create result
                if isinstance(analysis_result, dict) and not analysis_result.get('error'):
                    confidence = analysis_result.get('confidence_score', 0.5)
                    result = MTFAgentResult(
                        agent_name=agent_name,
                        success=True,
                        processing_time=processing_time,
                        analysis_data=analysis_result,
                        confidence_score=confidence
                    )
                    
                    mtf_agents_logger.log_agent_execution(
                        operation_id, agent_name, True, processing_time, confidence=confidence
                    )
                else:
                    # Agent returned error
                    error_msg = analysis_result.get('error', 'Unknown agent error')
                    result = MTFAgentResult(
                        agent_name=agent_name,
                        success=False,
                        processing_time=processing_time,
                        error_message=error_msg
                    )
                    
                    mtf_agents_logger.log_agent_execution(
                        operation_id, agent_name, False, processing_time, error_message=error_msg
                    )
                
                return agent_name, result
                
            except asyncio.TimeoutError:
                processing_time = time.time() - agent_start
                error_msg = f"Agent {agent_name} timed out after {timeout}s"
                logger.error(f"[{operation_id}] {error_msg}")
                
                result = MTFAgentResult(
                    agent_name=agent_name,
                    success=False,
                    processing_time=processing_time,
                    error_message=error_msg
                )
                
                mtf_agents_logger.log_agent_execution(
                    operation_id, agent_name, False, processing_time, error_message=error_msg
                )
                
                return agent_name, result
                
            except Exception as e:
                processing_time = time.time() - agent_start
                error_msg = f"Agent {agent_name} failed: {str(e)}"
                logger.error(f"[{operation_id}] {error_msg}")
                
                result = MTFAgentResult(
                    agent_name=agent_name,
                    success=False,
                    processing_time=processing_time,
                    error_message=error_msg
                )
                
                mtf_agents_logger.log_agent_execution(
                    operation_id, agent_name, False, processing_time, error_message=error_msg
                )
                
                return agent_name, result
        
        # Execute all agents concurrently
        tasks = [_execute_single_agent(agent_name) for agent_name in agent_names]
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        agent_results = {}
        for result in completed_results:
            if isinstance(result, Exception):
                logger.error(f"[{operation_id}] Agent execution exception: {result}")
                continue
            
            agent_name, agent_result = result
            agent_results[agent_name] = agent_result
        
        return agent_results
    
    async def _aggregate_results(
        self, 
        operation_id: str,
        core_result: MTFAnalysisResult, 
        agent_results: Dict[str, MTFAgentResult],
        symbol: str,
        exchange: str
    ) -> AggregatedMTFAnalysis:
        """Aggregate core analysis and agent results into unified analysis"""
        
        try:
            successful_agents = {name: result for name, result in agent_results.items() if result.success}
            failed_agents = {name: result for name, result in agent_results.items() if not result.success}
            
            # Build unified analysis
            unified_analysis = self._build_unified_analysis(core_result, successful_agents)
            
            # Generate consensus signals
            consensus_signals = self._generate_consensus_signals(core_result, successful_agents)
            
            # Generate trading recommendations
            trading_recommendations = self._generate_trading_recommendations(
                core_result, successful_agents, consensus_signals
            )
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(core_result, successful_agents)
            
            return AggregatedMTFAnalysis(
                success=True,
                individual_results=agent_results,
                core_analysis={
                    'success': core_result.success,
                    'symbol': core_result.symbol,
                    'exchange': core_result.exchange,
                    'analysis_timestamp': core_result.analysis_timestamp,
                    'timeframe_analyses': core_result.timeframe_analyses,
                    'cross_timeframe_validation': core_result.cross_timeframe_validation,
                    'summary': core_result.summary,
                    'processing_time': core_result.processing_time
                },
                unified_analysis=unified_analysis,
                total_agents_run=len(agent_results),
                successful_agents=len(successful_agents),
                failed_agents=len(failed_agents),
                overall_confidence=overall_confidence,
                consensus_signals=consensus_signals,
                trading_recommendations=trading_recommendations
            )
            
        except Exception as e:
            logger.error(f"[{operation_id}] Error aggregating results: {e}")
            return AggregatedMTFAnalysis(
                success=False,
                error_message=f"Result aggregation failed: {str(e)}"
            )
    
    def _build_unified_analysis(
        self, 
        core_result: MTFAnalysisResult, 
        successful_agents: Dict[str, MTFAgentResult]
    ) -> Dict[str, Any]:
        """Build unified analysis combining core and agent insights"""
        
        unified = {
            'core_consensus': {
                'trend': core_result.summary.get('overall_signal', 'neutral'),
                'confidence': core_result.confidence_score,
                'timeframes_analyzed': core_result.summary.get('timeframes_analyzed', 0),
                'risk_level': core_result.summary.get('risk_level', 'unknown')
            },
            'agent_insights': {},
            'cross_agent_validation': {},
            'key_opportunities': [],
            'risk_factors': []
        }
        
        # Add agent insights
        for agent_name, result in successful_agents.items():
            agent_data = result.analysis_data
            unified['agent_insights'][agent_name] = {
                'primary_signal': agent_data.get('summary', {}).get('overall_signal', 'neutral'),
                'confidence': result.confidence_score,
                'key_insights': agent_data.get('summary', {}),
                'processing_time': result.processing_time
            }
        
        # Cross-agent validation
        if len(successful_agents) > 1:
            agent_signals = [
                result.analysis_data.get('summary', {}).get('overall_signal', 'neutral')
                for result in successful_agents.values()
            ]
            
            bullish_count = agent_signals.count('bullish')
            bearish_count = agent_signals.count('bearish')
            neutral_count = agent_signals.count('neutral')
            
            unified['cross_agent_validation'] = {
                'agents_agreement': len(set(agent_signals)) == 1,
                'majority_signal': max(['bullish', 'bearish', 'neutral'], 
                                     key=lambda x: agent_signals.count(x)),
                'signal_distribution': {
                    'bullish': bullish_count,
                    'bearish': bearish_count,
                    'neutral': neutral_count
                }
            }
        
        return unified
    
    def _generate_consensus_signals(
        self, 
        core_result: MTFAnalysisResult, 
        successful_agents: Dict[str, MTFAgentResult]
    ) -> Dict[str, Any]:
        """Generate consensus signals across core and agents"""
        
        # Extract signals
        core_signal = core_result.summary.get('overall_signal', 'neutral')
        core_confidence = core_result.confidence_score
        
        agent_signals = []
        agent_weights = []
        
        for agent_name, result in successful_agents.items():
            agent_signal = result.analysis_data.get('summary', {}).get('overall_signal', 'neutral')
            agent_confidence = result.confidence_score or 0.5
            weight = self.agent_config[agent_name]['weight']
            
            agent_signals.append(agent_signal)
            agent_weights.append(weight * agent_confidence)
        
        # Calculate weighted consensus
        all_signals = [core_signal] + agent_signals
        all_weights = [0.5] + agent_weights  # Core gets base weight of 0.5
        
        # Weighted voting
        weighted_scores = {'bullish': 0.0, 'bearish': 0.0, 'neutral': 0.0}
        total_weight = sum(all_weights)
        
        for signal, weight in zip(all_signals, all_weights):
            if signal in weighted_scores:
                weighted_scores[signal] += weight
        
        # Determine consensus
        consensus_signal = max(weighted_scores.keys(), key=lambda k: weighted_scores[k])
        consensus_strength = weighted_scores[consensus_signal] / total_weight if total_weight > 0 else 0
        
        return {
            'consensus_signal': consensus_signal,
            'consensus_strength': consensus_strength,
            'signal_distribution': weighted_scores,
            'agreement_level': 'high' if consensus_strength > 0.7 else 'medium' if consensus_strength > 0.5 else 'low',
            'core_signal': core_signal,
            'core_confidence': core_confidence,
            'agent_signals': dict(zip([name for name in successful_agents.keys()], agent_signals))
        }
    
    def _generate_trading_recommendations(
        self, 
        core_result: MTFAnalysisResult, 
        successful_agents: Dict[str, MTFAgentResult],
        consensus_signals: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate trading recommendations based on consensus"""
        
        recommendations = []
        
        consensus_signal = consensus_signals.get('consensus_signal', 'neutral')
        consensus_strength = consensus_signals.get('consensus_strength', 0.0)
        agreement_level = consensus_signals.get('agreement_level', 'low')
        
        # Main recommendation based on consensus
        if consensus_signal == 'bullish' and consensus_strength > 0.6:
            if agreement_level == 'high':
                action = 'Strong Buy'
                confidence = 'High'
            else:
                action = 'Buy'
                confidence = 'Medium'
        elif consensus_signal == 'bearish' and consensus_strength > 0.6:
            if agreement_level == 'high':
                action = 'Strong Sell'
                confidence = 'High'
            else:
                action = 'Sell'
                confidence = 'Medium'
        else:
            action = 'Hold'
            confidence = 'Low'
        
        recommendations.append({
            'type': 'primary_recommendation',
            'action': action,
            'confidence': confidence,
            'rationale': f"Based on {consensus_signal} consensus with {consensus_strength:.1%} strength",
            'timeframe': 'multiple',
            'risk_level': core_result.summary.get('risk_level', 'unknown')
        })
        
        # Agent-specific recommendations
        for agent_name, result in successful_agents.items():
            agent_data = result.analysis_data
            if agent_data.get('summary'):
                recommendations.append({
                    'type': f'{agent_name}_recommendation',
                    'action': agent_data['summary'].get('recommendation', 'Hold'),
                    'confidence': result.confidence_score,
                    'rationale': f"{agent_name} analysis",
                    'timeframe': agent_name,
                    'details': agent_data['summary']
                })
        
        return recommendations
    
    def _calculate_overall_confidence(
        self, 
        core_result: MTFAnalysisResult, 
        successful_agents: Dict[str, MTFAgentResult]
    ) -> float:
        """Calculate overall confidence score combining core and agents"""
        
        # Start with core confidence
        core_confidence = core_result.confidence_score
        core_weight = 0.4
        
        # Add agent confidences with their weights
        agent_contribution = 0.0
        agent_total_weight = 0.0
        
        for agent_name, result in successful_agents.items():
            agent_confidence = result.confidence_score or 0.5
            agent_weight = self.agent_config[agent_name]['weight']
            
            agent_contribution += agent_confidence * agent_weight
            agent_total_weight += agent_weight
        
        # Normalize agent weights to sum to 0.6 (complement of core weight)
        if agent_total_weight > 0:
            agent_contribution = (agent_contribution / agent_total_weight) * 0.6
        
        # Combine core and agent confidences
        overall_confidence = core_confidence * core_weight + agent_contribution
        
        # Bonus for agent agreement
        if len(successful_agents) > 1:
            agent_signals = [
                result.analysis_data.get('summary', {}).get('overall_signal', 'neutral')
                for result in successful_agents.values()
            ]
            
            # If all agents agree, boost confidence
            if len(set(agent_signals)) == 1:
                overall_confidence = min(1.0, overall_confidence * 1.1)
        
        return max(0.0, min(1.0, overall_confidence))

# Global instance (optional - can be imported directly)
mtf_agents_orchestrator = MTFAgentsOrchestrator()