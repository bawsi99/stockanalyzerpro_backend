"""
Indicators Analysis Agents Orchestrator

Orchestrates the execution of specialized indicator agents and aggregates their results.
"""

import logging
import asyncio
import time
from typing import Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np

# Import indicator agents
from .trend import TrendIndicatorsProcessor
from .momentum import MomentumIndicatorsProcessor

logger = logging.getLogger(__name__)

@dataclass
class IndicatorAgentResult:
    """Individual indicator agent analysis result"""
    agent_name: str
    success: bool
    processing_time: float
    analysis_data: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    error_message: str = None

@dataclass
class AggregatedIndicatorAnalysis:
    """Aggregated results from all indicator agents"""
    individual_results: Dict[str, IndicatorAgentResult] = field(default_factory=dict)
    unified_analysis: Dict[str, Any] = field(default_factory=dict)
    total_processing_time: float = 0.0
    successful_agents: int = 0
    failed_agents: int = 0
    overall_confidence: float = 0.0

class IndicatorAgentsOrchestrator:
    """
    Central orchestrator for all indicator analysis agents
    
    Manages simultaneous execution of 2 indicator agents:
    - trend: Moving averages, trend direction, and strength
    - momentum: RSI, MACD, Stochastic oscillators
    """
    
    def __init__(self):
        # Initialize all agent processors
        self.trend = TrendIndicatorsProcessor()
        self.momentum = MomentumIndicatorsProcessor()
        
        # Agent configuration
        self.agent_config = {
            'trend': {
                'enabled': True,
                'weight': 0.30,
                'timeout': 30,
'processor': self.trend
            },
'momentum': {
                'enabled': True,
                'weight': 0.70,
                'timeout': 30,
                'processor': self.momentum,
                'charts': None
            }
        }
    
    async def analyze_indicators_comprehensive(self, symbol: str, stock_data: pd.DataFrame, 
                                            indicators: Dict[str, Any] = None, 
                                            context: str = "") -> AggregatedIndicatorAnalysis:
        """
        Execute all indicator analysis agents simultaneously and aggregate results
        
        Args:
            symbol: Stock symbol being analyzed
            stock_data: OHLCV price data
            indicators: Technical indicators data
            context: Additional analysis context
            
        Returns:
            AggregatedIndicatorAnalysis containing all agent results
        """
        start_time = time.time()
        
        try:
            # Create tasks for all enabled agents
            tasks = {}
            for agent_name, config in self.agent_config.items():
                if config.get('enabled', True):
                    task = self._execute_indicator_agent(
                        agent_name, config, symbol, stock_data, indicators, context
                    )
                    tasks[agent_name] = task
            
            # Execute all agents simultaneously
            logger.info(f"[INDICATORS_AGENTS] Executing {len(tasks)} indicator agents simultaneously...")
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            
            # Process results
            individual_results = {}
            successful_agents = 0
            failed_agents = 0
            
            for i, agent_name in enumerate(tasks.keys()):
                result = results[i]
                if isinstance(result, Exception):
                    # Handle agent failure
                    individual_results[agent_name] = IndicatorAgentResult(
                        agent_name=agent_name,
                        success=False,
                        processing_time=0.0,
                        error_message=str(result)
                    )
                    failed_agents += 1
                else:
                    individual_results[agent_name] = result
                    if result.success:
                        successful_agents += 1
                    else:
                        failed_agents += 1
            
            # Aggregate results
            total_processing_time = time.time() - start_time
            aggregated_analysis = self._aggregate_indicator_analysis(individual_results)
            
            # Create final result
            final_result = AggregatedIndicatorAnalysis(
                individual_results=individual_results,
                unified_analysis=aggregated_analysis,
                total_processing_time=total_processing_time,
                successful_agents=successful_agents,
                failed_agents=failed_agents,
                overall_confidence=self._calculate_overall_confidence(individual_results)
            )
            
            logger.info(f"[INDICATORS_AGENTS] Indicator analysis complete: {successful_agents}/{successful_agents+failed_agents} agents succeeded")
            return final_result
            
        except Exception as e:
            total_processing_time = time.time() - start_time
            logger.error(f"[INDICATORS_AGENTS] Analysis failed: {str(e)}")
            
            return AggregatedIndicatorAnalysis(
                total_processing_time=total_processing_time,
                failed_agents=len(self.agent_config),
                unified_analysis={'error': str(e), 'fallback_used': True}
            )
    
    async def _execute_indicator_agent(self, agent_name: str, config: Dict, symbol: str, 
                                     stock_data: pd.DataFrame, indicators: Dict[str, Any], 
                                     context: str) -> IndicatorAgentResult:
        """Execute a single indicator agent"""
        start_time = time.time()
        
        try:
            processor = config['processor']
            
            # Execute agent analysis
            analysis_data = await processor.analyze_async(stock_data, indicators, context)
            
            processing_time = time.time() - start_time
            
            return IndicatorAgentResult(
                agent_name=agent_name,
                success=True,
                processing_time=processing_time,
                analysis_data=analysis_data,
                confidence_score=analysis_data.get('confidence_score', 0.5)
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"[INDICATORS_AGENTS] Agent {agent_name} failed: {str(e)}")
            return IndicatorAgentResult(
                agent_name=agent_name,
                success=False,
                processing_time=processing_time,
                error_message=str(e)
            )
    
    def _aggregate_indicator_analysis(self, individual_results: Dict[str, IndicatorAgentResult]) -> Dict[str, Any]:
        """Aggregate individual indicator agent results into unified analysis"""
        
        unified = {
            'indicator_summary': {},
            'signal_consensus': {},
            'confidence_metrics': {},
            'trading_recommendations': {}
        }
        
        # Collect successful results
        successful_results = [r for r in individual_results.values() if r.success]
        
        if not successful_results:
            return unified
        
        # Aggregate trend analysis
        trend_result = individual_results.get('trend')
        if trend_result and trend_result.success:
            trend_data = trend_result.analysis_data
            unified['indicator_summary']['trend'] = {
                'direction': trend_data.get('overall_trend', {}).get('direction', 'neutral'),
                'strength': trend_data.get('overall_trend', {}).get('strength', 'weak'),
                'confidence': trend_data.get('confidence_score', 0.0)
            }
        
        # Aggregate momentum analysis
        momentum_result = individual_results.get('momentum')
        if momentum_result and momentum_result.success:
            momentum_data = momentum_result.analysis_data
            unified['indicator_summary']['momentum'] = {
                'direction': momentum_data.get('overall_momentum', {}).get('direction', 'neutral'),
                'strength': momentum_data.get('overall_momentum', {}).get('strength', 'weak'),
                'rsi_signal': momentum_data.get('rsi_analysis', {}).get('signal', 'neutral'),
                'confidence': momentum_data.get('confidence_score', 0.0)
            }
        
        # Signal consensus
        unified['signal_consensus'] = self._determine_signal_consensus(individual_results)
        
        # Confidence metrics
        confidence_scores = [r.confidence_score for r in successful_results]
        unified['confidence_metrics'] = {
            'average_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
            'agent_success_rate': len(successful_results) / len(individual_results)
        }
        
        return unified
    
    def _determine_signal_consensus(self, individual_results: Dict[str, IndicatorAgentResult]) -> Dict[str, Any]:
        """Determine consensus signals across indicator agents"""
        
        signals = []
        
        # Trend signals
        trend_result = individual_results.get('trend')
        if trend_result and trend_result.success:
            trend_direction = trend_result.analysis_data.get('overall_trend', {}).get('direction', 'neutral')
            if trend_direction != 'neutral':
                signals.append(trend_direction)
        
        # Momentum signals
        momentum_result = individual_results.get('momentum')
        if momentum_result and momentum_result.success:
            momentum_direction = momentum_result.analysis_data.get('overall_momentum', {}).get('direction', 'neutral')
            if momentum_direction != 'neutral':
                signals.append(momentum_direction)
        
        # Determine consensus
        if not signals:
            return {'consensus': 'neutral', 'strength': 'weak', 'agreement': 0.0}
        
        bullish_count = signals.count('bullish')
        bearish_count = signals.count('bearish')
        total_signals = len(signals)
        
        if bullish_count > bearish_count:
            return {
                'consensus': 'bullish',
                'strength': 'strong' if bullish_count/total_signals > 0.7 else 'moderate',
                'agreement': bullish_count / total_signals
            }
        elif bearish_count > bullish_count:
            return {
                'consensus': 'bearish',
                'strength': 'strong' if bearish_count/total_signals > 0.7 else 'moderate',
                'agreement': bearish_count / total_signals
            }
        else:
            return {
                'consensus': 'mixed',
                'strength': 'weak',
                'agreement': 0.5
            }
    
    def _calculate_overall_confidence(self, individual_results: Dict[str, IndicatorAgentResult]) -> float:
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

# Global orchestrator instance
indicators_orchestrator = IndicatorAgentsOrchestrator()