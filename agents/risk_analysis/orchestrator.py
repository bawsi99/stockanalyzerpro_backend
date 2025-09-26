#!/usr/bin/env python3
"""
Risk Analysis Agents Orchestrator

Coordinates and orchestrates multiple risk analysis agents to provide
comprehensive risk assessment for trading decisions.
"""

import asyncio
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

# Import individual risk agents
from .market_risk import MarketRiskProcessor
from .volatility_risk import VolatilityRiskProcessor
from .liquidity_risk import LiquidityRiskProcessor
from .technical_risk import TechnicalRiskProcessor

logger = logging.getLogger(__name__)

@dataclass
class RiskAgentResult:
    """Result from individual risk agent"""
    agent_name: str
    success: bool
    processing_time: float
    confidence_score: float
    result_data: Dict
    error_message: Optional[str] = None

@dataclass
class RiskAnalysisResult:
    """Aggregated result from all risk agents"""
    individual_results: Dict[str, RiskAgentResult]
    unified_analysis: Dict
    total_processing_time: float
    successful_agents: int
    failed_agents: int
    overall_risk_score: float
    overall_confidence: float

class RiskAgentsOrchestrator:
    """
    Orchestrates multiple risk analysis agents for comprehensive risk assessment
    """
    
    def __init__(self):
        self.agents = {
            'market': MarketRiskProcessor(),
            'volatility': VolatilityRiskProcessor(), 
            'liquidity': LiquidityRiskProcessor(),
            'technical': TechnicalRiskProcessor()
        }
        self.agent_weights = {
            'market': 0.35,      # Market risk is most critical
            'volatility': 0.25,  # Volatility risk is very important
            'liquidity': 0.20,   # Liquidity risk affects execution
            'technical': 0.20    # Technical risk affects signal reliability
        }
    
    async def analyze_risk_comprehensive(
        self, 
        symbol: str,
        stock_data: pd.DataFrame, 
        indicators: Dict,
        context: str = ""
    ) -> RiskAnalysisResult:
        """
        Run comprehensive risk analysis using all risk agents
        
        Args:
            symbol: Stock symbol being analyzed
            stock_data: OHLCV price data
            indicators: Technical indicators dictionary
            context: Additional context for analysis
            
        Returns:
            RiskAnalysisResult with aggregated findings
        """
        logger.info(f"[RISK_AGENTS] Executing {len(self.agents)} risk agents simultaneously...")
        
        start_time = datetime.now()
        
        # Execute all agents concurrently
        tasks = {}
        for agent_name, agent in self.agents.items():
            task = asyncio.create_task(
                self._execute_agent_safely(agent, stock_data, indicators, f"{context} - {agent_name}")
            )
            tasks[agent_name] = task
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        # Process results
        individual_results = {}
        successful_agents = 0
        failed_agents = 0
        
        for i, (agent_name, task) in enumerate(tasks.items()):
            result = results[i]
            
            if isinstance(result, Exception):
                # Task failed with exception
                individual_results[agent_name] = RiskAgentResult(
                    agent_name=agent_name,
                    success=False,
                    processing_time=0.0,
                    confidence_score=0.0,
                    result_data={},
                    error_message=str(result)
                )
                failed_agents += 1
            elif isinstance(result, dict) and 'error' in result:
                # Agent returned error result
                individual_results[agent_name] = RiskAgentResult(
                    agent_name=agent_name,
                    success=False,
                    processing_time=result.get('processing_time', 0.0),
                    confidence_score=0.0,
                    result_data=result,
                    error_message=result['error']
                )
                failed_agents += 1
            else:
                # Successful result
                individual_results[agent_name] = RiskAgentResult(
                    agent_name=agent_name,
                    success=True,
                    processing_time=result.get('processing_time', 0.0),
                    confidence_score=result.get('confidence_score', 0.0),
                    result_data=result
                )
                successful_agents += 1
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Generate unified analysis
        unified_analysis = await self._create_unified_analysis(individual_results, symbol, context)
        
        # Calculate overall metrics
        overall_risk_score, overall_confidence = self._calculate_overall_metrics(individual_results)
        
        logger.info(f"[RISK_AGENTS] Risk analysis complete: {successful_agents}/{len(self.agents)} agents succeeded")
        
        return RiskAnalysisResult(
            individual_results=individual_results,
            unified_analysis=unified_analysis,
            total_processing_time=total_time,
            successful_agents=successful_agents,
            failed_agents=failed_agents,
            overall_risk_score=overall_risk_score,
            overall_confidence=overall_confidence
        )
    
    async def _execute_agent_safely(self, agent, stock_data: pd.DataFrame, indicators: Dict, context: str) -> Dict:
        """Execute a single agent with error handling"""
        try:
            return await agent.analyze_async(stock_data, indicators, context)
        except Exception as e:
            logger.error(f"[RISK_AGENTS] Agent {getattr(agent, 'agent_name', 'unknown')} failed: {str(e)}")
            raise e
    
    async def _create_unified_analysis(
        self, 
        individual_results: Dict[str, RiskAgentResult], 
        symbol: str, 
        context: str
    ) -> Dict:
        """Create unified analysis from all agent results"""
        
        # Collect successful results
        successful_results = {
            name: result for name, result in individual_results.items() 
            if result.success
        }
        
        if not successful_results:
            return {
                'error': 'No successful risk analysis results',
                'failed_agents': len(individual_results)
            }
        
        # Risk level consensus
        risk_levels = []
        risk_scores = []
        
        for name, result in successful_results.items():
            data = result.result_data
            
            # Extract risk level
            if 'risk_level' in data:
                risk_levels.append(data['risk_level'])
            elif 'overall_risk_level' in data:
                risk_levels.append(data['overall_risk_level'])
            
            # Extract risk score
            if 'risk_score' in data:
                risk_scores.append(data['risk_score'])
            elif 'overall_technical_risk' in data:
                risk_scores.append(data['overall_technical_risk'])
        
        # Determine consensus
        risk_level_counts = {}
        for level in risk_levels:
            risk_level_counts[level] = risk_level_counts.get(level, 0) + 1
        
        consensus_level = max(risk_level_counts, key=risk_level_counts.get) if risk_level_counts else 'moderate'
        consensus_strength = 'strong' if len(set(risk_levels)) <= 2 else 'weak'
        
        # Average risk score
        avg_risk_score = sum(risk_scores) / len(risk_scores) if risk_scores else 0.5
        
        # Key risk factors
        key_risks = []
        mitigation_strategies = []
        
        for name, result in successful_results.items():
            data = result.result_data
            
            # Extract key risks
            if name == 'market' and 'volatility_risk' in data:
                if data['volatility_risk'].get('risk_level') == 'high':
                    key_risks.append('High market volatility detected')
            
            if name == 'liquidity' and 'risk_level' in data:
                if data['risk_level'] == 'high':
                    key_risks.append('Liquidity constraints identified')
            
            if name == 'technical' and 'risk_level' in data:
                if data['risk_level'] == 'high':
                    key_risks.append('Technical analysis reliability concerns')
            
            # Extract mitigation strategies
            if 'mitigation_strategies' in data:
                mitigation_strategies.extend(data['mitigation_strategies'][:2])  # Top 2 per agent
            elif 'recommendations' in data:
                mitigation_strategies.extend(data['recommendations'][:2])
        
        # Trading recommendations
        position_sizing = await self._determine_position_sizing(avg_risk_score, consensus_level)
        hedging_strategy = await self._determine_hedging_strategy(successful_results)
        
        return {
            'risk_summary': {
                'overall_level': consensus_level,
                'consensus_strength': consensus_strength,
                'average_risk_score': float(avg_risk_score),
                'key_risk_factors': key_risks[:5]  # Top 5 risks
            },
            'risk_breakdown': {
                'market_risk': successful_results.get('market', {}).result_data.get('overall_risk_level', 'unknown'),
                'volatility_risk': successful_results.get('volatility', {}).result_data.get('risk_level', 'unknown'),
                'liquidity_risk': successful_results.get('liquidity', {}).result_data.get('risk_level', 'unknown'),
                'technical_risk': successful_results.get('technical', {}).result_data.get('risk_level', 'unknown')
            },
            'trading_implications': {
                'position_sizing_recommendation': position_sizing,
                'hedging_strategy': hedging_strategy,
                'risk_monitoring_priorities': self._get_monitoring_priorities(successful_results)
            },
            'mitigation_strategies': list(set(mitigation_strategies))[:6],  # Unique top 6
            'confidence_metrics': {
                'analysis_completeness': len(successful_results) / len(individual_results),
                'signal_reliability': sum(r.confidence_score for r in successful_results.values()) / len(successful_results),
                'consensus_strength': 1.0 if consensus_strength == 'strong' else 0.6
            }
        }
    
    async def _determine_position_sizing(self, risk_score: float, risk_level: str) -> str:
        """Determine position sizing recommendation based on risk"""
        
        if risk_score > 0.7 or risk_level == 'high':
            return 'reduce_significantly'  # 30-50% of normal
        elif risk_score > 0.5 or risk_level == 'moderate':
            return 'reduce_moderately'     # 60-80% of normal
        else:
            return 'normal'                # 100% of normal
    
    async def _determine_hedging_strategy(self, successful_results: Dict[str, RiskAgentResult]) -> str:
        """Determine hedging strategy based on risk analysis"""
        
        high_risk_agents = sum(
            1 for result in successful_results.values() 
            if result.result_data.get('risk_level') == 'high' or 
               result.result_data.get('overall_risk_level') == 'high'
        )
        
        if high_risk_agents >= 2:
            return 'comprehensive_hedging'  # Multiple hedging instruments
        elif high_risk_agents >= 1:
            return 'selective_hedging'      # Targeted hedging
        else:
            return 'monitoring_only'        # No immediate hedging needed
    
    def _get_monitoring_priorities(self, successful_results: Dict[str, RiskAgentResult]) -> List[str]:
        """Get risk monitoring priorities"""
        
        priorities = []
        
        for name, result in successful_results.items():
            data = result.result_data
            risk_level = data.get('risk_level', data.get('overall_risk_level', 'low'))
            
            if risk_level == 'high':
                if name == 'market':
                    priorities.append('Monitor market volatility and correlation breakdowns')
                elif name == 'volatility':
                    priorities.append('Track volatility clustering and regime changes')
                elif name == 'liquidity':
                    priorities.append('Watch volume patterns and execution costs')
                elif name == 'technical':
                    priorities.append('Verify signal reliability and check for divergences')
        
        # Add default priorities if none identified
        if not priorities:
            priorities = [
                'Monitor overall market conditions',
                'Track position performance vs benchmarks',
                'Watch for changes in volatility patterns'
            ]
        
        return priorities[:4]  # Top 4 priorities
    
    def _calculate_overall_metrics(self, individual_results: Dict[str, RiskAgentResult]) -> Tuple[float, float]:
        """Calculate overall risk score and confidence"""
        
        successful_results = {
            name: result for name, result in individual_results.items()
            if result.success
        }
        
        if not successful_results:
            return 0.5, 0.0  # Default moderate risk, zero confidence
        
        # Weighted risk score
        weighted_risk_score = 0.0
        total_weight = 0.0
        
        for name, result in successful_results.items():
            weight = self.agent_weights.get(name, 0.25)
            
            # Extract risk score from result
            data = result.result_data
            risk_score = (
                data.get('risk_score', 0.0) or 
                data.get('overall_technical_risk', 0.0) or
                (0.8 if data.get('risk_level') == 'high' else 
                 0.5 if data.get('risk_level') == 'moderate' else 0.2)
            )
            
            weighted_risk_score += risk_score * weight
            total_weight += weight
        
        final_risk_score = weighted_risk_score / total_weight if total_weight > 0 else 0.5
        
        # Overall confidence (based on successful agents and their confidence)
        confidence_scores = [result.confidence_score for result in successful_results.values()]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Adjust confidence based on completeness
        completeness_factor = len(successful_results) / len(individual_results)
        overall_confidence = avg_confidence * completeness_factor
        
        return float(final_risk_score), float(overall_confidence)

# Global orchestrator instance
risk_orchestrator = RiskAgentsOrchestrator()