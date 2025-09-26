#!/usr/bin/env python3
"""
Market Risk Analysis Agent

Analyzes market-wide risks including systemic risks, macroeconomic factors,
correlation breakdowns, and market regime changes that could impact positions.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class MarketRiskProcessor:
    """Processor for market risk analysis"""
    
    def __init__(self):
        self.agent_name = "market_risk"
        self.risk_categories = [
            "systemic_risk",
            "correlation_risk", 
            "regime_change_risk",
            "macroeconomic_risk",
            "liquidity_cascade_risk"
        ]
    
    async def analyze_async(self, stock_data: pd.DataFrame, indicators: Dict, context: str = "") -> Dict:
        """
        Perform comprehensive market risk analysis
        
        Args:
            stock_data: OHLCV price data
            indicators: Technical indicators dictionary
            context: Additional context for analysis
            
        Returns:
            Dictionary containing market risk assessment
        """
        try:
            logger.info(f"[MARKET_RISK] Starting market risk analysis...")
            start_time = datetime.now()
            
            # Analyze market volatility patterns
            volatility_risk = await self._analyze_volatility_patterns(stock_data, indicators)
            
            # Analyze correlation risks
            correlation_risk = await self._analyze_correlation_breakdown(stock_data, indicators)
            
            # Analyze market regime changes
            regime_risk = await self._analyze_regime_changes(stock_data, indicators)
            
            # Analyze macroeconomic risks
            macro_risk = await self._analyze_macro_risks(stock_data, indicators)
            
            # Analyze systemic risks
            systemic_risk = await self._analyze_systemic_risks(stock_data, indicators)
            
            # Calculate overall risk assessment
            risk_assessment = await self._calculate_risk_assessment(
                volatility_risk, correlation_risk, regime_risk, macro_risk, systemic_risk
            )
            
            # Generate risk scenarios
            scenarios = await self._generate_risk_scenarios(stock_data, indicators)
            
            # Create mitigation strategies
            mitigation = await self._generate_mitigation_strategies(risk_assessment, scenarios)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"[MARKET_RISK] Analysis completed in {processing_time:.2f}s")
            
            return {
                'agent_name': self.agent_name,
                'analysis_timestamp': datetime.now().isoformat(),
                'processing_time': processing_time,
                'context': context,
                
                # Risk Analysis
                'volatility_risk': volatility_risk,
                'correlation_risk': correlation_risk,
                'regime_change_risk': regime_risk,
                'macroeconomic_risk': macro_risk,
                'systemic_risk': systemic_risk,
                
                # Overall Assessment
                'overall_risk_level': risk_assessment['level'],
                'risk_score': risk_assessment['score'],
                'confidence_score': risk_assessment['confidence'],
                
                # Scenarios and Mitigation
                'risk_scenarios': scenarios,
                'mitigation_strategies': mitigation,
                
                # Trading Implications
                'position_sizing_recommendation': risk_assessment['position_sizing'],
                'hedging_recommendations': risk_assessment['hedging'],
                
                # Key Metrics
                'risk_metrics': {
                    'var_estimate': risk_assessment['var'],
                    'tail_risk': risk_assessment['tail_risk'],
                    'max_drawdown_risk': risk_assessment['max_drawdown'],
                    'correlation_stability': correlation_risk['stability']
                }
            }
            
        except Exception as e:
            logger.error(f"[MARKET_RISK] Analysis failed: {str(e)}")
            return {
                'agent_name': self.agent_name,
                'error': str(e),
                'confidence_score': 0.0,
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    async def _analyze_volatility_patterns(self, stock_data: pd.DataFrame, indicators: Dict) -> Dict:
        """Analyze market volatility patterns and clustering"""
        
        returns = stock_data['close'].pct_change().fillna(0)
        rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
        
        # Volatility clustering analysis
        vol_changes = rolling_vol.pct_change().fillna(0)
        clustering_score = abs(vol_changes.autocorr(lag=1)) if len(vol_changes) > 1 else 0.5
        
        # Current volatility regime
        current_vol = rolling_vol.iloc[-1] if len(rolling_vol) > 0 else 0.20
        historical_vol = rolling_vol.mean() if len(rolling_vol) > 0 else 0.20
        vol_regime = "high" if current_vol > historical_vol * 1.5 else "normal" if current_vol > historical_vol * 0.75 else "low"
        
        # Volatility trend
        vol_trend = "increasing" if rolling_vol.iloc[-5:].mean() > rolling_vol.iloc[-20:-5].mean() else "decreasing"
        
        return {
            'current_volatility': float(current_vol),
            'historical_volatility': float(historical_vol),
            'volatility_regime': vol_regime,
            'volatility_trend': vol_trend,
            'clustering_score': float(clustering_score),
            'risk_level': 'high' if current_vol > 0.30 or clustering_score > 0.7 else 'moderate'
        }
    
    async def _analyze_correlation_breakdown(self, stock_data: pd.DataFrame, indicators: Dict) -> Dict:
        """Analyze correlation breakdown risks"""
        
        # Simulate market correlation (in real implementation, this would use market data)
        returns = stock_data['close'].pct_change().fillna(0)
        
        # Rolling correlation stability
        rolling_corr = returns.rolling(window=30).corr(returns.shift(1))
        correlation_stability = 1 - rolling_corr.std() if len(rolling_corr.dropna()) > 0 else 0.5
        
        # Tail correlation risk
        extreme_days = returns[abs(returns) > returns.std() * 2]
        tail_correlation_risk = len(extreme_days) / len(returns) if len(returns) > 0 else 0.05
        
        return {
            'correlation_stability': float(correlation_stability),
            'tail_correlation_risk': float(tail_correlation_risk),
            'breakdown_probability': float(1 - correlation_stability),
            'stability': 'high' if correlation_stability > 0.8 else 'moderate' if correlation_stability > 0.6 else 'low'
        }
    
    async def _analyze_regime_changes(self, stock_data: pd.DataFrame, indicators: Dict) -> Dict:
        """Analyze market regime change risks"""
        
        returns = stock_data['close'].pct_change().fillna(0)
        
        # Trend persistence
        trend_periods = []
        current_trend = 0
        for ret in returns:
            if ret > 0:
                current_trend = current_trend + 1 if current_trend > 0 else 1
            elif ret < 0:
                current_trend = current_trend - 1 if current_trend < 0 else -1
            else:
                if current_trend != 0:
                    trend_periods.append(abs(current_trend))
                current_trend = 0
        
        avg_trend_length = np.mean(trend_periods) if trend_periods else 5
        
        # Volatility regime detection
        vol = returns.rolling(window=10).std()
        regime_changes = sum(1 for i in range(1, len(vol)) if abs(vol.iloc[i] - vol.iloc[i-1]) > vol.std())
        regime_stability = 1 - (regime_changes / len(vol)) if len(vol) > 0 else 0.8
        
        return {
            'regime_stability': float(regime_stability),
            'average_trend_length': float(avg_trend_length),
            'regime_change_frequency': float(regime_changes / len(vol)) if len(vol) > 0 else 0.1,
            'change_probability': 'high' if regime_stability < 0.6 else 'moderate' if regime_stability < 0.8 else 'low'
        }
    
    async def _analyze_macro_risks(self, stock_data: pd.DataFrame, indicators: Dict) -> Dict:
        """Analyze macroeconomic risk factors"""
        
        # Trend analysis for macro risk assessment
        returns = stock_data['close'].pct_change().fillna(0)
        
        # Interest rate risk proxy (trend strength)
        trend_strength = abs(returns.rolling(window=50).mean().iloc[-1]) if len(returns) > 50 else 0.01
        
        # Inflation risk proxy (volatility clustering)
        vol_clustering = returns.rolling(window=20).std().autocorr(lag=1) if len(returns) > 20 else 0.1
        
        # Currency risk proxy (return volatility)
        currency_risk = returns.rolling(window=30).std().iloc[-1] if len(returns) > 30 else 0.02
        
        return {
            'interest_rate_sensitivity': float(trend_strength * 10),  # Scale to 0-1
            'inflation_risk_exposure': float(abs(vol_clustering)),
            'currency_risk_exposure': float(currency_risk * 10),  # Scale to 0-1
            'overall_macro_risk': 'high' if trend_strength > 0.05 or abs(vol_clustering) > 0.5 else 'moderate'
        }
    
    async def _analyze_systemic_risks(self, stock_data: pd.DataFrame, indicators: Dict) -> Dict:
        """Analyze systemic risk factors"""
        
        returns = stock_data['close'].pct_change().fillna(0)
        
        # Liquidity proxy (volume patterns)
        volume_trend = stock_data['volume'].rolling(window=20).mean()
        liquidity_risk = 1 - (volume_trend.iloc[-1] / volume_trend.mean()) if len(volume_trend) > 0 else 0.3
        liquidity_risk = max(0, min(1, liquidity_risk))  # Clamp to 0-1
        
        # Contagion risk proxy (extreme return frequency)
        extreme_returns = sum(1 for r in returns if abs(r) > 0.05)
        contagion_risk = extreme_returns / len(returns) if len(returns) > 0 else 0.05
        
        # System stress proxy (volatility spikes)
        vol = returns.rolling(window=10).std()
        stress_events = sum(1 for i in range(1, len(vol)) if vol.iloc[i] > vol.iloc[i-1] * 2)
        system_stress = stress_events / len(vol) if len(vol) > 1 else 0.05
        
        return {
            'liquidity_risk': float(liquidity_risk),
            'contagion_risk': float(contagion_risk),
            'system_stress_level': float(system_stress),
            'systemic_risk_level': 'high' if liquidity_risk > 0.7 or contagion_risk > 0.1 else 'moderate'
        }
    
    async def _calculate_risk_assessment(self, vol_risk, corr_risk, regime_risk, macro_risk, sys_risk) -> Dict:
        """Calculate overall risk assessment"""
        
        # Risk scoring (0-1 scale)
        risk_scores = {
            'volatility': 0.8 if vol_risk['risk_level'] == 'high' else 0.4,
            'correlation': corr_risk['breakdown_probability'],
            'regime': 1 - regime_risk['regime_stability'],
            'macro': 0.7 if macro_risk['overall_macro_risk'] == 'high' else 0.3,
            'systemic': 0.8 if sys_risk['systemic_risk_level'] == 'high' else 0.4
        }
        
        # Weighted average risk score
        weights = {'volatility': 0.25, 'correlation': 0.20, 'regime': 0.20, 'macro': 0.20, 'systemic': 0.15}
        overall_score = sum(risk_scores[k] * weights[k] for k in risk_scores)
        
        # Risk level classification
        if overall_score > 0.7:
            risk_level = 'high'
            position_sizing = 'reduce_significantly'
        elif overall_score > 0.5:
            risk_level = 'moderate'
            position_sizing = 'reduce_moderately'
        else:
            risk_level = 'low'
            position_sizing = 'normal'
        
        return {
            'level': risk_level,
            'score': float(overall_score),
            'confidence': float(0.8 - abs(overall_score - 0.5) * 0.4),  # Higher confidence near extremes
            'position_sizing': position_sizing,
            'hedging': 'recommended' if overall_score > 0.6 else 'optional',
            'var': float(overall_score * 0.05),  # 1-day VaR estimate
            'tail_risk': float(overall_score * 0.15),  # Tail risk estimate
            'max_drawdown': float(overall_score * 0.25)  # Max drawdown risk
        }
    
    async def _generate_risk_scenarios(self, stock_data: pd.DataFrame, indicators: Dict) -> List[Dict]:
        """Generate risk scenario analysis"""
        
        current_price = stock_data['close'].iloc[-1]
        returns_vol = stock_data['close'].pct_change().std()
        
        scenarios = [
            {
                'name': 'market_correction',
                'probability': 0.25,
                'impact': -0.15,
                'description': 'Broad market correction of 15%',
                'estimated_loss': float(current_price * 0.15),
                'timeframe': '1-3 months'
            },
            {
                'name': 'volatility_spike',
                'probability': 0.35,
                'impact': -0.08,
                'description': 'Sharp volatility increase with 8% decline',
                'estimated_loss': float(current_price * 0.08),
                'timeframe': '1-2 weeks'
            },
            {
                'name': 'liquidity_crisis',
                'probability': 0.15,
                'impact': -0.25,
                'description': 'Liquidity crunch causing 25% decline',
                'estimated_loss': float(current_price * 0.25),
                'timeframe': '2-6 months'
            },
            {
                'name': 'base_case',
                'probability': 0.40,
                'impact': 0.05,
                'description': 'Normal market conditions with modest gains',
                'estimated_loss': float(-current_price * 0.05),
                'timeframe': '3-6 months'
            }
        ]
        
        return scenarios
    
    async def _generate_mitigation_strategies(self, risk_assessment: Dict, scenarios: List[Dict]) -> List[str]:
        """Generate risk mitigation strategies"""
        
        strategies = []
        
        if risk_assessment['level'] == 'high':
            strategies.extend([
                'Reduce position sizes by 40-60% to limit downside exposure',
                'Implement protective puts or collar strategies for downside protection',
                'Increase cash allocation to 20-30% for defensive positioning',
                'Consider inverse ETFs or VIX calls as portfolio hedges'
            ])
        elif risk_assessment['level'] == 'moderate':
            strategies.extend([
                'Reduce position sizes by 20-30% as precautionary measure',
                'Consider protective stops 10-15% below current levels',
                'Maintain higher cash reserves (15-20%) for opportunities'
            ])
        else:
            strategies.extend([
                'Maintain normal position sizing with active monitoring',
                'Keep standard stop losses in place (5-8% below entry)'
            ])
        
        # Add scenario-specific strategies
        high_prob_scenarios = [s for s in scenarios if s['probability'] > 0.3]
        for scenario in high_prob_scenarios:
            if scenario['impact'] < -0.10:
                strategies.append(f"Prepare for {scenario['name']}: {scenario['description']}")
        
        return strategies[:6]  # Return top 6 strategies