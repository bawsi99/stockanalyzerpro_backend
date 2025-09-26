"""
Technical Overview Processor

This module provides comprehensive technical analysis using the optimized_technical_overview
prompt template with JSON-structured output.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
import json
import asyncio

logger = logging.getLogger(__name__)

class TechnicalOverviewProcessor:
    """
    Processor for comprehensive technical overview analysis.
    
    This processor provides:
    - Overall trend analysis with multiple indicators
    - Volume confirmation analysis 
    - Support/resistance level identification
    - Risk assessment and confidence scoring
    - JSON-structured output for integration
    """
    
    def __init__(self):
        self.name = "technical_overview"
        self.description = "Provides comprehensive technical analysis overview"
    
    async def analyze_async(self, stock_data: pd.DataFrame, indicators: Dict[str, Any] = None, 
                          context: str = "", chart_image: bytes = None) -> Dict[str, Any]:
        """
        Asynchronous comprehensive technical analysis
        
        Returns JSON-structured analysis matching optimized_technical_overview prompt format
        """
        try:
            start_time = pd.Timestamp.now()
            
            # Analyze trend components
            trend_analysis = self._analyze_trend(stock_data, indicators)
            volume_analysis = self._analyze_volume(stock_data)
            momentum_analysis = self._analyze_momentum(indicators)
            support_resistance = self._analyze_support_resistance(stock_data)
            risk_assessment = self._assess_risk(stock_data, indicators)
            
            # Calculate overall confidence
            confidence_score = self._calculate_confidence(trend_analysis, volume_analysis, momentum_analysis)
            
            processing_time = (pd.Timestamp.now() - start_time).total_seconds()
            
            # Build JSON-structured result matching prompt template
            analysis_result = {
                'agent_name': self.name,
                'processing_time': processing_time,
                'timestamp': pd.Timestamp.now().isoformat(),
                
                # Core analysis components (matches JSON schema in prompt)
                'trend_analysis': trend_analysis,
                'volume_analysis': volume_analysis, 
                'momentum_analysis': momentum_analysis,
                'support_resistance': support_resistance,
                'risk_assessment': risk_assessment,
                'confidence_score': confidence_score,
                'analysis_quality': self._assess_analysis_quality(indicators),
                
                # Additional metadata
                'primary_signal': self._determine_primary_signal(trend_analysis, momentum_analysis),
                'patterns': self._identify_patterns(stock_data, indicators),
                'trading_recommendations': self._generate_trading_recommendations(trend_analysis, risk_assessment)
            }
            
            logger.info(f"[TECHNICAL_OVERVIEW] Analysis completed in {processing_time:.2f}s")
            return analysis_result
            
        except Exception as e:
            logger.error(f"[TECHNICAL_OVERVIEW] Analysis failed: {str(e)}")
            return {
                'agent_name': self.name,
                'error': str(e),
                'success': False,
                'confidence_score': 0.0
            }
    
    def _analyze_trend(self, stock_data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall trend using multiple indicators"""
        current_price = stock_data['close'].iloc[-1]
        recent_prices = stock_data['close'].tail(20)
        
        # Determine trend direction
        price_change = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
        
        if price_change > 0.05:
            overall_trend = "bullish"
            trend_strength = "strong" if price_change > 0.15 else "medium"
        elif price_change < -0.05:
            overall_trend = "bearish" 
            trend_strength = "strong" if price_change < -0.15 else "medium"
        else:
            overall_trend = "neutral"
            trend_strength = "weak"
        
        # Determine trend duration based on consistency
        trend_duration = self._assess_trend_duration(stock_data)
        
        # Calculate trend confidence based on indicator alignment
        trend_confidence = self._calculate_trend_confidence(indicators, overall_trend)
        
        return {
            "overall_trend": overall_trend,
            "trend_strength": trend_strength,
            "trend_duration": trend_duration,
            "trend_confidence": int(trend_confidence * 100)
        }
    
    def _analyze_volume(self, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume trends and confirmation"""
        if 'volume' not in stock_data.columns:
            return {
                "volume_trend": "unknown",
                "volume_confirmation": "unavailable", 
                "volume_significance": "unknown",
                "institutional_activity": "unknown"
            }
        
        volume = stock_data['volume'].values
        recent_volume = np.mean(volume[-10:])
        avg_volume = np.mean(volume)
        
        # Determine volume trend
        if recent_volume > avg_volume * 1.2:
            volume_trend = "increasing"
            volume_significance = "high"
        elif recent_volume < avg_volume * 0.8:
            volume_trend = "decreasing"
            volume_significance = "low"
        else:
            volume_trend = "stable"
            volume_significance = "medium"
        
        # Assess volume confirmation with price
        price_up_days = sum(1 for i in range(1, len(stock_data)) if stock_data['close'].iloc[i] > stock_data['close'].iloc[i-1])
        total_days = len(stock_data) - 1
        
        if price_up_days / total_days > 0.6 and volume_trend == "increasing":
            volume_confirmation = "confirmed"
        elif price_up_days / total_days < 0.4 and volume_trend == "increasing":
            volume_confirmation = "diverging"
        else:
            volume_confirmation = "neutral"
        
        # Assess institutional activity based on volume spikes
        volume_spikes = sum(1 for v in volume if v > avg_volume * 2)
        institutional_activity = "high" if volume_spikes > len(volume) * 0.05 else "medium" if volume_spikes > 0 else "low"
        
        return {
            "volume_trend": volume_trend,
            "volume_confirmation": volume_confirmation,
            "volume_significance": volume_significance,
            "institutional_activity": institutional_activity
        }
    
    def _analyze_momentum(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze momentum indicators"""
        if not indicators:
            return {
                "macd_signal": "neutral",
                "rsi_status": "neutral", 
                "stochastic_signal": "neutral",
                "momentum_alignment": "unknown"
            }
        
        # MACD analysis
        macd_signal = "neutral"
        if 'macd' in indicators and 'macd_signal' in indicators:
            macd = indicators['macd']
            signal = indicators['macd_signal']
            if isinstance(macd, (list, np.ndarray)) and isinstance(signal, (list, np.ndarray)):
                if len(macd) > 0 and len(signal) > 0:
                    if macd[-1] > signal[-1] and macd[-1] > 0:
                        macd_signal = "bullish"
                    elif macd[-1] < signal[-1] and macd[-1] < 0:
                        macd_signal = "bearish"
        
        # RSI analysis
        rsi_status = "neutral"
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if isinstance(rsi, (list, np.ndarray)) and len(rsi) > 0:
                current_rsi = rsi[-1]
                if current_rsi > 70:
                    rsi_status = "overbought"
                elif current_rsi < 30:
                    rsi_status = "oversold"
        
        # Stochastic analysis (simplified)
        stochastic_signal = "neutral"
        
        # Determine momentum alignment
        bullish_signals = sum([1 for signal in [macd_signal, rsi_status, stochastic_signal] if "bullish" in signal or "oversold" in signal])
        bearish_signals = sum([1 for signal in [macd_signal, rsi_status, stochastic_signal] if "bearish" in signal or "overbought" in signal])
        
        if bullish_signals > bearish_signals:
            momentum_alignment = "aligned" if bullish_signals >= 2 else "neutral"
        elif bearish_signals > bullish_signals:
            momentum_alignment = "conflicting" if bearish_signals >= 2 else "neutral"  
        else:
            momentum_alignment = "neutral"
        
        return {
            "macd_signal": macd_signal,
            "rsi_status": rsi_status,
            "stochastic_signal": stochastic_signal,
            "momentum_alignment": momentum_alignment
        }
    
    def _analyze_support_resistance(self, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """Identify key support and resistance levels"""
        highs = stock_data['high'].values
        lows = stock_data['low'].values
        
        # Find recent significant levels
        window = min(10, len(highs) // 4)
        
        support_levels = []
        resistance_levels = []
        
        # Find local maxima (resistance)
        for i in range(window, len(highs) - window):
            if highs[i] == np.max(highs[i-window:i+window+1]):
                resistance_levels.append(round(highs[i], 2))
        
        # Find local minima (support) 
        for i in range(window, len(lows) - window):
            if lows[i] == np.min(lows[i-window:i+window+1]):
                support_levels.append(round(lows[i], 2))
        
        # Keep most significant levels
        resistance_levels = sorted(list(set(resistance_levels)), reverse=True)[:3]
        support_levels = sorted(list(set(support_levels)), reverse=True)[:3]
        
        # Assess strength based on number of touches
        support_strength = "strong" if len(support_levels) >= 2 else "medium" if len(support_levels) == 1 else "weak"
        resistance_strength = "strong" if len(resistance_levels) >= 2 else "medium" if len(resistance_levels) == 1 else "weak"
        
        return {
            "key_support_levels": support_levels,
            "key_resistance_levels": resistance_levels,
            "support_strength": support_strength,
            "resistance_strength": resistance_strength
        }
    
    def _assess_risk(self, stock_data: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall risk levels"""
        # Calculate volatility
        returns = stock_data['close'].pct_change().dropna()
        volatility = returns.std()
        
        # Determine risk level
        if volatility > 0.03:
            risk_level = "high"
        elif volatility > 0.015:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Identify key risk factors
        risk_factors = []
        
        if volatility > 0.025:
            risk_factors.append("High volatility")
        
        current_price = stock_data['close'].iloc[-1]
        if 'rsi' in indicators and indicators['rsi'][-1] > 80:
            risk_factors.append("Extreme overbought conditions")
        elif 'rsi' in indicators and indicators['rsi'][-1] < 20:
            risk_factors.append("Extreme oversold conditions")
        
        # Calculate simple stop loss levels
        recent_low = stock_data['low'].tail(20).min()
        recent_high = stock_data['high'].tail(20).max()
        
        stop_loss_levels = [
            round(recent_low * 0.98, 2),  # 2% below recent low
            round(current_price * 0.95, 2)  # 5% below current price
        ]
        
        # Simple risk-reward calculation
        potential_gain = (recent_high - current_price) / current_price
        potential_loss = (current_price - recent_low) / current_price
        risk_reward_ratio = round(potential_gain / potential_loss, 2) if potential_loss > 0 else 0.0
        
        return {
            "risk_level": risk_level,
            "key_risk_factors": risk_factors,
            "stop_loss_levels": stop_loss_levels,
            "risk_reward_ratio": risk_reward_ratio
        }
    
    def _calculate_confidence(self, trend_analysis: Dict, volume_analysis: Dict, momentum_analysis: Dict) -> int:
        """Calculate overall confidence score"""
        confidence = 50  # Base confidence
        
        # Adjust for trend strength
        if trend_analysis.get('trend_strength') == 'strong':
            confidence += 20
        elif trend_analysis.get('trend_strength') == 'medium':
            confidence += 10
        
        # Adjust for volume confirmation
        if volume_analysis.get('volume_confirmation') == 'confirmed':
            confidence += 15
        elif volume_analysis.get('volume_confirmation') == 'diverging':
            confidence -= 10
        
        # Adjust for momentum alignment
        if momentum_analysis.get('momentum_alignment') == 'aligned':
            confidence += 15
        elif momentum_analysis.get('momentum_alignment') == 'conflicting':
            confidence -= 15
        
        return max(0, min(100, confidence))
    
    def _assess_trend_duration(self, stock_data: pd.DataFrame) -> str:
        """Assess how long the current trend has been in place"""
        prices = stock_data['close'].tail(50)  # Look at last 50 periods
        
        if len(prices) < 10:
            return "short_term"
        
        # Simple trend duration assessment
        recent_trend_periods = 0
        current_direction = None
        
        for i in range(1, len(prices)):
            if prices.iloc[i] > prices.iloc[i-1]:
                direction = "up"
            else:
                direction = "down"
            
            if current_direction == direction:
                recent_trend_periods += 1
            else:
                current_direction = direction
                recent_trend_periods = 1
        
        if recent_trend_periods > 30:
            return "long_term"
        elif recent_trend_periods > 10:
            return "medium_term"
        else:
            return "short_term"
    
    def _calculate_trend_confidence(self, indicators: Dict[str, Any], trend_direction: str) -> float:
        """Calculate confidence in trend direction based on indicators"""
        if not indicators:
            return 0.5
        
        confidence_factors = []
        
        # Check moving average alignment
        if 'sma_20' in indicators and 'sma_50' in indicators:
            sma_20 = indicators['sma_20']
            sma_50 = indicators['sma_50']
            
            if isinstance(sma_20, (list, np.ndarray)) and isinstance(sma_50, (list, np.ndarray)):
                if len(sma_20) > 0 and len(sma_50) > 0:
                    if trend_direction == "bullish" and sma_20[-1] > sma_50[-1]:
                        confidence_factors.append(0.8)
                    elif trend_direction == "bearish" and sma_20[-1] < sma_50[-1]:
                        confidence_factors.append(0.8)
                    else:
                        confidence_factors.append(0.4)
        
        # Check RSI alignment
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if isinstance(rsi, (list, np.ndarray)) and len(rsi) > 0:
                current_rsi = rsi[-1]
                if trend_direction == "bullish" and 30 < current_rsi < 80:
                    confidence_factors.append(0.7)
                elif trend_direction == "bearish" and 20 < current_rsi < 70:
                    confidence_factors.append(0.7)
                else:
                    confidence_factors.append(0.3)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _assess_analysis_quality(self, indicators: Dict[str, Any]) -> str:
        """Assess quality of analysis based on available indicators"""
        if not indicators:
            return "low"
        
        indicator_count = len([k for k, v in indicators.items() if v is not None])
        
        if indicator_count >= 5:
            return "high"
        elif indicator_count >= 3:
            return "medium"
        else:
            return "low"
    
    def _determine_primary_signal(self, trend_analysis: Dict, momentum_analysis: Dict) -> str:
        """Determine primary trading signal"""
        trend = trend_analysis.get('overall_trend', 'neutral')
        strength = trend_analysis.get('trend_strength', 'weak')
        momentum = momentum_analysis.get('momentum_alignment', 'neutral')
        
        if trend == "bullish" and strength in ["strong", "medium"] and momentum == "aligned":
            return "strong_bullish"
        elif trend == "bullish" and strength == "medium":
            return "bullish"
        elif trend == "bearish" and strength in ["strong", "medium"] and momentum == "aligned":
            return "strong_bearish"
        elif trend == "bearish" and strength == "medium":
            return "bearish"
        else:
            return "neutral"
    
    def _identify_patterns(self, stock_data: pd.DataFrame, indicators: Dict[str, Any]) -> List[str]:
        """Identify basic chart patterns"""
        patterns = []
        
        # Simple pattern identification
        prices = stock_data['close'].tail(20)
        if len(prices) >= 10:
            recent_high = prices.max()
            recent_low = prices.min()
            current_price = prices.iloc[-1]
            
            # Check for breakout patterns
            if current_price > recent_high * 0.98:
                patterns.append("potential_breakout_upward")
            elif current_price < recent_low * 1.02:
                patterns.append("potential_breakdown")
            
            # Check for consolidation
            price_range = (recent_high - recent_low) / recent_low
            if price_range < 0.05:
                patterns.append("consolidation")
        
        return patterns
    
    def _generate_trading_recommendations(self, trend_analysis: Dict, risk_assessment: Dict) -> Dict[str, Any]:
        """Generate basic trading recommendations"""
        trend = trend_analysis.get('overall_trend', 'neutral')
        strength = trend_analysis.get('trend_strength', 'weak')
        risk_level = risk_assessment.get('risk_level', 'medium')
        
        if trend == "bullish" and strength == "strong" and risk_level == "low":
            action = "buy"
            position_size = "large"
        elif trend == "bullish" and strength == "medium":
            action = "buy"
            position_size = "medium"
        elif trend == "bearish" and strength == "strong" and risk_level == "low":
            action = "sell"
            position_size = "large"
        elif trend == "bearish" and strength == "medium":
            action = "sell"
            position_size = "medium"
        else:
            action = "hold"
            position_size = "small"
        
        return {
            "recommended_action": action,
            "position_size": position_size,
            "risk_level": risk_level,
            "confidence": trend_analysis.get('trend_confidence', 50)
        }