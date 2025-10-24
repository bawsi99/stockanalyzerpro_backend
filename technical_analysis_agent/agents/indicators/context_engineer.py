#!/usr/bin/env python3
"""
Indicator Context Engineer

Handles context engineering specifically for indicator agents.
Extracted from backend/gemini/context_engineer.py but focused only on indicator needs.

Key features:
- Market regime detection (trending vs ranging)
- Volume strength classification  
- Momentum vs trend relationship analysis
- Conflict detection and resolution
- Context formatting for indicator analysis
"""

import json
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class IndicatorMarketRegime:
    """Market regime classification for indicator analysis."""
    trend_regime: str  # strong_uptrend, uptrend, downtrend, strong_downtrend, sideways, transitional
    volatility_regime: str  # high_volatility, medium_volatility, low_volatility
    volume_regime: str  # high_volume, above_average_volume, normal_volume, low_volume
    momentum_regime: str  # strong_momentum, moderate_momentum, weak_momentum
    composite_regime: str  # trending_momentum, choppy_range, steady_trend, breakout_mode, mixed


class IndicatorContextEngineer:
    """
    Context engineering specifically for indicator analysis agents.
    
    Handles:
    - Market regime detection and analysis
    - Indicator conflict detection and resolution
    - Context formatting for LLM consumption
    - Volume strength classification
    - Momentum vs trend relationship analysis
    """
    
    # Conflict priority weights for indicator analysis
    CONFLICT_WEIGHTS = {
        'timeframe_conflicts': {
            'base_weight': 1.0,  # Highest priority
            'multipliers': {
                'Short-term bullish signals against long-term bearish trend': 1.2,
                'Short-term bearish signals against long-term bullish trend': 1.2,
                'Price significantly below SMA200 while RSI bullish': 1.1,
                'Mixed moving average alignment signals': 0.9
            }
        },
        'momentum_vs_trend': {
            'base_weight': 0.8,  # High priority
            'multipliers': {
                'RSI extremely overbought with strong MACD bullish': 1.3,
                'RSI extremely oversold with strong MACD bearish': 1.3,
                'RSI overbought while MACD bullish': 0.9,
                'RSI oversold while MACD bearish': 0.9
            }
        },
        'volume_conflicts': {
            'base_weight': 0.7,  # Medium-high priority
            'multipliers': {
                'Very strong momentum signals with low volume': 1.2,
                'Very strong momentum signals with below average volume': 1.1,
                'Moderate momentum signals with low volume confirmation': 0.8
            }
        },
        'signal_strength': {
            'base_weight': 0.5,  # Medium priority
            'multipliers': {
                'Strong MACD signals during low volume conditions': 0.9,
                'Weak MACD signals during high volume conditions': 0.7,
                'Extreme RSI with weak momentum confirmation': 0.8
            }
        }
    }
    
    def __init__(self):
        pass
    
    def build_indicator_context(self, 
                              curated_data: Dict[str, Any], 
                              symbol: str, 
                              timeframe: str, 
                              knowledge_context: str = "") -> str:
        """
        Build complete context for indicator analysis.
        
        Args:
            curated_data: Curated indicator data from agents
            symbol: Stock symbol
            timeframe: Analysis timeframe
            knowledge_context: Additional context (MTF, sector, etc.)
            
        Returns:
            Formatted context string for LLM
        """
        try:
            key_indicators = curated_data.get('key_indicators', {})
            critical_levels = curated_data.get('critical_levels', {})
            detected_conflicts = curated_data.get('detected_conflicts', {})
            
            # Format confidence values as percentages
            key_indicators_formatted = self._format_confidence_values(key_indicators)
            
            # Build context sections
            context_parts = [
                f"**Symbol**: {symbol} | **Timeframe**: {timeframe}",
                "",
                "## Technical Data:",
                json.dumps(key_indicators_formatted, indent=1),
                "",
                "## Levels:",
                json.dumps(critical_levels, indent=1),
                ""
            ]
            
            # Add conflict analysis if any conflicts detected
            if detected_conflicts.get('has_conflicts', False):
                conflict_section = self._format_indicator_conflicts(detected_conflicts)
                context_parts.append(conflict_section)
                context_parts.append("")
            else:
                context_parts.append("## Signal Conflicts: None detected")
                context_parts.append("")
            
            # Add additional knowledge context
            if knowledge_context and knowledge_context.strip():
                context_parts.append(knowledge_context.strip())
                context_parts.append("")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            raise RuntimeError(f"Failed to build indicator context: {e}")
    
    def detect_market_regime(self, data: Dict[str, Any]) -> IndicatorMarketRegime:
        """
        Detect market regime for context-aware indicator analysis.
        
        Args:
            data: Technical indicator data
            
        Returns:
            IndicatorMarketRegime object with regime classifications
        """
        # Safe numeric value extraction
        price_to_sma200 = self._safe_numeric_value(data.get('price_to_sma_200'), 0)
        sma_alignment = self._safe_numeric_value(data.get('sma_20_to_sma_50'), 0)
        volume_ratio = self._safe_numeric_value(data.get('volume_ratio'), 1.0)
        rsi = self._safe_numeric_value(data.get('rsi'), 50)
        macd_histogram = abs(self._safe_numeric_value(data.get('macd_histogram'), 0))
        
        # Trend regime detection
        if price_to_sma200 > 0.10 and sma_alignment > 0.05:
            trend_regime = "strong_uptrend"
        elif price_to_sma200 > 0.03 and sma_alignment > 0.01:
            trend_regime = "uptrend"
        elif price_to_sma200 < -0.10 and sma_alignment < -0.05:
            trend_regime = "strong_downtrend"
        elif price_to_sma200 < -0.03 and sma_alignment < -0.01:
            trend_regime = "downtrend"
        elif abs(price_to_sma200) < 0.03 and abs(sma_alignment) < 0.02:
            trend_regime = "sideways"
        else:
            trend_regime = "transitional"
        
        # Volatility regime detection
        volatility_score = 0
        if volume_ratio > 2.0: volatility_score += 2
        elif volume_ratio > 1.5: volatility_score += 1
        if rsi > 80 or rsi < 20: volatility_score += 2
        elif rsi > 70 or rsi < 30: volatility_score += 1
        if macd_histogram > 5: volatility_score += 2
        elif macd_histogram > 2: volatility_score += 1
        
        if volatility_score >= 4:
            volatility_regime = "high_volatility"
        elif volatility_score >= 2:
            volatility_regime = "medium_volatility"
        else:
            volatility_regime = "low_volatility"
        
        # Volume regime
        volume_regime = self._classify_volume_strength(volume_ratio)
        
        # Momentum regime
        if abs(macd_histogram) > 3 and data.get('rsi_trend') in ['up', 'down']:
            momentum_regime = "strong_momentum"
        elif abs(macd_histogram) > 1 or data.get('rsi_trend') in ['up', 'down']:
            momentum_regime = "moderate_momentum"
        else:
            momentum_regime = "weak_momentum"
        
        # Composite regime determination
        if trend_regime in ['strong_uptrend', 'strong_downtrend'] and momentum_regime == 'strong_momentum':
            composite_regime = "trending_momentum"
        elif trend_regime in ['sideways', 'transitional'] and volatility_regime == 'high_volatility':
            composite_regime = "choppy_range"
        elif trend_regime in ['uptrend', 'downtrend'] and volatility_regime == 'low_volatility':
            composite_regime = "steady_trend"
        elif volume_regime == 'high_volume' and momentum_regime == 'strong_momentum':
            composite_regime = "breakout_mode"
        else:
            composite_regime = "mixed"
        
        return IndicatorMarketRegime(
            trend_regime=trend_regime,
            volatility_regime=volatility_regime,
            volume_regime=volume_regime,
            momentum_regime=momentum_regime,
            composite_regime=composite_regime
        )
    
    def detect_indicator_conflicts(self, key_indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect conflicts between different indicators.
        
        Args:
            key_indicators: Key indicator data
            
        Returns:
            Conflict analysis with categories, severity, and resolution suggestions
        """
        try:
            conflicts = {
                'has_conflicts': False,
                'conflict_count': 0,
                'conflict_severity': 'none',
                'conflict_list': [],
                'conflict_categories': {},
                'weighted_score': 0.0
            }
            
            # Extract indicator data safely
            trend_indicators = key_indicators.get('trend_indicators', {})
            momentum_indicators = key_indicators.get('momentum_indicators', {})
            volume_indicators = key_indicators.get('volume_indicators', {})
            
            # Extract values for conflict analysis
            data = {
                'price_to_sma_200': trend_indicators.get('price_to_sma_200'),
                'sma_20_to_sma_50': trend_indicators.get('sma_20_to_sma_50'),
                'rsi': momentum_indicators.get('rsi_current'),
                'macd_histogram': momentum_indicators.get('macd', {}).get('histogram') if isinstance(momentum_indicators.get('macd'), dict) else None,
                'volume_ratio': volume_indicators.get('volume_ratio'),
                'trend_direction': trend_indicators.get('direction'),
                'momentum_direction': momentum_indicators.get('direction'),
                'rsi_status': momentum_indicators.get('rsi_status')
            }
            
            # Detect different types of conflicts
            conflict_categories = {}
            
            # 1. Momentum vs Trend conflicts
            momentum_trend_conflicts = self._detect_momentum_trend_conflicts(data)
            if momentum_trend_conflicts:
                conflict_categories['momentum_vs_trend'] = momentum_trend_conflicts
            
            # 2. Volume confirmation conflicts  
            volume_conflicts = self._detect_volume_conflicts(data)
            if volume_conflicts:
                conflict_categories['volume_conflicts'] = volume_conflicts
            
            # 3. Timeframe conflicts (short vs long term)
            timeframe_conflicts = self._detect_timeframe_conflicts(data)
            if timeframe_conflicts:
                conflict_categories['timeframe_conflicts'] = timeframe_conflicts
            
            # 4. Signal strength conflicts
            signal_strength_conflicts = self._detect_signal_strength_conflicts(data)
            if signal_strength_conflicts:
                conflict_categories['signal_strength'] = signal_strength_conflicts
            
            # Calculate weighted severity
            if conflict_categories:
                weighted_score, severity = self._calculate_weighted_severity(conflict_categories)
                all_conflicts = []
                for category_conflicts in conflict_categories.values():
                    all_conflicts.extend(category_conflicts)
                
                conflicts.update({
                    'has_conflicts': True,
                    'conflict_count': len(all_conflicts),
                    'conflict_severity': severity,
                    'conflict_list': all_conflicts,
                    'conflict_categories': conflict_categories,
                    'weighted_score': weighted_score
                })
            
            return conflicts
            
        except Exception as e:
            # Return safe fallback on error
            return {
                'has_conflicts': False,
                'conflict_count': 0,
                'conflict_severity': 'none',
                'conflict_list': [f"Error in conflict detection: {str(e)}"],
                'conflict_categories': {},
                'weighted_score': 0.0
            }
    
    def _format_confidence_values(self, key_indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Convert decimal confidence values to percentage strings."""
        try:
            import copy
            formatted = copy.deepcopy(key_indicators)
            
            # Format trend indicators confidence
            trend_indicators = formatted.get('trend_indicators')
            if isinstance(trend_indicators, dict) and 'confidence' in trend_indicators:
                confidence = trend_indicators['confidence']
                trend_indicators['confidence'] = self._to_percentage_string(confidence)
            
            # Format momentum indicators confidence  
            momentum_indicators = formatted.get('momentum_indicators')
            if isinstance(momentum_indicators, dict) and 'confidence' in momentum_indicators:
                confidence = momentum_indicators['confidence']
                momentum_indicators['confidence'] = self._to_percentage_string(confidence)
            
            return formatted
            
        except Exception:
            return key_indicators
    
    def _format_indicator_conflicts(self, detected_conflicts: Dict[str, Any]) -> str:
        """Format conflicts in a streamlined, actionable manner for indicators."""
        if not detected_conflicts.get('has_conflicts', False):
            return "## Signal Conflicts: None detected"
        
        conflict_count = detected_conflicts.get('conflict_count', 0)
        severity = detected_conflicts.get('conflict_severity', 'low')
        
        section = f"## Signal Conflicts ({conflict_count} detected, {severity.upper()} severity):\n"
        
        # Get top conflicts only
        conflict_categories = detected_conflicts.get('conflict_categories', {})
        sorted_categories = sorted(
            conflict_categories.items(), 
            key=lambda x: self.CONFLICT_WEIGHTS.get(x[0], {'base_weight': 0})['base_weight'], 
            reverse=True
        )[:2]  # Only top 2 categories
        
        for category, conflicts in sorted_categories:
            if conflicts:
                section += f"\n**{category.replace('_', ' ').title()}:**\n"
                for conflict in conflicts[:1]:  # Only top conflict per category
                    section += f"- {conflict}\n"
        
        # Simplified resolution instructions based on severity
        if severity in ['high', 'critical']:
            section += "\n**Resolution:** Wait for clearer signals or reduce position size"
        else:
            section += "\n**Resolution:** Prioritize trend indicators over momentum signals"
        
        return section
    
    def _detect_momentum_trend_conflicts(self, data: Dict[str, Any]) -> list:
        """Detect conflicts between momentum and trend indicators."""
        conflicts = []
        
        rsi = self._safe_numeric_value(data.get('rsi'), 50)
        macd_histogram = self._safe_numeric_value(data.get('macd_histogram'), 0)
        trend_direction = data.get('trend_direction', 'neutral')
        momentum_direction = data.get('momentum_direction', 'neutral')
        
        # RSI vs MACD conflicts
        if rsi > 70 and macd_histogram < -1:
            conflicts.append("RSI overbought while MACD bearish")
        elif rsi < 30 and macd_histogram > 1:
            conflicts.append("RSI oversold while MACD bullish")
        
        # Extreme RSI vs strong MACD
        if rsi > 80 and macd_histogram > 2:
            conflicts.append("RSI extremely overbought with strong MACD bullish")
        elif rsi < 20 and macd_histogram < -2:
            conflicts.append("RSI extremely oversold with strong MACD bearish")
        
        # Trend vs momentum direction conflicts
        if trend_direction == 'bullish' and momentum_direction == 'bearish':
            conflicts.append("Bullish trend with bearish momentum")
        elif trend_direction == 'bearish' and momentum_direction == 'bullish':
            conflicts.append("Bearish trend with bullish momentum")
        
        return conflicts
    
    def _detect_volume_conflicts(self, data: Dict[str, Any]) -> list:
        """Detect volume confirmation conflicts."""
        conflicts = []
        
        volume_ratio = self._safe_numeric_value(data.get('volume_ratio'), 1.0)
        macd_histogram = abs(self._safe_numeric_value(data.get('macd_histogram'), 0))
        
        volume_strength = self._classify_volume_strength(volume_ratio)
        
        # Strong momentum with weak volume
        if macd_histogram > 3 and volume_strength in ['low', 'below_average']:
            conflicts.append(f"Very strong momentum signals with {volume_strength.replace('_', ' ')} volume")
        elif macd_histogram > 1 and volume_strength == 'low':
            conflicts.append("Moderate momentum signals with low volume confirmation")
        
        # High volume with weak momentum (possible distribution)
        elif volume_strength in ['high', 'very_high'] and macd_histogram < 0.5:
            conflicts.append(f"{volume_strength.replace('_', ' ').title()} volume with weak momentum (possible distribution)")
        
        return conflicts
    
    def _detect_timeframe_conflicts(self, data: Dict[str, Any]) -> list:
        """Detect conflicts between short-term and long-term signals."""
        conflicts = []
        
        price_to_sma200 = self._safe_numeric_value(data.get('price_to_sma_200'), 0)
        sma_alignment = self._safe_numeric_value(data.get('sma_20_to_sma_50'), 0)
        rsi = self._safe_numeric_value(data.get('rsi'), 50)
        
        # Price significantly below long-term trend but short-term bullish
        if price_to_sma200 < -0.05 and rsi > 60:
            conflicts.append("Price significantly below SMA200 while RSI bullish")
        
        # Mixed moving average alignment
        if price_to_sma200 > 0.02 and sma_alignment < -0.02:
            conflicts.append("Mixed moving average alignment signals")
        elif price_to_sma200 < -0.02 and sma_alignment > 0.02:
            conflicts.append("Mixed moving average alignment signals")
        
        return conflicts
    
    def _detect_signal_strength_conflicts(self, data: Dict[str, Any]) -> list:
        """Detect conflicts in signal strength consistency."""
        conflicts = []
        
        volume_ratio = self._safe_numeric_value(data.get('volume_ratio'), 1.0)
        macd_histogram = self._safe_numeric_value(data.get('macd_histogram'), 0)
        rsi = self._safe_numeric_value(data.get('rsi'), 50)
        
        volume_strength = self._classify_volume_strength(volume_ratio)
        
        # Strong MACD during low volume
        if abs(macd_histogram) > 2 and volume_strength == 'low':
            conflicts.append("Strong MACD signals during low volume conditions")
        
        # Weak MACD during high volume
        elif abs(macd_histogram) < 0.5 and volume_strength in ['high', 'very_high']:
            conflicts.append("Weak MACD signals during high volume conditions")
        
        # Extreme RSI with weak momentum confirmation
        if (rsi > 75 or rsi < 25) and abs(macd_histogram) < 1:
            conflicts.append("Extreme RSI with weak momentum confirmation")
        
        return conflicts
    
    def _classify_volume_strength(self, volume_ratio: float) -> str:
        """Classify volume strength based on ratio to average volume."""
        if volume_ratio is None:
            return "unknown"
        
        volume_ratio = self._safe_numeric_value(volume_ratio, 1.0)
        
        if volume_ratio >= 2.0:
            return "very_high"      # 2x+ average volume
        elif volume_ratio >= 1.5:
            return "high"           # 1.5-2x average volume
        elif volume_ratio >= 1.1:
            return "above_average"  # 1.1-1.5x average volume
        elif volume_ratio >= 0.8:
            return "average"        # 0.8-1.1x average volume
        elif volume_ratio >= 0.6:
            return "below_average"  # 0.6-0.8x average volume
        else:
            return "low"            # <0.6x average volume
    
    def _calculate_weighted_severity(self, conflict_categories: Dict[str, list]) -> tuple:
        """Calculate weighted severity score and classification."""
        total_weighted_score = 0.0
        max_individual_score = 0.0
        
        for category, conflicts in conflict_categories.items():
            if not conflicts:
                continue
            
            category_config = self.CONFLICT_WEIGHTS.get(category, {'base_weight': 0.2, 'multipliers': {}})
            base_weight = category_config['base_weight']
            multipliers = category_config['multipliers']
            
            for conflict in conflicts:
                multiplier = multipliers.get(conflict, 1.0)
                conflict_score = base_weight * multiplier
                
                total_weighted_score += conflict_score
                max_individual_score = max(max_individual_score, conflict_score)
        
        # Determine severity based on scores
        if max_individual_score >= 1.0 or total_weighted_score >= 2.5:
            return total_weighted_score, "critical"
        elif max_individual_score >= 0.7 or total_weighted_score >= 1.8:
            return total_weighted_score, "high"
        elif max_individual_score >= 0.5 or total_weighted_score >= 1.2:
            return total_weighted_score, "medium"
        elif total_weighted_score >= 0.5:
            return total_weighted_score, "low"
        else:
            return total_weighted_score, "none"
    
    def _safe_numeric_value(self, value, default=0):
        """Safely convert value to float, handling None, strings, and invalid values."""
        if value is None:
            return default
        
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        return default
    
    def _to_percentage_string(self, value) -> str:
        """Convert numeric value to percentage string."""
        try:
            f = float(value)
            if 0.0 <= f <= 1.0:
                f = f * 100.0
            return f"{f:.2f}%"
        except Exception:
            return str(value)


# Global instance for indicators
indicator_context_engineer = IndicatorContextEngineer()