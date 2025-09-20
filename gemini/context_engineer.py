import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class AnalysisType(Enum):
    INDICATOR_SUMMARY = "indicator_summary"
    VOLUME_ANALYSIS = "volume_analysis"
    REVERSAL_PATTERNS = "reversal_patterns"
    CONTINUATION_LEVELS = "continuation_levels"
    COMPREHENSIVE_OVERVIEW = "comprehensive_overview"
    FINAL_DECISION = "final_decision"

@dataclass
class ContextConfig:
    """Configuration for context engineering"""
    max_tokens: int = 8000
    prioritize_conflicts: bool = True
    include_mathematical_validation: bool = True
    compress_indicators: bool = True
    focus_on_recent_data: bool = True

class ContextEngineer:
    """
    Handles context curation, structuring, and optimization for LLM calls.
    Implements context engineering principles to improve analysis quality and reduce token usage.
    """
    
    # Conflict priority weights for enhanced analysis
    CONFLICT_WEIGHTS = {
        'timeframe_conflicts': {
            'base_weight': 1.0,  # Highest priority - affects trade validity
            'multipliers': {
                'Short-term bullish signals against long-term bearish trend': 1.2,
                'Short-term bearish signals against long-term bullish trend': 1.2,
                'Price significantly below SMA200 while RSI bullish': 1.1,
                'Mixed moving average alignment signals': 0.9
            }
        },
        'momentum_vs_trend': {
            'base_weight': 0.8,  # High priority - momentum disagreements
            'multipliers': {
                'RSI extremely overbought with strong MACD bullish': 1.3,
                'RSI extremely oversold with strong MACD bearish': 1.3,
                'RSI overbought while MACD bullish': 0.9,
                'RSI oversold while MACD bearish': 0.9
            }
        },
        'volume_conflicts': {
            'base_weight': 0.7,  # Medium-high priority - execution confirmation
            'multipliers': {
                'Very strong momentum signals with low volume': 1.2,
                'Very strong momentum signals with below average volume': 1.1,
                'Moderate momentum signals with low volume confirmation': 0.8
            }
        },
        'signal_strength': {
            'base_weight': 0.5,  # Medium priority - context mismatches
            'multipliers': {
                'Strong MACD signals during low volume conditions': 0.9,
                'Weak MACD signals during high volume conditions': 0.7,
                'Extreme RSI with weak momentum confirmation': 0.8
            }
        },
        'market_regime': {
            'base_weight': 0.4,  # Lower priority - contextual information
            'multipliers': {
                'Ranging market with RSI overbought and continued bullish momentum': 0.8,
                'Ranging market with RSI oversold and continued bearish momentum': 0.8,
                'High volatility environment with neutral indicators': 0.6
            }
        },
        'statistical_divergences': {
            'base_weight': 0.3,  # Lowest priority - often noise
            'multipliers': {
                'MACD above signal line while RSI extremely overbought': 0.9,
                'MACD below signal line while RSI extremely oversold': 0.9,
                'RSI trending down while MACD bullish': 0.7,
                'RSI trending up while MACD bearish': 0.7
            }
        }
    }
    
    def __init__(self, config: ContextConfig = None):
        self.config = config or ContextConfig()
    
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
    
    def _safe_boolean_value(self, value, default=False):
        """Safely convert value to boolean, handling None, strings, and invalid values."""
        if value is None:
            return default
        
        if isinstance(value, bool):
            return value
        
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        
        if isinstance(value, (int, float)):
            return bool(value)
        
        return default
    
    def _classify_volume_strength(self, volume_ratio):
        """Enhanced volume classification with nuanced tiers."""
        if volume_ratio is None:
            return "unknown"
        
        # Convert to numeric if needed
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
    
    def _calculate_weighted_severity(self, conflict_details):
        """Calculate severity using weighted priority system."""
        categories = conflict_details.get('conflict_categories', {})
        total_weighted_score = 0.0
        max_individual_score = 0.0
        
        for category, conflicts in categories.items():
            if not conflicts:
                continue
                
            category_config = self.CONFLICT_WEIGHTS.get(category, {'base_weight': 0.2, 'multipliers': {}})
            base_weight = category_config['base_weight']
            multipliers = category_config['multipliers']
            
            for conflict in conflicts:
                # Get specific multiplier for this conflict, or use 1.0
                multiplier = multipliers.get(conflict, 1.0)
                conflict_score = base_weight * multiplier
                
                total_weighted_score += conflict_score
                max_individual_score = max(max_individual_score, conflict_score)
        
        # Determine severity based on both total score and max individual score
        if max_individual_score >= 1.0 or total_weighted_score >= 2.5:
            return "critical", total_weighted_score
        elif max_individual_score >= 0.7 or total_weighted_score >= 1.8:
            return "high", total_weighted_score
        elif max_individual_score >= 0.5 or total_weighted_score >= 1.2:
            return "medium", total_weighted_score
        elif total_weighted_score >= 0.5:
            return "low", total_weighted_score
        else:
            return "none", total_weighted_score
    
    def _get_priority_level(self, category):
        """Get human-readable priority level for a conflict category."""
        weight = self.CONFLICT_WEIGHTS.get(category, {'base_weight': 0})['base_weight']
        
        if weight >= 0.8:
            return "CRITICAL"
        elif weight >= 0.6:
            return "HIGH"
        elif weight >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _detect_comprehensive_market_regime(self, data):
        """Advanced market regime detection with volatility context."""
        
        # Use safe numeric conversion for all values
        price_to_sma200 = self._safe_numeric_value(data.get('price_to_sma200'), 0)
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
        
        # Volatility regime detection (approximated)
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
        if volume_ratio > 2.0:
            volume_regime = "high_volume"
        elif volume_ratio > 1.3:
            volume_regime = "above_average_volume"
        elif volume_ratio > 0.7:
            volume_regime = "normal_volume"
        else:
            volume_regime = "low_volume"
        
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
        
        return {
            'trend_regime': trend_regime,
            'volatility_regime': volatility_regime,
            'volume_regime': volume_regime,
            'momentum_regime': momentum_regime,
            'composite_regime': composite_regime
        }
    
    def _apply_regime_specific_conflicts(self, data, market_regime):
        """Apply different conflict detection rules based on market regime."""
        conflicts = []
        regime = market_regime['composite_regime']
        rsi = self._safe_numeric_value(data.get('rsi'), 50)
        macd_trend = data.get('macd_trend', 'neutral')
        volume_ratio = self._safe_numeric_value(data.get('volume_ratio'), 1.0)
        
        # Regime-specific conflict rules
        if regime == "trending_momentum":
            # In strong trends, overbought/oversold less reliable
            if rsi > 80:
                conflicts.append("RSI overbought in strong trending market (momentum may continue)")
            elif rsi < 20:
                conflicts.append("RSI oversold in strong trending market (momentum may continue)")
        
        elif regime == "choppy_range":
            # In ranging markets, extreme RSI more reliable for reversals
            if rsi > 70 and macd_trend == 'bullish':
                conflicts.append("RSI overbought with bullish MACD in ranging market (likely reversal)")
            elif rsi < 30 and macd_trend == 'bearish':
                conflicts.append("RSI oversold with bearish MACD in ranging market (likely reversal)")
        
        elif regime == "breakout_mode":
            # In breakouts, volume is critical for confirmation
            if volume_ratio < 1.5:
                conflicts.append("Potential breakout without sufficient volume confirmation")
        
        elif regime == "steady_trend":
            # In steady trends, look for momentum divergences
            if data.get('rsi_trend') and data.get('rsi_trend') != macd_trend:
                conflicts.append("Momentum divergence in steady trend (potential reversal signal)")
        
        return conflicts
    
    def curate_indicators(self, indicators: Dict[str, Any], analysis_type: AnalysisType) -> Dict[str, Any]:
        """
        Curate and filter indicators based on analysis type and relevance.
        """
        if analysis_type == AnalysisType.INDICATOR_SUMMARY:
            return self._curate_for_indicator_summary(indicators)
        elif analysis_type == AnalysisType.VOLUME_ANALYSIS:
            return self._curate_for_volume_analysis(indicators)
        elif analysis_type == AnalysisType.REVERSAL_PATTERNS:
            return self._curate_for_reversal_patterns(indicators)
        elif analysis_type == AnalysisType.CONTINUATION_LEVELS:
            return self._curate_for_continuation_levels(indicators)
        elif analysis_type == AnalysisType.FINAL_DECISION:
            return self._curate_for_final_decision(indicators)
        else:
            return self._curate_general_indicators(indicators)
    
    def _curate_for_indicator_summary(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Curate indicators specifically for comprehensive summary analysis.
        Focus on essential indicators and recent data.
        """
        curated = {
            "analysis_focus": "technical_indicators_summary",
            "key_indicators": {},
            "critical_levels": {},
            "conflict_analysis_needed": False
        }
        
        # Extract essential trend indicators with safe processing
        if 'moving_averages' in indicators:
            # New optimized structure with safe value conversion
            ma_data = indicators['moving_averages']
            curated["key_indicators"]["trend_indicators"] = {
                "sma_20": self._safe_numeric_value(ma_data.get('sma_20')),
                "sma_50": self._safe_numeric_value(ma_data.get('sma_50')),
                "sma_200": self._safe_numeric_value(ma_data.get('sma_200')),
                "ema_20": self._safe_numeric_value(ma_data.get('ema_20')),  # Restored for trend signals
                "ema_50": self._safe_numeric_value(ma_data.get('ema_50')),  # Restored for crossovers
                "price_to_sma_200": self._safe_numeric_value(ma_data.get('price_to_sma_200')),
                "sma_20_to_sma_50": self._safe_numeric_value(ma_data.get('sma_20_to_sma_50')),
                "golden_cross": self._safe_boolean_value(ma_data.get('golden_cross')),
                "death_cross": self._safe_boolean_value(ma_data.get('death_cross'))
            }
        elif 'sma' in indicators:
            # Old structure
            curated["key_indicators"]["trend_indicators"] = {
                "sma_20": self._get_latest_value(indicators['sma'], 20),
                "sma_50": self._get_latest_value(indicators['sma'], 50),
                "sma_200": self._get_latest_value(indicators['sma'], 200)
            }
        
        # Extract momentum indicators with safe processing
        if 'rsi' in indicators:
            rsi_data = indicators['rsi']
            # Handle both old (list) and new (dict) RSI structure
            if isinstance(rsi_data, dict) and 'rsi_14' in rsi_data:
                # New optimized structure with safe conversion
                rsi_current = self._safe_numeric_value(rsi_data.get('rsi_14'))
                rsi_trend = rsi_data.get('trend', 'neutral')
                rsi_status = rsi_data.get('status', 'neutral')
                curated["key_indicators"]["momentum_indicators"] = {
                    "rsi_current": rsi_current,
                    "rsi_status": rsi_status if rsi_status in ['overbought', 'oversold', 'neutral'] else 'neutral'
                }
            else:
                # Old structure (list) or invalid data
                curated["key_indicators"]["momentum_indicators"] = {
                    "rsi_current": self._get_latest_value(rsi_data) if rsi_data else None,
                    "rsi_status": "neutral"  # Simplified - determine from current value if needed
                }
        
        if 'macd' in indicators:
            macd_data = indicators['macd']
            # Handle both old (dict with lists) and new (dict with single values) MACD structure
            if isinstance(macd_data, dict) and 'macd_line' in macd_data:
                # New optimized structure with safe processing - FIX for line 119 crash
                histogram = self._safe_numeric_value(macd_data.get('histogram'))
                if "momentum_indicators" not in curated["key_indicators"]:
                    curated["key_indicators"]["momentum_indicators"] = {}
                curated["key_indicators"]["momentum_indicators"]["macd"] = {
                    "histogram": histogram,
                    "trend": "bullish" if histogram > 0 else ("bearish" if histogram < 0 else "neutral")
                }
            else:
                # Old structure with safe processing
                if "momentum_indicators" not in curated["key_indicators"]:
                    curated["key_indicators"]["momentum_indicators"] = {}
                # Handle both single values and lists for backward compatibility
                signal_val = macd_data.get('signal', 0) if macd_data else None
                histogram_val = macd_data.get('histogram', 0) if macd_data else None
                
                curated["key_indicators"]["momentum_indicators"]["macd"] = {
                    "signal": self._safe_numeric_value(signal_val) if isinstance(signal_val, (int, float, str)) else self._get_latest_value(signal_val),
                    "histogram": self._safe_numeric_value(histogram_val) if isinstance(histogram_val, (int, float, str)) else self._get_latest_value(histogram_val),
                    "trend": self._calculate_macd_trend(macd_data) if macd_data else 'neutral'
                }
        
        # Extract volume indicators with safe processing
        if 'volume' in indicators:
            volume_data = indicators['volume']
            # Handle both old (list) and new (dict) volume structure
            if isinstance(volume_data, dict) and 'volume_ratio' in volume_data:
                # New optimized structure with safe conversion
                curated["key_indicators"]["volume_indicators"] = {
                    "volume_ratio": self._safe_numeric_value(volume_data.get('volume_ratio'), 1.0),
                    "volume_trend": volume_data.get('volume_trend', 'neutral')
                }
            else:
                # Old structure (list) with safe processing
                curated["key_indicators"]["volume_indicators"] = {
                    "volume_sma": self._get_latest_value(indicators.get('volume_sma', [])),
                    "volume_ratio": self._calculate_volume_ratio(volume_data) if volume_data else 1.0,
                    "volume_trend": self._calculate_trend(volume_data, 20) if volume_data else 'neutral'
                }
        
        # Extract critical levels only (simplified approach)
        curated["critical_levels"] = self._extract_enhanced_critical_levels(indicators)
        
        # Detect conflicts and get detailed conflict information
        has_conflicts = self._detect_conflicts(curated["key_indicators"])
        curated["conflict_analysis_needed"] = has_conflicts
        
        # Add detailed conflict information for the LLM
        if hasattr(self, '_last_conflict_details'):
            curated["detected_conflicts"] = self._last_conflict_details
        else:
            curated["detected_conflicts"] = {"has_conflicts": False, "conflict_count": 0, "conflict_list": []}
        
        # Enforce JSON-safe structure
        curated = self._make_json_safe(curated)
        return curated
    
    def _curate_for_volume_analysis(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Curate indicators specifically for volume analysis.
        Focus on volume-related data and price-volume relationships.
        """
        curated = {
            "analysis_focus": "volume_pattern_analysis",
            "technical_context": {},
            "volume_metrics": {},
            "specific_questions": [
                "Are volume spikes confirming price movements?",
                "Is there volume divergence indicating trend weakness?",
                "What are the key volume-based support/resistance levels?"
            ]
        }
        
        # Extract price trend context
        price_data = None
        if 'close' in indicators:
            price_data = indicators['close']
            curated["technical_context"]["price_trend"] = self._determine_trend(price_data)
            curated["technical_context"]["price_volatility"] = self._calculate_volatility(price_data)
        
        # Extract volume metrics
        if 'volume' in indicators:
            volume_data = indicators['volume']
            curated["volume_metrics"] = {
                "volume_trend": self._calculate_trend(volume_data, 20),
                "volume_sma": self._get_latest_value(indicators.get('volume_sma', [])),
                "volume_ratio": self._calculate_volume_ratio(volume_data),
                "key_volume_levels": self._extract_volume_levels(volume_data)
            }
        
        try:
            curated = self._make_json_safe(curated)
        except Exception:
            pass
        return curated
    
    def _curate_for_reversal_patterns(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Curate indicators specifically for reversal pattern analysis.
        Focus on momentum divergences and trend strength.
        """
        curated = {
            "analysis_focus": "reversal_pattern_validation",
            "technical_context": {},
            "momentum_analysis": {},
            "validation_questions": [
                "Are divergence patterns confirmed by volume?",
                "What is the probability of pattern completion?",
                "What would invalidate these reversal signals?"
            ]
        }
        
        # Extract trend context
        price_data = None
        if 'close' in indicators:
            price_data = indicators['close']
            curated["technical_context"]["current_trend"] = self._determine_trend(price_data)
            curated["technical_context"]["trend_strength"] = self._calculate_trend_strength(price_data)
            curated["technical_context"]["key_reversal_levels"] = self._extract_reversal_levels(indicators)
        
        # Extract momentum analysis
        if 'rsi' in indicators:
            rsi_data = indicators['rsi']
            curated["momentum_analysis"]["rsi"] = {
                "current": self._get_latest_value(rsi_data),
                "divergence": self._detect_rsi_divergence(price_data, rsi_data) if price_data else False,
                "extremes": self._count_extremes(rsi_data)
            }
        
        if 'macd' in indicators:
            macd_data = indicators['macd']
            curated["momentum_analysis"]["macd"] = {
                "signal_divergence": self._detect_macd_divergence(price_data, macd_data) if price_data else False,
                "histogram_trend": self._calculate_histogram_trend(macd_data)
            }
        
        return curated
    
    def _curate_for_continuation_levels(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Curate indicators specifically for continuation and level analysis.
        Focus on support/resistance levels and trend continuation patterns.
        """
        curated = {
            "analysis_focus": "continuation_level_analysis",
            "technical_context": {},
            "level_analysis": {},
            "enhanced_levels": {},  # NEW: Add enhanced levels
            "continuation_signals": {}
        }
        
        # Extract trend context
        price_data = None
        if 'close' in indicators:
            price_data = indicators['close']
            curated["technical_context"]["trend_direction"] = self._determine_trend(price_data)
            curated["technical_context"]["trend_strength"] = self._calculate_trend_strength(price_data)
        
        # Extract level analysis (Level 2: Swing Point Analysis)
        curated["level_analysis"] = {
            "support_levels": self._extract_enhanced_support_levels(indicators),
            "resistance_levels": self._extract_enhanced_resistance_levels(indicators),
            "breakout_potential": self._assess_breakout_potential(indicators)
        }
        
        # Extract enhanced levels (Level 3: Comprehensive Analysis)
        if 'enhanced_levels' in indicators:
            curated["enhanced_levels"] = indicators['enhanced_levels']
        
        # Extract continuation signals
        if 'sma' in indicators:
            sma_data = indicators['sma']
            curated["continuation_signals"]["moving_averages"] = {
                "price_vs_sma20": self._compare_price_to_sma(price_data, sma_data, 20),
                "price_vs_sma50": self._compare_price_to_sma(price_data, sma_data, 50),
                "sma_alignment": self._check_sma_alignment(sma_data)
            }
        
        return curated
    
    def _curate_general_indicators(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        General curation for other analysis types.
        """
        return {
            "analysis_focus": "general_technical_analysis",
            "key_metrics": self._extract_key_metrics(indicators),
            "trend_context": self._extract_trend_context(indicators)
        }
    
    def _curate_for_final_decision(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Curate indicators specifically for final decision synthesis.
        """
        curated = {
            "analysis_focus": "final_decision_synthesis",
            "key_indicators": {},
            "consensus_analysis": {},
            "risk_assessment": {},
            "decision_framework": {}
        }
        
        # Extract key indicators from the general curation
        general_curated = self._curate_for_indicator_summary(indicators)
        curated["key_indicators"] = general_curated.get("key_indicators", {})
        
        # Add consensus analysis
        curated["consensus_analysis"] = {
            "trend_consensus": "neutral",
            "momentum_consensus": "neutral",
            "volume_consensus": "neutral",
            "conflicts_detected": general_curated.get("conflict_analysis_needed", False)
        }
        
        # Add risk assessment
        curated["risk_assessment"] = {
            "overall_risk": "medium",
            "trend_risk": "medium",
            "momentum_risk": "medium",
            "volume_risk": "medium"
        }
        
        # Add decision framework
        curated["decision_framework"] = {
            "entry_strategy": "wait_for_confirmation",
            "exit_strategy": "trailing_stop",
            "risk_management": "2%_stop_loss"
        }
        
        return curated
    
    def structure_context(self, curated_data: Dict[str, Any], analysis_type: AnalysisType, 
                         symbol: str, timeframe: str, knowledge_context: str = "") -> str:
        """
        Structure the curated data into a well-formatted context for LLM calls.
        """
        if analysis_type == AnalysisType.INDICATOR_SUMMARY:
            return self._structure_indicator_summary_context(curated_data, symbol, timeframe, knowledge_context)
        elif analysis_type == AnalysisType.VOLUME_ANALYSIS:
            return self._structure_volume_analysis_context(curated_data)
        elif analysis_type == AnalysisType.REVERSAL_PATTERNS:
            return self._structure_reversal_patterns_context(curated_data)
        elif analysis_type == AnalysisType.CONTINUATION_LEVELS:
            return self._structure_continuation_levels_context(curated_data)
        elif analysis_type == AnalysisType.FINAL_DECISION:
            return self._structure_final_decision_context(curated_data)
        else:
            return self._structure_general_context(curated_data)
    
    def _structure_indicator_summary_context(self, curated_data: Dict[str, Any], 
                                           symbol: str, timeframe: str, knowledge_context: str) -> str:
        """
        Structure context for indicator summary analysis - ULTRA SIMPLIFIED version.
        """
        key_indicators = curated_data.get('key_indicators', {})
        critical_levels = curated_data.get('critical_levels', {})
        detected_conflicts = curated_data.get('detected_conflicts', {})
        
        # Build ultra-concise context
        context = f"""**Symbol**: {symbol} | **Timeframe**: {timeframe}

## Technical Data:
{json.dumps(key_indicators, indent=1)}

## Levels:
{json.dumps(critical_levels, indent=1)}

{self._format_simplified_conflicts(detected_conflicts)}

{knowledge_context}
"""
        
        return context
    
    def _structure_volume_analysis_context(self, curated_data: Dict[str, Any]) -> str:
        """
        Structure context for volume analysis.
        """
        try:
            # Ensure all data is JSON-serializable
            safe_technical_context = self._make_json_safe(curated_data.get('technical_context', {}))
            safe_volume_metrics = self._make_json_safe(curated_data.get('volume_metrics', {}))
            
            context = f"""## Volume Analysis Context:
**Analysis Focus**: {curated_data.get('analysis_focus', 'volume_pattern_analysis')}

## Technical Context:
{json.dumps(safe_technical_context, indent=2)}

## Volume Metrics:
{json.dumps(safe_volume_metrics, indent=2)}

## Specific Analysis Questions:
{chr(10).join(f"- {question}" for question in curated_data.get('specific_questions', []))}

## Analysis Instructions:
1. Analyze volume patterns in relation to price movements
2. Identify volume confirmation or divergence signals
3. Determine volume-based support/resistance levels
4. Assess overall volume trend implications"""
            
            return context
        except Exception as e:
            # Fallback to simple context if JSON serialization fails
            return f"""## Volume Analysis Context:
**Analysis Focus**: {curated_data.get('analysis_focus', 'volume_pattern_analysis')}

## Analysis Instructions:
1. Analyze volume patterns in relation to price movements
2. Identify volume confirmation or divergence signals
3. Determine volume-based support/resistance levels
4. Assess overall volume trend implications"""
    
    def _structure_reversal_patterns_context(self, curated_data: Dict[str, Any]) -> str:
        """
        Structure context for reversal pattern analysis.
        """
        try:
            # Ensure all data is JSON-serializable
            safe_technical_context = self._make_json_safe(curated_data.get('technical_context', {}))
            safe_momentum_analysis = self._make_json_safe(curated_data.get('momentum_analysis', {}))
            
            context = f"""## Reversal Pattern Analysis Context:
**Analysis Focus**: {curated_data.get('analysis_focus', 'reversal_pattern_validation')}

## Technical Context:
{json.dumps(safe_technical_context, indent=2)}

## Momentum Analysis:
{json.dumps(safe_momentum_analysis, indent=2)}

## Validation Questions:
{chr(10).join(f"- {question}" for question in curated_data.get('validation_questions', []))}

## Analysis Instructions:
1. Identify and validate reversal patterns
2. Assess pattern completion probability
3. Determine confirmation and invalidation levels
4. Provide risk management recommendations"""
            
            return context
        except Exception as e:
            # Fallback to simple context if JSON serialization fails
            return f"""## Reversal Pattern Analysis Context:
**Analysis Focus**: {curated_data.get('analysis_focus', 'reversal_pattern_validation')}

## Analysis Instructions:
1. Identify and validate reversal patterns
2. Assess pattern completion probability
3. Determine confirmation and invalidation levels
4. Provide risk management recommendations"""
    
    def _structure_continuation_levels_context(self, curated_data: Dict[str, Any]) -> str:
        """
        Structure context for continuation and level analysis.
        """
        try:
            # Ensure all data is JSON-serializable
            safe_technical_context = self._make_json_safe(curated_data.get('technical_context', {}))
            safe_level_analysis = self._make_json_safe(curated_data.get('level_analysis', {}))
            safe_enhanced_levels = self._make_json_safe(curated_data.get('enhanced_levels', {}))
            safe_continuation_signals = self._make_json_safe(curated_data.get('continuation_signals', {}))
            
            context = f"""## Continuation & Level Analysis Context:
**Analysis Focus**: {curated_data.get('analysis_focus', 'continuation_level_analysis')}

## Technical Context:
{json.dumps(safe_technical_context, indent=2)}

## Level Analysis (Level 2 - Swing Point Analysis):
{json.dumps(safe_level_analysis, indent=2)}

## Enhanced Levels (Level 3 - Comprehensive Analysis):
{json.dumps(safe_enhanced_levels, indent=2)}

## Continuation Signals:
{json.dumps(safe_continuation_signals, indent=2)}

## Analysis Instructions:
1. **Primary Analysis**: Use Level 2 (swing point) levels as your main support/resistance reference
2. **Confirmation**: Use Level 3 (enhanced) levels to validate and confirm Level 2 levels
3. **Level Convergence**: Identify where Level 2 and Level 3 levels align (strong confirmation)
4. **Fibonacci Validation**: Use Fibonacci levels from Level 3 to validate swing points
5. **Pivot Point Integration**: Incorporate pivot points for additional confirmation
6. **Breakout Assessment**: Determine breakout/breakdown potential based on level strength
7. **Entry/Exit Planning**: Provide specific entry/exit levels with confidence ratings
8. **Risk Management**: Use level confirmation to assess pattern failure risk"""
            
            return context
        except Exception as e:
            # Fallback to simple context if JSON serialization fails
            return f"""## Continuation & Level Analysis Context:
**Analysis Focus**: {curated_data.get('analysis_focus', 'continuation_level_analysis')}

## Analysis Instructions:
1. **Primary Analysis**: Use Level 2 (swing point) levels as your main support/resistance reference
2. **Confirmation**: Use Level 3 (enhanced) levels to validate and confirm Level 2 levels
3. **Level Convergence**: Identify where Level 2 and Level 3 levels align (strong confirmation)
4. **Fibonacci Validation**: Use Fibonacci levels from Level 3 to validate swing points
5. **Pivot Point Integration**: Incorporate pivot points for additional confirmation
6. **Breakout Assessment**: Determine breakout/breakdown potential based on level strength
7. **Entry/Exit Planning**: Provide specific entry/exit levels with confidence ratings
8. **Risk Management**: Use level confirmation to assess pattern failure risk"""
    
    def _structure_final_decision_context(self, curated_data: Dict[str, Any]) -> str:
        """
        Structure context for final decision synthesis.
        """
        try:
            # Ensure all data is JSON-serializable
            safe_consensus_analysis = self._make_json_safe(curated_data.get('consensus_analysis', {}))
            safe_risk_assessment = self._make_json_safe(curated_data.get('risk_assessment', {}))
            safe_decision_framework = self._make_json_safe(curated_data.get('decision_framework', {}))
            
            context = f"""## Final Decision Synthesis Context:
**Analysis Focus**: {curated_data.get('analysis_focus', 'final_decision_synthesis')}

## Consensus Analysis:
{json.dumps(safe_consensus_analysis, indent=2)}

## Risk Assessment:
{json.dumps(safe_risk_assessment, indent=2)}

## Decision Framework:
{json.dumps(safe_decision_framework, indent=2)}

## Synthesis Instructions:
1. Resolve any conflicts between different analyses
2. Provide clear, actionable trading recommendations
3. Include specific entry, exit, and risk management levels"""
            
            return context
        except Exception as e:
            # Fallback to simple context if JSON serialization fails
            return f"""## Final Decision Synthesis Context:
**Analysis Focus**: {curated_data.get('analysis_focus', 'final_decision_synthesis')}

## Synthesis Instructions:
1. Resolve any conflicts between different analyses
2. Provide clear, actionable trading recommendations
3. Include specific entry, exit, and risk management levels"""
    
    def _structure_general_context(self, curated_data: Dict[str, Any]) -> str:
        """
        Structure general context for other analysis types.
        """
        try:
            # Ensure all data is JSON-serializable
            safe_curated_data = self._make_json_safe(curated_data)
            
            return f"""## General Analysis Context:
{json.dumps(safe_curated_data, indent=2)}"""
        except Exception as e:
            # Fallback to simple context if JSON serialization fails
            return f"""## General Analysis Context:
**Analysis Focus**: {curated_data.get('analysis_focus', 'general_analysis')}

## Analysis Instructions:
1. Analyze the provided data comprehensively
2. Provide actionable insights and recommendations
3. Include risk assessment and management strategies"""
    
    # Helper methods for data extraction and analysis
    def _get_latest_value(self, data: List[float], period: int = None) -> float:
        """Get the latest value from a data series."""
        if not data or len(data) == 0:
            return None
        
        # Handle case where data might be a string or other type
        if not isinstance(data, list):
            try:
                # Try to convert to float if it's a single value
                return float(data)
            except (ValueError, TypeError):
                return None
        
        try:
            # Ensure all values are numeric
            numeric_data = []
            for item in data:
                try:
                    numeric_data.append(float(item))
                except (ValueError, TypeError):
                    continue
            
            if not numeric_data:
                return None
                
            if period and len(numeric_data) >= period:
                return numeric_data[-period]
            return numeric_data[-1]
        except (IndexError, KeyError):
            return None
    
    def _calculate_trend(self, data: List[float], period: int = 20) -> str:
        """Calculate trend direction from data series."""
        # Handle case where data might be a string or other type
        if not isinstance(data, list):
            return "insufficient_data"
        
        # Ensure all values are numeric
        numeric_data = []
        for item in data:
            try:
                numeric_data.append(float(item))
            except (ValueError, TypeError):
                continue
        
        if len(numeric_data) < period:
            return "insufficient_data"
        
        recent = numeric_data[-period:]
        if len(recent) < 2:
            return "insufficient_data"
        
        slope = (recent[-1] - recent[0]) / len(recent)
        if slope > 0.01:
            return "bullish"
        elif slope < -0.01:
            return "bearish"
        else:
            return "neutral"
    
    def _count_extremes(self, rsi_data: List[float]) -> Dict[str, int]:
        """Count oversold and overbought periods in RSI."""
        if not rsi_data:
            return {"oversold": 0, "overbought": 0}
        
        # Handle case where data might be a string or other type
        if not isinstance(rsi_data, list):
            return {"oversold": 0, "overbought": 0}
        
        # Ensure all values are numeric
        numeric_data = []
        for item in rsi_data:
            try:
                numeric_data.append(float(item))
            except (ValueError, TypeError):
                continue
        
        oversold = sum(1 for rsi in numeric_data if rsi < 30)
        overbought = sum(1 for rsi in numeric_data if rsi > 70)
        
        return {"oversold": oversold, "overbought": overbought}
    
    def _calculate_macd_trend(self, macd_data: Dict[str, List[float]]) -> str:
        """Calculate MACD trend direction."""
        if not isinstance(macd_data, dict):
            return "insufficient_data"
        
        histogram = macd_data.get('histogram', [])
        if not isinstance(histogram, list) or len(histogram) < 5:
            return "insufficient_data"
        
        # Ensure all values are numeric
        numeric_histogram = []
        for item in histogram:
            try:
                numeric_histogram.append(float(item))
            except (ValueError, TypeError):
                continue
        
        if len(numeric_histogram) < 5:
            return "insufficient_data"
        
        recent_histogram = numeric_histogram[-5:]
        if all(h > 0 for h in recent_histogram):
            return "bullish"
        elif all(h < 0 for h in recent_histogram):
            return "bearish"
        else:
            return "mixed"
    
    def _calculate_volume_ratio(self, volume_data: List[float]) -> float:
        """Calculate current volume ratio to average."""
        # Handle case where data might be a string or other type
        if not isinstance(volume_data, list):
            return 1.0
        
        # Ensure all values are numeric
        numeric_data = []
        for item in volume_data:
            try:
                numeric_data.append(float(item))
            except (ValueError, TypeError):
                continue
        
        if len(numeric_data) < 20:
            return 1.0
        
        current_volume = numeric_data[-1]
        avg_volume = sum(numeric_data[-20:]) / 20
        
        return current_volume / avg_volume if avg_volume > 0 else 1.0
    
    def _extract_critical_levels(self, indicators: Dict[str, Any]) -> Dict[str, List[float]]:
        """Extract critical support and resistance levels with improved multi-level detection."""
        levels = {"support": [], "resistance": []}
        
        # Try to use proper swing point analysis if we have enough price data
        if self._can_perform_swing_analysis(indicators):
            levels = self._extract_swing_point_levels(indicators)
        
        # If swing analysis didn't work or didn't provide enough levels, use fallback methods
        if len(levels["support"]) < 3 or len(levels["resistance"]) < 3:
            fallback_levels = self._extract_basic_levels(indicators)
            
            # Merge levels, avoiding duplicates
            for level_type in ["support", "resistance"]:
                existing_levels = set(levels[level_type])
                for new_level in fallback_levels[level_type]:
                    # Only add if not too close to existing levels (2% threshold)
                    is_duplicate = any(abs(new_level - existing) / existing < 0.02 
                                     for existing in existing_levels if existing > 0)
                    if not is_duplicate:
                        levels[level_type].append(new_level)
        
        # Ensure we have at least 1 level of each type, generate artificial levels if needed
        current_price = self._get_current_price(indicators)
        
        if len(levels["support"]) == 0 and current_price:
            # Generate basic support levels based on recent price action
            levels["support"] = [
                current_price * 0.98,  # 2% below current
                current_price * 0.95,  # 5% below current
                current_price * 0.92   # 8% below current
            ]
        
        if len(levels["resistance"]) == 0 and current_price:
            # Generate basic resistance levels based on recent price action
            levels["resistance"] = [
                current_price * 1.02,  # 2% above current
                current_price * 1.05,  # 5% above current
                current_price * 1.08   # 8% above current
            ]
        
        # Sort and limit to 3 levels each
        levels["support"] = sorted(levels["support"], reverse=True)[:3]  # Highest to lowest
        levels["resistance"] = sorted(levels["resistance"])[:3]  # Lowest to highest
        
        return levels
    
    def _can_perform_swing_analysis(self, indicators: Dict[str, Any]) -> bool:
        """Check if we have enough data for swing point analysis."""
        price_data = indicators.get('close', [])
        high_data = indicators.get('high', [])
        low_data = indicators.get('low', [])
        
        return (isinstance(price_data, list) and len(price_data) >= 20 and
                isinstance(high_data, list) and len(high_data) >= 20 and
                isinstance(low_data, list) and len(low_data) >= 20)
    
    def _extract_swing_point_levels(self, indicators: Dict[str, Any]) -> Dict[str, List[float]]:
        """Extract support/resistance using swing point analysis."""
        levels = {"support": [], "resistance": []}
        
        try:
            # Get price data
            high_data = [float(x) for x in indicators['high'] if x is not None]
            low_data = [float(x) for x in indicators['low'] if x is not None]
            
            if len(high_data) < 20 or len(low_data) < 20:
                return levels
            
            # Use scipy for swing point detection if available
            try:
                from scipy.signal import argrelextrema
                import numpy as np
                
                # Try different order values to find swing points
                support_candidates = set()
                resistance_candidates = set()
                
                for order in [3, 5, 8]:
                    if len(low_data) >= order * 2 + 1:
                        # Find local minima (support)
                        lows_idx = argrelextrema(np.array(low_data), np.less, order=order)[0]
                        for idx in lows_idx:
                            if 0 <= idx < len(low_data):
                                support_candidates.add(float(low_data[idx]))
                    
                    if len(high_data) >= order * 2 + 1:
                        # Find local maxima (resistance)
                        highs_idx = argrelextrema(np.array(high_data), np.greater, order=order)[0]
                        for idx in highs_idx:
                            if 0 <= idx < len(high_data):
                                resistance_candidates.add(float(high_data[idx]))
                
                levels["support"] = sorted(list(support_candidates), reverse=True)
                levels["resistance"] = sorted(list(resistance_candidates))
                
            except ImportError:
                # Fallback to simple min/max approach if scipy not available
                levels["support"] = [min(low_data)]
                levels["resistance"] = [max(high_data)]
                
        except Exception:
            # If anything fails, return empty levels
            pass
        
        return levels
    
    def _extract_basic_levels(self, indicators: Dict[str, Any]) -> Dict[str, List[float]]:
        """Extract basic support/resistance levels from price data and moving averages."""
        levels = {"support": [], "resistance": []}
        
        # Extract from moving averages
        if 'sma' in indicators:
            sma_data = indicators['sma']
            if isinstance(sma_data, dict):
                if 20 in sma_data and isinstance(sma_data[20], list) and len(sma_data[20]) > 0:
                    try:
                        levels["support"].append(float(sma_data[20][-1]))
                    except (ValueError, TypeError):
                        pass
                if 50 in sma_data and isinstance(sma_data[50], list) and len(sma_data[50]) > 0:
                    try:
                        levels["support"].append(float(sma_data[50][-1]))
                    except (ValueError, TypeError):
                        pass
        
        # Extract from price data ranges
        if 'close' in indicators:
            close_data = indicators['close']
            if isinstance(close_data, list) and len(close_data) >= 20:
                # Ensure all values are numeric
                numeric_data = []
                for item in close_data:
                    try:
                        numeric_data.append(float(item))
                    except (ValueError, TypeError):
                        continue
                
                if len(numeric_data) >= 20:
                    # Add different percentiles for more levels
                    sorted_data = sorted(numeric_data[-20:])
                    levels["support"].extend([
                        sorted_data[0],  # Minimum
                        sorted_data[len(sorted_data)//4],  # 25th percentile
                        sorted_data[len(sorted_data)//3]   # 33rd percentile
                    ])
                    levels["resistance"].extend([
                        sorted_data[-1],  # Maximum
                        sorted_data[-len(sorted_data)//4],  # 75th percentile
                        sorted_data[-len(sorted_data)//3]   # 67th percentile
                    ])
        
        return levels
    
    def _get_current_price(self, indicators: Dict[str, Any]) -> float:
        """Get current price from indicators."""
        if 'close' in indicators:
            close_data = indicators['close']
            if isinstance(close_data, list) and len(close_data) > 0:
                try:
                    return float(close_data[-1])
                except (ValueError, TypeError):
                    pass
        return None
    
    def _detect_conflicts(self, key_indicators: Dict[str, Any]) -> bool:
        """Enhanced comprehensive conflict detection system that catches all possible conflicts."""
        conflict_details = self._comprehensive_conflict_analysis(key_indicators)
        # Store detailed conflict info for use in context structuring
        self._last_conflict_details = conflict_details
        return conflict_details["has_conflicts"]
    
    def _comprehensive_conflict_analysis(self, key_indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Multi-layered conflict detection that returns detailed conflict information for the LLM."""
        conflicts = []
        conflict_details = {
            "has_conflicts": False,
            "conflict_count": 0,
            "conflict_list": [],
            "conflict_categories": {
                "momentum_vs_trend": [],
                "timeframe_conflicts": [],
                "volume_conflicts": [],
                "signal_strength": [],
                "market_regime": [],
                "statistical_divergences": []
            },
            "conflict_severity": "none"  # none, low, medium, high, critical
        }
        
        # Extract key data safely
        momentum_indicators = key_indicators.get('momentum_indicators', {})
        trend_indicators = key_indicators.get('trend_indicators', {})
        volume_indicators = key_indicators.get('volume_indicators', {})
        
        # Parse all indicator values safely
        parsed_data = self._parse_indicator_values(momentum_indicators, trend_indicators, volume_indicators)
        
        # Detect market regime for context-sensitive analysis
        market_regime = self._detect_comprehensive_market_regime(parsed_data)
        
        # Layer 1: Explicit Rule-Based Conflicts
        rule_conflicts = self._detect_rule_based_conflicts(parsed_data)
        for conflict in rule_conflicts:
            conflicts.append(conflict)
            # Categorize conflicts
            if "RSI" in conflict and "MACD" in conflict:
                conflict_details["conflict_categories"]["momentum_vs_trend"].append(conflict)
            elif "SMA" in conflict or "price" in conflict.lower():
                conflict_details["conflict_categories"]["timeframe_conflicts"].append(conflict)
            else:
                conflict_details["conflict_categories"]["momentum_vs_trend"].append(conflict)
        
        # Layer 2: Statistical Divergence Conflicts
        divergence_conflicts = self._detect_statistical_divergences(parsed_data)
        for conflict in divergence_conflicts:
            conflicts.append(conflict)
            conflict_details["conflict_categories"]["statistical_divergences"].append(conflict)
        
        # Layer 3: Cross-Timeframe Conflicts
        timeframe_conflicts = self._detect_timeframe_conflicts(parsed_data)
        for conflict in timeframe_conflicts:
            conflicts.append(conflict)
            conflict_details["conflict_categories"]["timeframe_conflicts"].append(conflict)
        
        # Layer 4: Signal Strength Conflicts
        strength_conflicts = self._detect_signal_strength_conflicts(parsed_data)
        for conflict in strength_conflicts:
            conflicts.append(conflict)
            conflict_details["conflict_categories"]["signal_strength"].append(conflict)
        
        # Layer 5: Market Regime Conflicts
        regime_conflicts = self._detect_market_regime_conflicts(parsed_data)
        for conflict in regime_conflicts:
            conflicts.append(conflict)
            conflict_details["conflict_categories"]["market_regime"].append(conflict)
        
        # Layer 6: Meta-Analysis (Any remaining edge cases)
        meta_conflicts = self._detect_meta_conflicts(parsed_data)
        for conflict in meta_conflicts:
            conflicts.append(conflict)
            # Meta conflicts usually relate to volume or signal strength
            if "volume" in conflict.lower():
                conflict_details["conflict_categories"]["volume_conflicts"].append(conflict)
            else:
                conflict_details["conflict_categories"]["signal_strength"].append(conflict)
        
        # Layer 7: Market Regime-Specific Conflicts (NEW)
        regime_specific_conflicts = self._apply_regime_specific_conflicts(parsed_data, market_regime)
        for conflict in regime_specific_conflicts:
            conflicts.append(conflict)
            conflict_details["conflict_categories"]["market_regime"].append(conflict)
        
        # Populate conflict details
        conflict_details["has_conflicts"] = len(conflicts) > 0
        conflict_details["conflict_count"] = len(conflicts)
        conflict_details["conflict_list"] = conflicts
        conflict_details["market_regime"] = market_regime  # Add regime context
        
        # Determine conflict severity using weighted system
        if len(conflicts) == 0:
            conflict_details["conflict_severity"] = "none"
            conflict_details["weighted_score"] = 0.0
        else:
            severity, weighted_score = self._calculate_weighted_severity(conflict_details)
            conflict_details["conflict_severity"] = severity
            conflict_details["weighted_score"] = weighted_score
        
        return conflict_details
    
    def _parse_indicator_values(self, momentum_indicators, trend_indicators, volume_indicators):
        """Parse all indicator values safely and return structured data."""
        data = {
            'rsi': None, 'rsi_trend': None, 'rsi_status': None,
            'macd_line': None, 'macd_signal': None, 'macd_histogram': None, 'macd_trend': None,
            'price_to_sma200': None, 'sma_20_to_sma_50': None,
            'golden_cross': False, 'death_cross': False,
            'volume_ratio': None, 'obv': None, 'volume_trend': None,
            'sma_20': None, 'sma_50': None, 'sma_200': None,
            'ema_20': None, 'ema_50': None
        }
        
        # Parse momentum indicators safely
        if isinstance(momentum_indicators, dict):
            # RSI data with safe conversion
            rsi_data = momentum_indicators.get('rsi_current') or momentum_indicators.get('rsi', {}).get('rsi_current')
            data['rsi'] = self._safe_numeric_value(rsi_data)
            data['rsi_trend'] = momentum_indicators.get('rsi_trend')
            data['rsi_status'] = momentum_indicators.get('rsi_status')
            
            # MACD data with safe conversion
            macd_data = momentum_indicators.get('macd', {})
            if isinstance(macd_data, dict):
                data['macd_line'] = self._safe_numeric_value(macd_data.get('macd_line'))
                data['macd_signal'] = self._safe_numeric_value(macd_data.get('signal_line'))
                data['macd_histogram'] = self._safe_numeric_value(macd_data.get('histogram'))
                data['macd_trend'] = macd_data.get('trend')
        
        # Parse trend indicators safely
        if isinstance(trend_indicators, dict):
            for key in ['price_to_sma_200', 'sma_20_to_sma_50', 'sma_20', 'sma_50', 'sma_200', 'ema_20', 'ema_50']:
                data[key] = self._safe_numeric_value(trend_indicators.get(key))
            
            data['golden_cross'] = self._safe_boolean_value(trend_indicators.get('golden_cross'))
            data['death_cross'] = self._safe_boolean_value(trend_indicators.get('death_cross'))
        
        # Parse volume indicators safely
        if isinstance(volume_indicators, dict):
            data['volume_ratio'] = self._safe_numeric_value(volume_indicators.get('volume_ratio'), 1.0)
            data['obv'] = self._safe_numeric_value(volume_indicators.get('obv'))
            data['volume_trend'] = volume_indicators.get('volume_trend')
        
        return data
    
    def _detect_rule_based_conflicts(self, data):
        """Layer 1: Explicit rule-based conflict detection."""
        conflicts = []
        
        # RSI vs MACD Conflicts
        if data['rsi'] is not None and data['macd_trend'] is not None:
            if data['rsi'] > 70 and data['macd_trend'] == 'bullish':
                conflicts.append("RSI overbought while MACD bullish")
            if data['rsi'] < 30 and data['macd_trend'] == 'bearish':
                conflicts.append("RSI oversold while MACD bearish")
            if data['rsi'] > 75 and data['macd_histogram'] is not None and data['macd_histogram'] > 3:
                conflicts.append("RSI extremely overbought with strong MACD bullish")
            if data['rsi'] < 25 and data['macd_histogram'] is not None and data['macd_histogram'] < -3:
                conflicts.append("RSI extremely oversold with strong MACD bearish")
        
        # Price vs Moving Average Conflicts
        if data['price_to_sma200'] is not None and data['macd_trend'] is not None:
            if data['price_to_sma200'] < -0.05 and data['macd_trend'] == 'bullish':
                conflicts.append("Price below SMA200 while MACD bullish")
            if data['price_to_sma200'] > 0.05 and data['macd_trend'] == 'bearish':
                conflicts.append("Price above SMA200 while MACD bearish")
            if data['price_to_sma200'] < -0.10 and data['rsi'] is not None and data['rsi'] > 60:
                conflicts.append("Price significantly below SMA200 while RSI bullish")
        
        # Moving Average Alignment Conflicts
        if data['sma_20_to_sma_50'] is not None and data['rsi'] is not None:
            if data['sma_20_to_sma_50'] < -0.02 and data['rsi'] > 65:
                conflicts.append("SMA20 below SMA50 while RSI overbought")
            if data['sma_20_to_sma_50'] > 0.02 and data['rsi'] < 35:
                conflicts.append("SMA20 above SMA50 while RSI oversold")
        
        # Volume vs Price/Momentum Conflicts
        if data['volume_ratio'] is not None and data['obv'] is not None and data['macd_trend'] is not None:
            if data['volume_ratio'] > 1.5 and data['obv'] < 0 and data['macd_trend'] == 'bullish':
                conflicts.append("High volume with negative OBV during bullish momentum")
            if data['volume_ratio'] < 0.7 and data['macd_histogram'] is not None and abs(data['macd_histogram']) > 2:
                conflicts.append("Low volume with strong momentum signals")
        
        # Golden/Death Cross Conflicts
        if data['golden_cross'] and data['rsi'] is not None and data['rsi'] > 70:
            conflicts.append("Golden cross while RSI overbought")
        if data['death_cross'] and data['rsi'] is not None and data['rsi'] < 30:
            conflicts.append("Death cross while RSI oversold")
        
        return conflicts
    
    def _detect_statistical_divergences(self, data):
        """Layer 2: Statistical divergence detection."""
        conflicts = []
        
        # RSI Trend vs Price Trend Conflicts
        if data['rsi_trend'] is not None and data['macd_trend'] is not None:
            if data['rsi_trend'] == 'down' and data['macd_trend'] == 'bullish':
                conflicts.append("RSI trending down while MACD bullish")
            if data['rsi_trend'] == 'up' and data['macd_trend'] == 'bearish':
                conflicts.append("RSI trending up while MACD bearish")
        
        # Extreme RSI with Contrary Signals
        if data['rsi'] is not None:
            if data['rsi'] > 80 and data['sma_20_to_sma_50'] is not None and data['sma_20_to_sma_50'] < 0:
                conflicts.append("Extremely overbought RSI with bearish MA alignment")
            if data['rsi'] < 20 and data['sma_20_to_sma_50'] is not None and data['sma_20_to_sma_50'] > 0:
                conflicts.append("Extremely oversold RSI with bullish MA alignment")
        
        # MACD Signal Line Conflicts
        if data['macd_line'] is not None and data['macd_signal'] is not None:
            macd_above_signal = data['macd_line'] > data['macd_signal']
            if macd_above_signal and data['rsi'] is not None and data['rsi'] > 75:
                conflicts.append("MACD above signal line while RSI extremely overbought")
            if not macd_above_signal and data['rsi'] is not None and data['rsi'] < 25:
                conflicts.append("MACD below signal line while RSI extremely oversold")
        
        return conflicts
    
    def _detect_timeframe_conflicts(self, data):
        """Layer 3: Multi-timeframe conflict detection."""
        conflicts = []
        
        # Short vs Long-term Trend Conflicts
        short_term_bullish = (data['sma_20_to_sma_50'] is not None and data['sma_20_to_sma_50'] > 0) or (data['macd_trend'] == 'bullish')
        long_term_bearish = data['price_to_sma200'] is not None and data['price_to_sma200'] < -0.03
        
        if short_term_bullish and long_term_bearish:
            conflicts.append("Short-term bullish signals against long-term bearish trend")
        
        short_term_bearish = (data['sma_20_to_sma_50'] is not None and data['sma_20_to_sma_50'] < 0) or (data['macd_trend'] == 'bearish')
        long_term_bullish = data['price_to_sma200'] is not None and data['price_to_sma200'] > 0.03
        
        if short_term_bearish and long_term_bullish:
            conflicts.append("Short-term bearish signals against long-term bullish trend")
        
        # Cross Validation Between Different MA Periods
        if data['sma_20'] is not None and data['sma_50'] is not None and data['sma_200'] is not None:
            sma_order_bullish = data['sma_20'] > data['sma_50'] > data['sma_200']
            sma_order_bearish = data['sma_20'] < data['sma_50'] < data['sma_200']
            
            if not sma_order_bullish and not sma_order_bearish:
                conflicts.append("Mixed moving average alignment signals")
        
        return conflicts
    
    def _detect_signal_strength_conflicts(self, data):
        """Layer 4: Signal strength vs context conflicts."""
        conflicts = []
        
        # Weak signals in strong market conditions
        if data['macd_histogram'] is not None and abs(data['macd_histogram']) < 1:
            if data['volume_ratio'] is not None and data['volume_ratio'] > 2:
                conflicts.append("Weak MACD signals during high volume conditions")
        
        # Strong signals in weak market conditions
        if data['macd_histogram'] is not None and abs(data['macd_histogram']) > 5:
            if data['volume_ratio'] is not None and data['volume_ratio'] < 0.6:
                conflicts.append("Strong MACD signals during low volume conditions")
        
        # RSI extreme with weak confirmation
        if data['rsi'] is not None:
            if data['rsi'] > 80 and data['macd_histogram'] is not None and abs(data['macd_histogram']) < 2:
                conflicts.append("Extreme RSI with weak momentum confirmation")
            if data['rsi'] < 20 and data['macd_histogram'] is not None and abs(data['macd_histogram']) < 2:
                conflicts.append("Extreme RSI with weak momentum confirmation")
        
        return conflicts
    
    def _detect_market_regime_conflicts(self, data):
        """Layer 5: Market regime-specific conflicts."""
        conflicts = []
        
        # Trending vs Ranging Market Conflicts
        trending_up = data['price_to_sma200'] is not None and data['price_to_sma200'] > 0.05
        trending_down = data['price_to_sma200'] is not None and data['price_to_sma200'] < -0.05
        ranging = not trending_up and not trending_down
        
        if ranging:
            # In ranging markets, extreme RSI should mean reversal, not continuation
            if data['rsi'] is not None and data['rsi'] > 70 and data['macd_trend'] == 'bullish':
                conflicts.append("Ranging market with RSI overbought and continued bullish momentum")
            if data['rsi'] is not None and data['rsi'] < 30 and data['macd_trend'] == 'bearish':
                conflicts.append("Ranging market with RSI oversold and continued bearish momentum")
        
        # High volatility regime conflicts (approximated by volume)
        high_vol = data['volume_ratio'] is not None and data['volume_ratio'] > 1.8
        if high_vol:
            if data['rsi'] is not None and 40 < data['rsi'] < 60 and data['macd_histogram'] is not None and abs(data['macd_histogram']) < 1:
                conflicts.append("High volatility environment with neutral indicators")
        
        return conflicts
    
    def _detect_meta_conflicts(self, data):
        """Layer 6: Meta-analysis for any remaining edge cases."""
        conflicts = []
        
        # Check for any indicator combinations that seem contradictory
        indicator_signals = []
        
        # Collect all bullish/bearish signals
        if data['rsi'] is not None and data['rsi'] > 60:
            indicator_signals.append('rsi_bullish')
        elif data['rsi'] is not None and data['rsi'] < 40:
            indicator_signals.append('rsi_bearish')
        
        if data['macd_trend'] == 'bullish':
            indicator_signals.append('macd_bullish')
        elif data['macd_trend'] == 'bearish':
            indicator_signals.append('macd_bearish')
        
        if data['sma_20_to_sma_50'] is not None and data['sma_20_to_sma_50'] > 0.01:
            indicator_signals.append('ma_bullish')
        elif data['sma_20_to_sma_50'] is not None and data['sma_20_to_sma_50'] < -0.01:
            indicator_signals.append('ma_bearish')
        
        if data['price_to_sma200'] is not None and data['price_to_sma200'] > 0.02:
            indicator_signals.append('longterm_bullish')
        elif data['price_to_sma200'] is not None and data['price_to_sma200'] < -0.02:
            indicator_signals.append('longterm_bearish')
        
        # Count bullish vs bearish signals
        bullish_count = len([s for s in indicator_signals if 'bullish' in s])
        bearish_count = len([s for s in indicator_signals if 'bearish' in s])
        
        # If we have both bullish and bearish signals, there's likely a conflict
        if bullish_count > 0 and bearish_count > 0:
            if bullish_count == bearish_count:
                conflicts.append("Equal bullish and bearish signals detected")
            elif abs(bullish_count - bearish_count) == 1:
                conflicts.append("Mixed signals with slight directional bias")
        
        # Enhanced volume confirmation meta-check
        volume_strength = self._classify_volume_strength(data.get('volume_ratio'))
        momentum_strength = abs(data.get('macd_histogram', 0))
        
        # Strong momentum (>3) needs at least above_average volume
        if momentum_strength > 3 and volume_strength in ['low', 'below_average']:
            conflicts.append(f"Very strong momentum signals ({momentum_strength:.1f}) with {volume_strength.replace('_', ' ')} volume")
        
        # Moderate momentum (1-3) needs at least average volume  
        elif momentum_strength > 1 and volume_strength == 'low':
            conflicts.append(f"Moderate momentum signals with low volume confirmation")
        
        # High volume (>1.5x) with weak momentum might indicate distribution
        elif volume_strength in ['high', 'very_high'] and momentum_strength < 0.5:
            conflicts.append(f"{volume_strength.replace('_', ' ').title()} volume with weak momentum (possible distribution)")
        
        # Final catch-all: if we have multiple strong indicators pointing in different directions
        extreme_rsi_bullish = data['rsi'] is not None and data['rsi'] > 75
        extreme_rsi_bearish = data['rsi'] is not None and data['rsi'] < 25
        strong_macd_opposite = False
        
        if extreme_rsi_bullish and data['macd_histogram'] is not None and data['macd_histogram'] < -2:
            strong_macd_opposite = True
        if extreme_rsi_bearish and data['macd_histogram'] is not None and data['macd_histogram'] > 2:
            strong_macd_opposite = True
        
        if strong_macd_opposite:
            conflicts.append("Extreme RSI contradicting strong MACD signals")
        
        return conflicts
    
    def _format_conflict_section(self, conflict_data: Dict[str, Any]) -> str:
        """Format detailed conflict information for the LLM."""
        if not conflict_data.get('has_conflicts', False):
            return "## Conflict Analysis:\n- No conflicts detected\n- All indicators are aligned"
        
        conflict_count = conflict_data.get('conflict_count', 0)
        severity = conflict_data.get('conflict_severity', 'unknown')
        conflict_list = conflict_data.get('conflict_list', [])
        conflict_categories = conflict_data.get('conflict_categories', {})
        
        weighted_score = conflict_data.get('weighted_score', 0.0)
        section = f"""## Prioritized Conflict Analysis:
- **Conflicts Detected**: {conflict_count}
- **Severity Level**: {severity.upper()}
- **Weighted Score**: {weighted_score:.2f}

### Conflicts by Priority Level:
"""
        
        # Sort categories by priority (highest weight first)
        sorted_categories = sorted(conflict_categories.items(), 
                                 key=lambda x: self.CONFLICT_WEIGHTS.get(x[0], {'base_weight': 0})['base_weight'], 
                                 reverse=True)
        
        # Add conflicts by category with priority levels
        for category, conflicts in sorted_categories:
            if conflicts:
                priority_level = self._get_priority_level(category)
                category_name = category.replace('_', ' ').title()
                section += f"\n🔥 **{priority_level} Priority** - {category_name}:\n"
                for conflict in conflicts:
                    # Calculate individual conflict weight
                    base_weight = self.CONFLICT_WEIGHTS.get(category, {'base_weight': 0.2})['base_weight']
                    multiplier = self.CONFLICT_WEIGHTS.get(category, {'multipliers': {}})['multipliers'].get(conflict, 1.0)
                    conflict_weight = base_weight * multiplier
                    section += f"  • **[Weight: {conflict_weight:.1f}]** {conflict}\n"
        
        section += "\n### Resolution Instructions:"
        if severity in ['high', 'critical']:
            section += "\n- **HIGH PRIORITY**: These conflicts significantly impact decision confidence"
            section += "\n- Focus on resolving the most critical conflicts first"
            section += "\n- Consider waiting for clearer signals if conflicts are too severe"
        else:
            section += "\n- Analyze each conflict and determine which signals take priority"
            section += "\n- Provide clear rationale for conflict resolution in your analysis"
        
        return section
    
    def _format_simplified_conflicts(self, conflict_data: Dict[str, Any]) -> str:
        """Format conflicts in a streamlined, actionable manner."""
        if not conflict_data.get('has_conflicts', False):
            return "## Signal Conflicts: None detected"
        
        conflict_count = conflict_data.get('conflict_count', 0)
        severity = conflict_data.get('conflict_severity', 'low')
        
        section = f"## Signal Conflicts ({conflict_count} detected, {severity.upper()} severity):\n"
        
        # Get top conflicts only
        conflict_categories = conflict_data.get('conflict_categories', {})
        sorted_categories = sorted(conflict_categories.items(), 
                                 key=lambda x: self.CONFLICT_WEIGHTS.get(x[0], {'base_weight': 0})['base_weight'], 
                                 reverse=True)[:2]  # Only top 2 categories
        
        for category, conflicts in sorted_categories:
            if conflicts:
                section += f"\n**{category.replace('_', ' ').title()}:**\n"
                for conflict in conflicts[:1]:  # Only top conflict per category
                    section += f"- {conflict}\n"
        
        # Simplified resolution instructions
        if severity in ['high', 'critical']:
            section += "\n**Resolution:** Wait for clearer signals or reduce position size"
        else:
            section += "\n**Resolution:** Prioritize trend indicators over momentum signals"
            
        return section
    
    def _get_conflict_instruction(self, conflict_data: Dict[str, Any]) -> str:
        """Get appropriate instruction based on conflict status."""
        if not conflict_data.get('has_conflicts', False):
            return "Analyze indicators for consistent signals and confirm alignment"
        else:
            return "Resolve the specific conflicts listed above using your technical analysis expertise"
    
    def _determine_trend(self, price_data: List[float]) -> str:
        """Determine overall trend from price data."""
        return self._calculate_trend(price_data, 20)
    
    def _calculate_volatility(self, price_data: List[float]) -> float:
        """Calculate price volatility."""
        # Handle case where data might be a string or other type
        if not isinstance(price_data, list):
            return 0.0
        
        # Ensure all values are numeric
        numeric_data = []
        for item in price_data:
            try:
                numeric_data.append(float(item))
            except (ValueError, TypeError):
                continue
        
        if len(numeric_data) < 20:
            return 0.0
        
        returns = [(numeric_data[i] - numeric_data[i-1]) / numeric_data[i-1] 
                  for i in range(1, len(numeric_data))]
        
        return np.std(returns) if returns else 0.0
    
    def _extract_volume_levels(self, volume_data: List[float]) -> List[float]:
        """Extract key volume levels."""
        # Handle case where data might be a string or other type
        if not isinstance(volume_data, list):
            return []
        
        # Ensure all values are numeric
        numeric_data = []
        for item in volume_data:
            try:
                numeric_data.append(float(item))
            except (ValueError, TypeError):
                continue
        
        if len(numeric_data) < 20:
            return []
        
        recent_volume = numeric_data[-20:]
        return [min(recent_volume), sum(recent_volume) / len(recent_volume), max(recent_volume)]
    
    def _calculate_trend_strength(self, price_data: List[float]) -> str:
        """Calculate trend strength."""
        # Handle case where data might be a string or other type
        if not isinstance(price_data, list):
            return "insufficient_data"
        
        # Ensure all values are numeric
        numeric_data = []
        for item in price_data:
            try:
                numeric_data.append(float(item))
            except (ValueError, TypeError):
                continue
        
        if len(numeric_data) < 20:
            return "insufficient_data"
        
        trend = self._calculate_trend(numeric_data, 20)
        volatility = self._calculate_volatility(numeric_data)
        
        if trend == "neutral":
            return "weak"
        elif volatility < 0.02:
            return "strong"
        elif volatility < 0.05:
            return "moderate"
        else:
            return "weak"
    
    def _extract_reversal_levels(self, indicators: Dict[str, Any]) -> List[float]:
        """Extract potential reversal levels."""
        levels = []
        
        if 'close' in indicators:
            close_data = indicators['close']
            if isinstance(close_data, list) and len(close_data) >= 20:
                # Ensure all values are numeric
                numeric_data = []
                for item in close_data:
                    try:
                        numeric_data.append(float(item))
                    except (ValueError, TypeError):
                        continue
                
                if len(numeric_data) >= 20:
                    levels.extend([min(numeric_data[-20:]), max(numeric_data[-20:])])
        
        return levels
    
    def _detect_rsi_divergence(self, price_data: List[float], rsi_data: List[float]) -> bool:
        """Detect RSI divergence from price."""
        # Handle case where data might be a string or other type
        if not isinstance(price_data, list) or not isinstance(rsi_data, list):
            return False
        
        # Ensure all values are numeric
        numeric_price = []
        for item in price_data:
            try:
                numeric_price.append(float(item))
            except (ValueError, TypeError):
                continue
        
        numeric_rsi = []
        for item in rsi_data:
            try:
                numeric_rsi.append(float(item))
            except (ValueError, TypeError):
                continue
        
        if len(numeric_price) < 10 or len(numeric_rsi) < 10:
            return False
        
        price_trend = self._calculate_trend(numeric_price, 10)
        rsi_trend = self._calculate_trend(numeric_rsi, 10)
        
        return price_trend != rsi_trend
    
    def _detect_macd_divergence(self, price_data: List[float], macd_data: Dict[str, List[float]]) -> bool:
        """Detect MACD divergence from price."""
        # Handle case where data might be a string or other type
        if not isinstance(price_data, list):
            return False
        
        # Ensure all values are numeric
        numeric_price = []
        for item in price_data:
            try:
                numeric_price.append(float(item))
            except (ValueError, TypeError):
                continue
        
        if len(numeric_price) < 10:
            return False
        
        price_trend = self._calculate_trend(numeric_price, 10)
        macd_trend = self._calculate_macd_trend(macd_data)
        
        return price_trend != macd_trend
    
    def _calculate_histogram_trend(self, macd_data: Dict[str, List[float]]) -> str:
        """Calculate MACD histogram trend."""
        if not isinstance(macd_data, dict):
            return "insufficient_data"
        
        histogram = macd_data.get('histogram', [])
        if not isinstance(histogram, list) or len(histogram) < 5:
            return "insufficient_data"
        
        return self._calculate_trend(histogram, 5)
    
    def _extract_support_levels(self, indicators: Dict[str, Any]) -> List[float]:
        """Extract support levels."""
        levels = []
        
        if 'close' in indicators:
            close_data = indicators['close']
            if isinstance(close_data, list) and len(close_data) >= 20:
                # Ensure all values are numeric
                numeric_data = []
                for item in close_data:
                    try:
                        numeric_data.append(float(item))
                    except (ValueError, TypeError):
                        continue
                
                if len(numeric_data) >= 20:
                    levels.append(min(numeric_data[-20:]))
        
        if 'sma' in indicators:
            sma_data = indicators['sma']
            if isinstance(sma_data, dict):
                if 20 in sma_data and isinstance(sma_data[20], list) and len(sma_data[20]) > 0:
                    try:
                        levels.append(float(sma_data[20][-1]))
                    except (ValueError, TypeError):
                        pass
                if 50 in sma_data and isinstance(sma_data[50], list) and len(sma_data[50]) > 0:
                    try:
                        levels.append(float(sma_data[50][-1]))
                    except (ValueError, TypeError):
                        pass
        
        return sorted(levels, reverse=True)
    
    def _extract_resistance_levels(self, indicators: Dict[str, Any]) -> List[float]:
        """Extract resistance levels."""
        levels = []
        
        if 'close' in indicators:
            close_data = indicators['close']
            if isinstance(close_data, list) and len(close_data) >= 20:
                # Ensure all values are numeric
                numeric_data = []
                for item in close_data:
                    try:
                        numeric_data.append(float(item))
                    except (ValueError, TypeError):
                        continue
                
                if len(numeric_data) >= 20:
                    levels.append(max(numeric_data[-20:]))
        
        if 'sma' in indicators:
            sma_data = indicators['sma']
            if isinstance(sma_data, dict):
                if 20 in sma_data and isinstance(sma_data[20], list) and len(sma_data[20]) > 0:
                    try:
                        levels.append(float(sma_data[20][-1]))
                    except (ValueError, TypeError):
                        pass
                if 50 in sma_data and isinstance(sma_data[50], list) and len(sma_data[50]) > 0:
                    try:
                        levels.append(float(sma_data[50][-1]))
                    except (ValueError, TypeError):
                        pass
        
        return sorted(levels)
    
    def _assess_breakout_potential(self, indicators: Dict[str, Any]) -> str:
        """Assess breakout potential."""
        if 'close' in indicators and 'sma' in indicators:
            close_data = indicators['close']
            sma_data = indicators['sma']
            
            if len(close_data) > 0 and 20 in sma_data:
                current_price = close_data[-1]
                sma_20 = sma_data[20][-1]
                
                if current_price > sma_20 * 1.02:
                    return "high"
                elif current_price > sma_20:
                    return "medium"
                else:
                    return "low"
        
        return "unknown"
    
    def _compare_price_to_sma(self, price_data: List[float], sma_data: Dict[int, List[float]], period: int) -> str:
        """Compare current price to SMA."""
        if len(price_data) == 0 or period not in sma_data:
            return "insufficient_data"
        
        current_price = price_data[-1]
        sma_value = sma_data[period][-1]
        
        if current_price > sma_value:
            return "above"
        else:
            return "below"
    
    def _check_sma_alignment(self, sma_data: Dict[int, List[float]]) -> str:
        """Check if moving averages are aligned."""
        if 20 not in sma_data or 50 not in sma_data:
            return "insufficient_data"
        
        sma_20 = sma_data[20][-1]
        sma_50 = sma_data[50][-1]
        
        if sma_20 > sma_50:
            return "bullish"
        else:
            return "bearish"
    
    def _extract_key_metrics(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics for general analysis."""
        metrics = {}
        
        if 'close' in indicators:
            close_data = indicators['close']
            if len(close_data) > 0:
                metrics['current_price'] = close_data[-1]
                metrics['price_trend'] = self._calculate_trend(close_data, 20)
        
        if 'rsi' in indicators:
            rsi_data = indicators['rsi']
            if len(rsi_data) > 0:
                metrics['current_rsi'] = rsi_data[-1]
        
        return metrics
    
    def _extract_trend_context(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Extract trend context for general analysis."""
        context = {}
        
        if 'close' in indicators:
            close_data = indicators['close']
            context['trend_direction'] = self._calculate_trend(close_data, 20)
            context['trend_strength'] = self._calculate_trend_strength(close_data)
        
        return context 

    def _extract_enhanced_critical_levels(self, indicators: Dict[str, Any]) -> Dict[str, List[float]]:
        """Extract enhanced critical support and resistance levels using Level 2 analysis."""
        levels = {"support": [], "resistance": []}
        
        # Use enhanced levels if available (Level 2: Swing Point Analysis)
        if 'enhanced_levels' in indicators:
            enhanced = indicators['enhanced_levels']
            if 'dynamic_support' in enhanced:
                levels["support"] = enhanced['dynamic_support']
            if 'dynamic_resistance' in enhanced:
                levels["resistance"] = enhanced['dynamic_resistance']
        else:
            # Fallback to basic levels if enhanced not available
            levels = self._extract_critical_levels(indicators)
        
        return levels
    
    def _extract_enhanced_support_levels(self, indicators: Dict[str, Any]) -> List[float]:
        """Extract enhanced support levels using Level 2 analysis."""
        if 'enhanced_levels' in indicators:
            enhanced = indicators['enhanced_levels']
            if 'dynamic_support' in enhanced:
                return enhanced['dynamic_support']
        
        # Fallback to basic support levels
        return self._extract_support_levels(indicators)
    
    def _extract_enhanced_resistance_levels(self, indicators: Dict[str, Any]) -> List[float]:
        """Extract enhanced resistance levels using Level 2 analysis."""
        if 'enhanced_levels' in indicators:
            enhanced = indicators['enhanced_levels']
            if 'dynamic_resistance' in enhanced:
                return enhanced['dynamic_resistance']
        
        # Fallback to basic resistance levels
        return self._extract_resistance_levels(indicators) 

    def _make_json_safe(self, obj):
        """
        Convert an object to be JSON-serializable by handling non-serializable types.
        """
        if obj is None:
            return None
        elif isinstance(obj, bool):
            return obj
        elif isinstance(obj, int):
            return obj
        elif isinstance(obj, float):
            # Round floats to 2 decimal places for cleaner prompts
            return round(obj, 2)
        elif isinstance(obj, str):
            return obj
        elif isinstance(obj, list):
            return [self._make_json_safe(item) for item in obj]
        elif isinstance(obj, dict):
            return {str(key): self._make_json_safe(value) for key, value in obj.items()}
        else:
            # Convert any other type to string
            try:
                return str(obj)
            except:
                return "unserializable_object"
