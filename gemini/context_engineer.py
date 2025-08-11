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
    
    def __init__(self, config: ContextConfig = None):
        self.config = config or ContextConfig()
    
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
            "enhanced_levels": {},  # NEW: Add enhanced levels
            "conflict_analysis_needed": False,
            "mathematical_validation_required": True
        }
        
        # Extract essential trend indicators
        if 'moving_averages' in indicators:
            # New optimized structure
            ma_data = indicators['moving_averages']
            curated["key_indicators"]["trend_indicators"] = {
                "sma_20": ma_data.get('sma_20'),
                "sma_50": ma_data.get('sma_50'),
                "sma_200": ma_data.get('sma_200'),
                "ema_20": ma_data.get('ema_20'),
                "ema_50": ma_data.get('ema_50'),
                "price_to_sma_200": ma_data.get('price_to_sma_200'),
                "sma_20_to_sma_50": ma_data.get('sma_20_to_sma_50'),
                "golden_cross": ma_data.get('golden_cross'),
                "death_cross": ma_data.get('death_cross')
            }
        elif 'sma' in indicators:
            # Old structure
            curated["key_indicators"]["trend_indicators"] = {
                "sma_20": self._get_latest_value(indicators['sma'], 20),
                "sma_50": self._get_latest_value(indicators['sma'], 50),
                "sma_200": self._get_latest_value(indicators['sma'], 200)
            }
        
        # Extract momentum indicators
        if 'rsi' in indicators:
            rsi_data = indicators['rsi']
            # Handle both old (list) and new (dict) RSI structure
            if isinstance(rsi_data, dict) and 'rsi_14' in rsi_data:
                # New optimized structure
                rsi_current = rsi_data.get('rsi_14')
                rsi_trend = rsi_data.get('trend', 'neutral')
                rsi_status = rsi_data.get('status', 'neutral')
                curated["key_indicators"]["momentum_indicators"] = {
                    "rsi_current": rsi_current,
                    "rsi_trend": rsi_trend,
                    "rsi_status": rsi_status,
                    "rsi_extremes": {"oversold": 0, "overbought": 0}  # Simplified for optimized structure
                }
            else:
                # Old structure (list)
                curated["key_indicators"]["momentum_indicators"] = {
                    "rsi_current": self._get_latest_value(rsi_data),
                    "rsi_trend": self._calculate_trend(rsi_data, 14),
                    "rsi_extremes": self._count_extremes(rsi_data)
                }
        
        if 'macd' in indicators:
            macd_data = indicators['macd']
            # Handle both old (dict with lists) and new (dict with single values) MACD structure
            if isinstance(macd_data, dict) and 'macd_line' in macd_data:
                # New optimized structure
                curated["key_indicators"]["momentum_indicators"]["macd"] = {
                    "macd_line": macd_data.get('macd_line'),
                    "signal_line": macd_data.get('signal_line'),
                    "histogram": macd_data.get('histogram'),
                    "trend": "bullish" if macd_data.get('histogram', 0) > 0 else "bearish"
                }
            else:
                # Old structure
                curated["key_indicators"]["momentum_indicators"]["macd"] = {
                    "signal": self._get_latest_value(macd_data.get('signal', [])),
                    "histogram": self._get_latest_value(macd_data.get('histogram', [])),
                    "trend": self._calculate_macd_trend(macd_data)
                }
        
        # Extract volume indicators
        if 'volume' in indicators:
            volume_data = indicators['volume']
            # Handle both old (list) and new (dict) volume structure
            if isinstance(volume_data, dict) and 'volume_ratio' in volume_data:
                # New optimized structure
                curated["key_indicators"]["volume_indicators"] = {
                    "volume_ratio": volume_data.get('volume_ratio'),
                    "volume_trend": volume_data.get('volume_trend', 'neutral'),
                    "obv": volume_data.get('obv'),
                    "volume_sma": volume_data.get('volume_sma')
                }
            else:
                # Old structure (list)
                curated["key_indicators"]["volume_indicators"] = {
                    "volume_sma": self._get_latest_value(indicators.get('volume_sma', [])),
                    "volume_ratio": self._calculate_volume_ratio(volume_data),
                    "volume_trend": self._calculate_trend(volume_data, 20)
                }
        
        # Extract critical levels (Level 2: Swing Point Analysis)
        curated["critical_levels"] = self._extract_enhanced_critical_levels(indicators)
        
        # Extract enhanced levels (Level 3: Comprehensive Analysis)
        if 'enhanced_levels' in indicators:
            curated["enhanced_levels"] = indicators['enhanced_levels']
        
        # Detect conflicts
        curated["conflict_analysis_needed"] = self._detect_conflicts(curated["key_indicators"])
        
        # Enforce JSON-safe structure
        try:
            curated = self._make_json_safe(curated)
        except Exception:
            pass
        try:
            curated = self._make_json_safe(curated)
        except Exception:
            pass
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
        Structure context for indicator summary analysis.
        """
        try:
            # Ensure all data is JSON-serializable
            safe_key_indicators = self._make_json_safe(curated_data.get('key_indicators', {}))
            safe_critical_levels = self._make_json_safe(curated_data.get('critical_levels', {}))
            safe_enhanced_levels = self._make_json_safe(curated_data.get('enhanced_levels', {}))
            
            context = f"""## Analysis Context:
**Stock Symbol**: {symbol}
**Timeframe**: {timeframe}
**Analysis Focus**: {curated_data.get('analysis_focus', 'technical_indicators_summary')}

## Key Technical Indicators:
{json.dumps(safe_key_indicators, indent=2)}

## Critical Levels (Level 2 - Swing Point Analysis):
{json.dumps(safe_critical_levels, indent=2)}

## Enhanced Levels (Level 3 - Comprehensive Analysis):
{json.dumps(safe_enhanced_levels, indent=2)}

## Analysis Requirements:
- Conflict Analysis Needed: {curated_data.get('conflict_analysis_needed', False)}
- Mathematical Validation Required: {curated_data.get('mathematical_validation_required', True)}

## Knowledge Context:
{knowledge_context}

## Analysis Instructions:
1. Focus on the key indicators provided above
2. Use Level 2 (swing point) levels as primary support/resistance
3. Use Level 3 (enhanced) levels for confirmation and additional context
4. Identify and resolve any signal conflicts
5. Provide specific, actionable trading guidance
6. Include mathematical validation for all key metrics
7. Distinguish between short-term, medium-term, and long-term signals
8. Assess level confirmation between Level 2 and Level 3 methods"""
            
            return context
        except Exception as e:
            # Fallback to simple context if JSON serialization fails
            return f"""## Analysis Context:
**Stock Symbol**: {symbol}
**Timeframe**: {timeframe}
**Analysis Focus**: {curated_data.get('analysis_focus', 'technical_indicators_summary')}

## Knowledge Context:
{knowledge_context}

## Analysis Instructions:
1. Focus on the key indicators provided above
2. Use Level 2 (swing point) levels as primary support/resistance
3. Use Level 3 (enhanced) levels for confirmation and additional context
4. Identify and resolve any signal conflicts
5. Provide specific, actionable trading guidance
6. Include mathematical validation for all key metrics
7. Distinguish between short-term, medium-term, and long-term signals
8. Assess level confirmation between Level 2 and Level 3 methods"""
    
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
6. **Psychological Levels**: Consider round number levels for market psychology
7. **Breakout Assessment**: Determine breakout/breakdown potential based on level strength
8. **Entry/Exit Planning**: Provide specific entry/exit levels with confidence ratings
9. **Risk Management**: Use level confirmation to assess pattern failure risk"""
            
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
6. **Psychological Levels**: Consider round number levels for market psychology
7. **Breakout Assessment**: Determine breakout/breakdown potential based on level strength
8. **Entry/Exit Planning**: Provide specific entry/exit levels with confidence ratings
9. **Risk Management**: Use level confirmation to assess pattern failure risk"""
    
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
        """Extract critical support and resistance levels."""
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
        
        # Extract from price data
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
                    levels["support"].append(min(numeric_data[-20:]))
                    levels["resistance"].append(max(numeric_data[-20:]))
        
        return levels
    
    def _detect_conflicts(self, key_indicators: Dict[str, Any]) -> bool:
        """Detect if there are conflicting signals in the indicators."""
        conflicts = []
        
        # Check RSI vs trend
        if 'momentum_indicators' in key_indicators:
            rsi_data = key_indicators['momentum_indicators'].get('rsi', {})
            if isinstance(rsi_data, dict):
                rsi_current = rsi_data.get('rsi_current', 50)
                try:
                    rsi_current = float(rsi_current)
                    if rsi_current > 70:
                        conflicts.append("RSI overbought")
                    elif rsi_current < 30:
                        conflicts.append("RSI oversold")
                except (ValueError, TypeError):
                    pass
        
        # Check MACD vs trend
        if 'momentum_indicators' in key_indicators:
            macd_data = key_indicators['momentum_indicators'].get('macd', {})
            if isinstance(macd_data, dict) and macd_data.get('trend') == 'bearish' and 'trend_indicators' in key_indicators:
                conflicts.append("MACD bearish vs trend")
        
        return len(conflicts) > 0
    
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
        elif isinstance(obj, (str, int, float, bool)):
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