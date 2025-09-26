#!/usr/bin/env python3
"""
Indicator Agents Integration Manager

Provides a robust, fault-tolerant integration layer between the main analysis system
and the distributed indicator agents orchestrator. Similar to the volume agents 
integration manager, this provides:

- Health monitoring and validation
- Graceful fallback mechanisms
- Performance metrics tracking
- Error recovery and logging
- Standardized interface for the main system

The integration manager ensures that indicator agent failures don't break the entire
analysis pipeline and provides consistent results even when individual agents fail.
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd

from .indicators_agents import indicators_orchestrator, AggregatedIndicatorAnalysis
from ml.indicators.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

@dataclass
class IndicatorAgentsHealthMetrics:
    """Health metrics for the indicator agents system"""
    last_success_time: Optional[float] = None
    consecutive_failures: int = 0
    total_successful_runs: int = 0
    total_failed_runs: int = 0
    average_processing_time: float = 0.0
    last_error_message: Optional[str] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        total = self.total_successful_runs + self.total_failed_runs
        if total == 0:
            return 1.0
        return self.total_successful_runs / total
    
    @property
    def is_healthy(self) -> bool:
        """Determine if the system is healthy"""
        # System is unhealthy if we have too many consecutive failures
        # or if success rate is too low with sufficient samples
        if self.consecutive_failures >= 3:
            return False
        
        total_runs = self.total_successful_runs + self.total_failed_runs
        if total_runs >= 5 and self.success_rate < 0.6:
            return False
            
        return True

class IndicatorAgentIntegrationManager:
    """
    Integration manager for distributed indicator agents system.
    
    Provides a standardized interface to the indicator agents orchestrator with:
    - Health monitoring
    - Fallback mechanisms  
    - Performance tracking
    - Error recovery
    """
    
    def __init__(self):
        self.health_metrics = IndicatorAgentsHealthMetrics()
        self.orchestrator = indicators_orchestrator
        logger.info("IndicatorAgentIntegrationManager initialized")
    
    def is_indicator_agents_healthy(self) -> bool:
        """Check if the indicator agents system is healthy"""
        return self.health_metrics.is_healthy
    
    def get_health_reason(self) -> str:
        """Get a human-readable reason for the current health status"""
        if self.is_indicator_agents_healthy():
            return f"Healthy (Success rate: {self.health_metrics.success_rate:.1%})"
        
        reasons = []
        if self.health_metrics.consecutive_failures >= 3:
            reasons.append(f"Too many consecutive failures ({self.health_metrics.consecutive_failures})")
        
        total_runs = self.health_metrics.total_successful_runs + self.health_metrics.total_failed_runs
        if total_runs >= 5 and self.health_metrics.success_rate < 0.6:
            reasons.append(f"Low success rate ({self.health_metrics.success_rate:.1%})")
        
        return "; ".join(reasons) if reasons else "Unknown health issue"
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        return {
            'health_status': 'healthy' if self.is_indicator_agents_healthy() else 'unhealthy',
            'health_reason': self.get_health_reason(),
            'success_rate': self.health_metrics.success_rate,
            'consecutive_failures': self.health_metrics.consecutive_failures,
            'total_successful_runs': self.health_metrics.total_successful_runs,
            'total_failed_runs': self.health_metrics.total_failed_runs,
            'average_processing_time': self.health_metrics.average_processing_time,
            'last_error_message': self.health_metrics.last_error_message,
            'last_success_time': self.health_metrics.last_success_time
        }
    
    async def get_curated_indicators_analysis(
        self, 
        symbol: str, 
        stock_data: pd.DataFrame, 
        indicators: Dict[str, Any],
        context: str = ""
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Get curated indicators analysis from the indicator agents.
        
        Returns:
            Tuple[bool, Dict[str, Any]]: (success, curated_indicators_dict)
        """
        logger.info(f"[INDICATOR_AGENTS] Starting curated indicators analysis for {symbol}")
        start_time = time.time()
        
        try:
            # Check if system is healthy enough to attempt analysis
            if not self.is_indicator_agents_healthy():
                logger.warning(f"[INDICATOR_AGENTS] System unhealthy: {self.get_health_reason()}")
                return False, self._create_fallback_curated_indicators(indicators, stock_data)
            
            # Run the indicator agents orchestrator
            agent_result = await self.orchestrator.analyze_indicators_comprehensive(
                symbol=symbol,
                stock_data=stock_data, 
                indicators=indicators,
                context=context
            )
            
            processing_time = time.time() - start_time
            
            # Check if the analysis was successful
            if agent_result.successful_agents == 0:
                logger.warning(f"[INDICATOR_AGENTS] All agents failed for {symbol}")
                self._record_failure("All indicator agents failed", processing_time)
                return False, self._create_fallback_curated_indicators(indicators, stock_data)
            
            # Convert to curated format
            curated_indicators = self._curate_indicators_from_agents(
                agent_result.unified_analysis,
                raw_indicators=indicators,
                stock_data=stock_data
            )
            
            # Record success
            self._record_success(processing_time)
            
            logger.info(f"[INDICATOR_AGENTS] Successfully curated indicators with {agent_result.successful_agents}/{agent_result.successful_agents + agent_result.failed_agents} agents")
            
            return True, curated_indicators
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_message = str(e)
            logger.error(f"[INDICATOR_AGENTS] Analysis failed for {symbol}: {error_message}")
            
            self._record_failure(error_message, processing_time)
            
            # Return fallback result
            return False, self._create_fallback_curated_indicators(indicators, stock_data)
    
    def _curate_indicators_from_agents(
        self, 
        unified: Dict[str, Any], 
        raw_indicators: Dict[str, Any] = None, 
        stock_data: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """
        Convert IndicatorAgentsOrchestrator unified_analysis into the curated structure
        expected by the GeminiClient indicator summary template.
        """
        try:
            indicator_summary = (unified or {}).get('indicator_summary', {})
            signal_consensus = (unified or {}).get('signal_consensus', {})

            key_indicators: Dict[str, Any] = {}

            # Base: agent high-level trend summary
            trend = indicator_summary.get('trend') if isinstance(indicator_summary, dict) else None
            if isinstance(trend, dict):
                key_indicators["trend_indicators"] = {
                    "direction": trend.get('direction', 'neutral'),
                    "strength": trend.get('strength', 'weak'),
                    "confidence": trend.get('confidence', 0.0),
                }
            else:
                key_indicators["trend_indicators"] = {"direction": "neutral", "strength": "weak", "confidence": 0.0}

            # Merge numeric MAs if available from raw indicators
            try:
                if isinstance(raw_indicators, dict):
                    # Prefer flattened moving_averages structure if present
                    mov = raw_indicators.get('moving_averages')
                    if isinstance(mov, dict):
                        def fget(d, k):
                            v = d.get(k)
                            try:
                                return float(v) if v is not None else None
                            except Exception:
                                return None
                        sma_20 = fget(mov, 'sma_20')
                        sma_50 = fget(mov, 'sma_50')
                        sma_200 = fget(mov, 'sma_200')
                        ema_20 = fget(mov, 'ema_20')
                        ema_50 = fget(mov, 'ema_50')
                        if sma_20 is not None: key_indicators["trend_indicators"]["sma_20"] = round(sma_20, 2)
                        if sma_50 is not None: key_indicators["trend_indicators"]["sma_50"] = round(sma_50, 2)
                        if sma_200 is not None: key_indicators["trend_indicators"]["sma_200"] = round(sma_200, 2)
                        if ema_20 is not None: key_indicators["trend_indicators"]["ema_20"] = round(ema_20, 2)
                        if ema_50 is not None: key_indicators["trend_indicators"]["ema_50"] = round(ema_50, 2)
                        # percentage metrics might be in mov as decimals or percents; if percents, convert to fraction
                        p2sma200 = mov.get('price_to_sma_200')
                        if isinstance(p2sma200, (int, float)):
                            key_indicators["trend_indicators"]["price_to_sma_200"] = round(float(p2sma200), 2)
                        s20_to_50 = mov.get('sma_20_to_sma_50')
                        if isinstance(s20_to_50, (int, float)):
                            key_indicators["trend_indicators"]["sma_20_to_sma_50"] = round(float(s20_to_50), 2)
                        gc = mov.get('golden_cross')
                        dc = mov.get('death_cross')
                        if isinstance(gc, bool): key_indicators["trend_indicators"]["golden_cross"] = gc
                        if isinstance(dc, bool): key_indicators["trend_indicators"]["death_cross"] = dc
            except Exception:
                pass

            # Momentum block from agent plus numeric values
            key_indicators.setdefault("momentum_indicators", {})
            momentum = indicator_summary.get('momentum') if isinstance(indicator_summary, dict) else None
            if isinstance(momentum, dict):
                key_indicators["momentum_indicators"].update({
                    "rsi_status": momentum.get('rsi_signal', 'neutral'),
                    "direction": momentum.get('direction', 'neutral'),
                    "strength": momentum.get('strength', 'weak'),
                    "confidence": momentum.get('confidence', 0.0),
                })
            # Numeric RSI/MACD from raw indicators
            try:
                if isinstance(raw_indicators, dict):
                    # Handle rsi as dict or list
                    rsi = raw_indicators.get('rsi')
                    if isinstance(rsi, dict):
                        rv = rsi.get('rsi_14')
                        if isinstance(rv, (int, float)):
                            key_indicators["momentum_indicators"]["rsi_current"] = round(float(rv), 2)
                        status = rsi.get('status')
                        if isinstance(status, str):
                            key_indicators["momentum_indicators"].setdefault("rsi_status", status)
                    elif isinstance(rsi, (list, tuple)) and rsi:
                        key_indicators["momentum_indicators"]["rsi_current"] = round(float(rsi[-1]), 2)
                    # MACD dict with histogram
                    macd = raw_indicators.get('macd')
                    if isinstance(macd, dict):
                        hist = macd.get('histogram')
                        hist_val = None
                        if isinstance(hist, (list, tuple)) and hist:
                            hist_val = float(hist[-1])
                        elif isinstance(hist, (int, float)):
                            hist_val = float(hist)
                        if hist_val is not None:
                            key_indicators["momentum_indicators"]["macd"] = {
                                "histogram": round(hist_val, 2),
                                "trend": key_indicators["momentum_indicators"].get("direction", "neutral")
                            }
            except Exception:
                pass

            # Volume indicators (optional)
            try:
                if isinstance(raw_indicators, dict):
                    vol = raw_indicators.get('volume') or {}
                    vol_ratio = None
                    if isinstance(vol, dict):
                        vol_ratio = vol.get('volume_ratio')
                    ki_vol = {}
                    if isinstance(vol_ratio, (int, float)):
                        ki_vol["volume_ratio"] = round(float(vol_ratio), 2)
                    # best-effort trend
                    if ki_vol:
                        ki_vol.setdefault("volume_trend", "neutral")
                        key_indicators["volume_indicators"] = ki_vol
            except Exception:
                pass

            # Conflicts: prefer ContextEngineer detailed analysis based on numeric indicators
            detected_conflicts = {
                "has_conflicts": False,
                "conflict_count": 0,
                "conflict_list": []
            }
            try:
                from gemini.context_engineer import ContextEngineer
                ce = ContextEngineer()
                ce_conf = ce._comprehensive_conflict_analysis(key_indicators)
                if isinstance(ce_conf, dict):
                    detected_conflicts = ce_conf
                else:
                    raise ValueError("Invalid conflict data")
            except Exception:
                # Fallback: use agent consensus if mixed
                if isinstance(signal_consensus, dict) and signal_consensus.get('consensus') == 'mixed':
                    detected_conflicts.update({
                        "has_conflicts": True,
                        "conflict_count": 1,
                        "conflict_list": ["Mixed consensus across indicator agents"]
                    })

            # Critical levels: best-effort support/resistance via TechnicalIndicators if stock_data available
            critical_levels = {}
            try:
                if stock_data is not None and not stock_data.empty:
                    support_levels, resistance_levels = TechnicalIndicators.detect_support_resistance(stock_data)
                    # Deduplicate and round
                    if support_levels:
                        sl = sorted(set(float(x) for x in support_levels), reverse=True)
                        critical_levels["support"] = [round(v, 2) for v in sl[:3]]
                    if resistance_levels:
                        rl = sorted(set(float(x) for x in resistance_levels))
                        critical_levels["resistance"] = [round(v, 2) for v in rl[:3]]
            except Exception as _e:
                # Non-fatal; leave levels empty
                pass

            curated = {
                "analysis_focus": "technical_indicators_summary",
                "key_indicators": key_indicators,
                "critical_levels": critical_levels,
                "conflict_analysis_needed": detected_conflicts["has_conflicts"],
                "detected_conflicts": detected_conflicts,
            }
            return curated
        except Exception as ex:
            logger.error(f"[INDICATOR_AGENTS] Agent curation failed: {ex}")
            return {
                "analysis_focus": "technical_indicators_summary",
                "key_indicators": {},
                "critical_levels": {},
                "conflict_analysis_needed": False,
                "detected_conflicts": {"has_conflicts": False, "conflict_count": 0, "conflict_list": []}
            }
    
    def _create_fallback_curated_indicators(
        self, 
        indicators: Dict[str, Any], 
        stock_data: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """
        Create a fallback curated indicators structure when agents fail.
        This uses the raw indicators directly to maintain functionality.
        """
        try:
            logger.info("[INDICATOR_AGENTS] Creating fallback curated indicators")
            
            key_indicators = {}
            
            # Create basic trend indicators from raw data
            if isinstance(indicators, dict):
                # Moving averages
                mov = indicators.get('moving_averages', {})
                if isinstance(mov, dict):
                    trend_indicators = {
                        "direction": "neutral",
                        "strength": "weak", 
                        "confidence": 0.3  # Low confidence for fallback
                    }
                    
                    # Add numeric values if available
                    for key in ['sma_20', 'sma_50', 'sma_200', 'ema_20', 'ema_50']:
                        if key in mov:
                            try:
                                trend_indicators[key] = round(float(mov[key]), 2)
                            except:
                                pass
                    
                    # Basic trend determination from moving averages
                    try:
                        if stock_data is not None and not stock_data.empty:
                            current_price = float(stock_data['close'].iloc[-1])
                            sma_20 = mov.get('sma_20')
                            sma_50 = mov.get('sma_50')
                            
                            if sma_20 and sma_50:
                                sma_20 = float(sma_20)
                                sma_50 = float(sma_50)
                                
                                if current_price > sma_20 > sma_50:
                                    trend_indicators["direction"] = "bullish"
                                    trend_indicators["strength"] = "moderate"
                                elif current_price < sma_20 < sma_50:
                                    trend_indicators["direction"] = "bearish"
                                    trend_indicators["strength"] = "moderate"
                    except:
                        pass
                    
                    key_indicators["trend_indicators"] = trend_indicators
                
                # Momentum indicators
                momentum_indicators = {
                    "direction": "neutral",
                    "strength": "weak",
                    "confidence": 0.3,
                    "rsi_status": "neutral"
                }
                
                # RSI
                rsi = indicators.get('rsi')
                if isinstance(rsi, dict) and 'rsi_14' in rsi:
                    try:
                        rsi_val = float(rsi['rsi_14'])
                        momentum_indicators["rsi_current"] = round(rsi_val, 2)
                        
                        if rsi_val > 70:
                            momentum_indicators["rsi_status"] = "overbought"
                        elif rsi_val < 30:
                            momentum_indicators["rsi_status"] = "oversold"
                        else:
                            momentum_indicators["rsi_status"] = "neutral"
                    except:
                        pass
                
                # MACD  
                macd = indicators.get('macd')
                if isinstance(macd, dict) and 'histogram' in macd:
                    try:
                        hist_val = float(macd['histogram'])
                        momentum_indicators["macd"] = {
                            "histogram": round(hist_val, 2),
                            "trend": "bullish" if hist_val > 0 else "bearish"
                        }
                        
                        if hist_val > 0:
                            momentum_indicators["direction"] = "bullish"
                        elif hist_val < 0:
                            momentum_indicators["direction"] = "bearish"
                    except:
                        pass
                
                key_indicators["momentum_indicators"] = momentum_indicators
                
                # Volume indicators
                vol = indicators.get('volume', {})
                if isinstance(vol, dict) and 'volume_ratio' in vol:
                    try:
                        vol_ratio = float(vol['volume_ratio'])
                        key_indicators["volume_indicators"] = {
                            "volume_ratio": round(vol_ratio, 2),
                            "volume_trend": "high" if vol_ratio > 1.5 else "low" if vol_ratio < 0.8 else "neutral"
                        }
                    except:
                        pass
            
            # Critical levels using TechnicalIndicators
            critical_levels = {}
            try:
                if stock_data is not None and not stock_data.empty:
                    support_levels, resistance_levels = TechnicalIndicators.detect_support_resistance(stock_data)
                    if support_levels:
                        sl = sorted(set(float(x) for x in support_levels), reverse=True)
                        critical_levels["support"] = [round(v, 2) for v in sl[:3]]
                    if resistance_levels:
                        rl = sorted(set(float(x) for x in resistance_levels))
                        critical_levels["resistance"] = [round(v, 2) for v in rl[:3]]
            except:
                pass
            
            return {
                "analysis_focus": "technical_indicators_summary",
                "key_indicators": key_indicators,
                "critical_levels": critical_levels,
                "conflict_analysis_needed": False,
                "detected_conflicts": {"has_conflicts": False, "conflict_count": 0, "conflict_list": []},
                "fallback_used": True
            }
            
        except Exception as e:
            logger.error(f"[INDICATOR_AGENTS] Fallback creation failed: {e}")
            # Return minimal fallback
            return {
                "analysis_focus": "technical_indicators_summary",
                "key_indicators": {},
                "critical_levels": {},
                "conflict_analysis_needed": False,
                "detected_conflicts": {"has_conflicts": False, "conflict_count": 0, "conflict_list": []},
                "fallback_used": True,
                "error": "Fallback creation failed"
            }
    
    def _record_success(self, processing_time: float):
        """Record a successful analysis"""
        self.health_metrics.total_successful_runs += 1
        self.health_metrics.consecutive_failures = 0
        self.health_metrics.last_success_time = time.time()
        
        # Update average processing time
        total_runs = self.health_metrics.total_successful_runs + self.health_metrics.total_failed_runs
        if total_runs == 1:
            self.health_metrics.average_processing_time = processing_time
        else:
            # Running average
            self.health_metrics.average_processing_time = (
                (self.health_metrics.average_processing_time * (total_runs - 1) + processing_time) / total_runs
            )
    
    def _record_failure(self, error_message: str, processing_time: float):
        """Record a failed analysis"""
        self.health_metrics.total_failed_runs += 1
        self.health_metrics.consecutive_failures += 1
        self.health_metrics.last_error_message = error_message
        
        # Update average processing time (even for failures)
        total_runs = self.health_metrics.total_successful_runs + self.health_metrics.total_failed_runs
        if total_runs == 1:
            self.health_metrics.average_processing_time = processing_time
        else:
            # Running average
            self.health_metrics.average_processing_time = (
                (self.health_metrics.average_processing_time * (total_runs - 1) + processing_time) / total_runs
            )

# Global instance
indicator_agent_integration_manager = IndicatorAgentIntegrationManager()