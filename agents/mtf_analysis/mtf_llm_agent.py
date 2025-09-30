#!/usr/bin/env python3
"""
MTF LLM Agent

This agent takes MTF analysis results and sends them to the LLM for interpretation.
It provides natural language analysis of multi-timeframe patterns, conflicts, and trading opportunities.

Similar to indicator agents, this agent:
- Receives MTF technical analysis data
- Formats it for LLM consumption
- Calls Gemini to generate insights
- Returns structured analysis for the final decision LLM
"""

import logging
import asyncio
import time
import json
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class MTFLLMAgent:
    """
    MTF LLM Agent that analyzes multi-timeframe data using Gemini.
    
    This agent bridges the gap between technical MTF analysis and natural language insights.
    """
    
    def __init__(self):
        self.agent_name = "mtf_llm_agent"
        # Import here to avoid circular dependencies
        from gemini.gemini_client import GeminiClient
        self.gemini_client = GeminiClient()
        logger.info(f"[{self.agent_name.upper()}] Initialized")
    
    async def analyze_mtf_with_llm(
        self, 
        symbol: str,
        exchange: str,
        mtf_analysis: Dict[str, Any],
        context: str = ""
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Analyze MTF data using LLM to generate natural language insights.
        
        Args:
            symbol: Stock symbol
            exchange: Exchange name
            mtf_analysis: MTF analysis results from integration manager
            context: Additional context string
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (success, llm_analysis)
        """
        start_time = time.time()
        logger.info(f"[{self.agent_name.upper()}] Starting LLM analysis for {symbol}")
        print(f"[MTF_LLM_AGENT_DEBUG] Starting MTF LLM analysis for {symbol}...")
        
        try:
            # Step 1: Build the prompt from MTF data
            prompt = self._build_mtf_prompt(symbol, exchange, mtf_analysis, context)
            
            if not prompt:
                logger.error(f"[{self.agent_name.upper()}] Failed to build prompt for {symbol}")
                return False, {"error": "Failed to build MTF prompt", "agent": self.agent_name}
            
            logger.info(f"[{self.agent_name.upper()}] Built prompt ({len(prompt)} chars) for {symbol}")
            
            # Step 2: Call LLM with the prompt
            print(f"[MTF_LLM_AGENT_DEBUG] Sending request to Gemini API for {symbol}...")
            llm_start = time.time()
            
            try:
                # Use the core LLM call with code execution capability
                text_response, code_results, exec_results = await self.gemini_client.core.call_llm_with_code_execution(
                    prompt=prompt
                )
                
                llm_time = time.time() - llm_start
                print(f"[MTF_LLM_AGENT_DEBUG] Received response from Gemini API for {symbol} in {llm_time:.2f}s")
                logger.info(f"[{self.agent_name.upper()}] LLM response received in {llm_time:.2f}s for {symbol}")
                
            except Exception as llm_error:
                logger.error(f"[{self.agent_name.upper()}] LLM call failed for {symbol}: {llm_error}")
                print(f"[MTF_LLM_AGENT_DEBUG] LLM call failed for {symbol}: {llm_error}")
                return False, {
                    "error": f"LLM call failed: {str(llm_error)}",
                    "agent": self.agent_name
                }
            
            # Step 3: Structure the response
            processing_time = time.time() - start_time
            
            structured_analysis = {
                "success": True,
                "agent": self.agent_name,
                "symbol": symbol,
                "exchange": exchange,
                "timestamp": datetime.now().isoformat(),
                "processing_time": processing_time,
                "llm_processing_time": llm_time,
                
                # LLM Analysis
                "llm_analysis": text_response or "",
                "code_executions": len(code_results) if code_results else 0,
                "calculation_results": len(exec_results) if exec_results else 0,
                
                # Original MTF Summary (for reference)
                "mtf_summary": mtf_analysis.get("summary", {}),
                "cross_timeframe_validation": mtf_analysis.get("cross_timeframe_validation", {}),
                
                # Metadata
                "prompt_length": len(prompt),
                "response_length": len(text_response) if text_response else 0,
                "confidence": self._calculate_confidence(mtf_analysis, text_response)
            }
            
            logger.info(f"[{self.agent_name.upper()}] Successfully completed analysis for {symbol} in {processing_time:.2f}s")
            print(f"[MTF_LLM_AGENT_DEBUG] Finished MTF LLM analysis for {symbol}. Success: True")
            
            return True, structured_analysis
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"MTF LLM analysis failed: {str(e)}"
            logger.error(f"[{self.agent_name.upper()}] {error_msg} for {symbol}")
            print(f"[MTF_LLM_AGENT_DEBUG] Error in MTF LLM analysis for {symbol}: {e}")
            
            return False, {
                "success": False,
                "error": error_msg,
                "agent": self.agent_name,
                "symbol": symbol,
                "processing_time": processing_time
            }
    
    def _build_mtf_prompt(
        self, 
        symbol: str, 
        exchange: str, 
        mtf_analysis: Dict[str, Any],
        context: str
    ) -> str:
        """
        Build the LLM prompt from MTF analysis data.
        
        This creates a comprehensive, structured prompt that asks the LLM to:
        - Analyze cross-timeframe patterns
        - Identify conflicts and resolutions
        - Provide trading insights
        - Suggest entry/exit strategies
        """
        try:
            # Extract key components
            timeframe_analyses = mtf_analysis.get('timeframe_analyses', {})
            validation = mtf_analysis.get('cross_timeframe_validation', {})
            summary = mtf_analysis.get('summary', {})
            agent_insights = mtf_analysis.get('agent_insights', {})
            
            # Build structured prompt
            prompt = f"""# Multi-Timeframe Analysis for {symbol} ({exchange})

## Analysis Request
You are a professional trading analyst. Analyze the following comprehensive multi-timeframe technical analysis and provide actionable insights.

## Overall Summary
- **Overall Signal**: {summary.get('overall_signal', 'unknown').upper()}
- **Confidence**: {summary.get('confidence', 0) * 100:.1f}%
- **Signal Alignment**: {summary.get('signal_alignment', 'unknown')}
- **Risk Level**: {summary.get('risk_level', 'unknown')}
- **Recommendation**: {summary.get('recommendation', 'hold')}

## Cross-Timeframe Validation
- **Consensus Trend**: {validation.get('consensus_trend', 'neutral')}
- **Signal Strength**: {validation.get('signal_strength', 0) * 100:.1f}%
- **Confidence Score**: {validation.get('confidence_score', 0) * 100:.1f}%
- **Supporting Timeframes**: {', '.join(validation.get('supporting_timeframes', [])) or 'None'}
- **Conflicting Timeframes**: {', '.join(validation.get('conflicting_timeframes', [])) or 'None'}
- **Divergence Detected**: {'Yes' if validation.get('divergence_detected') else 'No'}
"""
            
            # Add divergence details if present
            if validation.get('divergence_detected'):
                prompt += f"- **Divergence Type**: {validation.get('divergence_type', 'unknown')}\n"
                
                key_conflicts = validation.get('key_conflicts', [])
                if key_conflicts:
                    prompt += "\n### Key Conflicts:\n"
                    for conflict in key_conflicts:
                        prompt += f"- {conflict}\n"
            
            # Add timeframe-by-timeframe analysis
            prompt += "\n## Timeframe Analysis\n\n"
            
            # Sort timeframes logically
            timeframe_order = ['1min', '5min', '15min', '30min', '1hour', '1day']
            sorted_timeframes = sorted(
                timeframe_analyses.keys(),
                key=lambda x: timeframe_order.index(x) if x in timeframe_order else 999
            )
            
            for tf in sorted_timeframes:
                analysis = timeframe_analyses[tf]
                if not isinstance(analysis, dict):
                    continue
                
                confidence = analysis.get('confidence', 0) * 100
                
                # Add importance indicator
                if confidence >= 80:
                    importance = "🔥 HIGH CONFIDENCE"
                elif confidence >= 60:
                    importance = "⚡ MEDIUM-HIGH CONFIDENCE"
                elif confidence >= 40:
                    importance = "📊 MEDIUM CONFIDENCE"
                else:
                    importance = "⚠️ LOW CONFIDENCE"
                
                prompt += f"""### {tf} Timeframe ({importance})
- **Trend**: {analysis.get('trend', 'unknown')}
- **Confidence**: {confidence:.1f}%
- **Data Points**: {analysis.get('data_points', 0)}

**Key Indicators**:
"""
                
                key_indicators = analysis.get('key_indicators', {})
                if key_indicators:
                    if 'rsi' in key_indicators and key_indicators['rsi'] is not None:
                        prompt += f"- RSI: {key_indicators['rsi']}\n"
                    if 'macd_signal' in key_indicators:
                        prompt += f"- MACD Signal: {key_indicators['macd_signal']}\n"
                    if 'volume_status' in key_indicators:
                        prompt += f"- Volume Status: {key_indicators['volume_status']}\n"
                    
                    support_levels = key_indicators.get('support_levels', [])
                    resistance_levels = key_indicators.get('resistance_levels', [])
                    
                    if support_levels:
                        prompt += f"- Support Levels: {', '.join(f'{s:.2f}' for s in support_levels[:3])}\n"
                    if resistance_levels:
                        prompt += f"- Resistance Levels: {', '.join(f'{r:.2f}' for r in resistance_levels[:3])}\n"
                
                prompt += "\n"
            
            # Add agent system info
            if agent_insights:
                prompt += f"""## Agent System Performance
- Total Agents Run: {agent_insights.get('total_agents_run', 0)}
- Successful Agents: {agent_insights.get('successful_agents', 0)}
- Failed Agents: {agent_insights.get('failed_agents', 0)}
- System Confidence: {agent_insights.get('confidence_score', 0) * 100:.1f}%

"""
            
            # Add additional context if provided
            if context:
                prompt += f"""## Additional Context
{context}

"""
            
            # Add the analysis request
            prompt += """## Analysis Task

Based on the comprehensive multi-timeframe analysis above, provide:

1. **Trend Synthesis**: What is the dominant trend across timeframes? How do shorter and longer timeframes align or conflict?

2. **Key Levels**: Identify the most critical support and resistance levels that traders should watch.

3. **Conflict Resolution**: If there are conflicting signals across timeframes, explain which timeframe should take precedence and why.

4. **Trading Strategy**: 
   - For short-term traders (intraday): What do the 1min-15min timeframes suggest?
   - For swing traders (days-weeks): What do the 30min-1hour timeframes suggest?
   - For position traders (weeks-months): What does the 1day timeframe suggest?

5. **Risk Assessment**: What are the key risks based on the divergences and conflicts?

6. **Entry/Exit Recommendations**: 
   - If the signal is BULLISH: Suggest entry levels, stop-loss, and target prices
   - If the signal is BEARISH: Suggest short entry levels, stop-loss, and cover prices
   - If the signal is NEUTRAL: Suggest wait conditions or range-bound strategies

7. **Confidence Level**: Rate your confidence in this analysis (Low/Medium/High) and explain why.

Be specific, actionable, and reference the actual technical levels and timeframe evidence from the data above.
"""
            
            return prompt
            
        except Exception as e:
            logger.error(f"[{self.agent_name.upper()}] Error building prompt: {e}")
            return ""
    
    def _calculate_confidence(self, mtf_analysis: Dict[str, Any], llm_response: str) -> float:
        """
        Calculate confidence score based on MTF quality and LLM response.
        """
        try:
            # Base confidence from MTF analysis
            mtf_confidence = mtf_analysis.get('summary', {}).get('confidence', 0.5)
            
            # Agent system confidence
            agent_confidence = mtf_analysis.get('agent_insights', {}).get('confidence_score', 0.5)
            
            # LLM response quality factor (longer, more detailed = higher confidence)
            response_quality = min(1.0, len(llm_response) / 2000) if llm_response else 0.5
            
            # Weighted average
            overall_confidence = (
                mtf_confidence * 0.4 +
                agent_confidence * 0.3 +
                response_quality * 0.3
            )
            
            return max(0.0, min(1.0, overall_confidence))
            
        except Exception:
            return 0.5


# Global instance
mtf_llm_agent = MTFLLMAgent()