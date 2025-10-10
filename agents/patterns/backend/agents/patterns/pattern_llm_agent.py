"""
Pattern LLM Agent

Main orchestrator for pattern analysis with LLM synthesis.
Coordinates pattern detection through PatternAgentsOrchestrator and provides
LLM-based analysis and trading insights.
"""

import logging
import asyncio
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd

from .patterns_agents import PatternAgentsOrchestrator
from .pattern_context_builder import PatternContextBuilder

logger = logging.getLogger(__name__)


class PatternLLMAgent:
    """
    Main Pattern LLM Agent that orchestrates comprehensive pattern analysis
    and provides LLM-based synthesis and trading insights.
    
    Follows the same pattern as MTF and Risk agents:
    1. Technical analysis (rule-based pattern detection via PatternAgentsOrchestrator)
    2. LLM synthesis (natural language analysis and trading recommendations)
    """
    
    def __init__(self, gemini_client=None):
        """
        Initialize Pattern LLM Agent.
        
        Args:
            gemini_client: LLM client for analysis (will be imported if None)
        """
        self.name = "pattern_llm"
        self.version = "1.0.0"
        
        # Initialize pattern orchestrator for technical analysis
        self.pattern_orchestrator = PatternAgentsOrchestrator(gemini_client)
        
        # Initialize context builder
        self.context_builder = PatternContextBuilder()
        
        # Initialize LLM client
        if gemini_client:
            self.llm_client = gemini_client
        else:
            self.llm_client = self._get_llm_client()
        
        logger.info(f"[PATTERN_LLM] Initialized Pattern LLM Agent v{self.version}")
    
    def _get_llm_client(self):
        """Get LLM client using the new backend/llm system."""
        try:
            from llm.clients.factory import get_llm_client
            client = get_llm_client()
            logger.info(f"[PATTERN_LLM] LLM client initialized: {type(client).__name__}")
            return client
        except ImportError as e:
            logger.error(f"[PATTERN_LLM] Failed to import LLM client: {e}")
            return None
        except Exception as e:
            logger.error(f"[PATTERN_LLM] Failed to initialize LLM client: {e}")
            return None
    
    async def analyze_patterns_with_llm(
        self, 
        symbol: str, 
        stock_data: pd.DataFrame,
        indicators: Dict[str, Any],
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Main entry point for comprehensive pattern analysis with LLM synthesis.
        
        Args:
            symbol: Stock symbol being analyzed
            stock_data: OHLCV price data DataFrame
            indicators: Technical indicators dictionary
            context: Additional context for analysis
            
        Returns:
            Dictionary containing technical pattern analysis and LLM synthesis
        """
        start_time = time.time()
        operation_id = f"pattern_llm_{symbol}_{int(time.time())}"
        
        logger.info(f"[PATTERN_LLM] Starting comprehensive analysis for {symbol} (ID: {operation_id})")
        
        try:
            # Step 1: Execute comprehensive technical pattern analysis (no LLM calls)
            logger.info(f"[PATTERN_LLM] Step 1: Technical pattern detection for {symbol}")
            pattern_analysis = await self._execute_pattern_analysis(
                symbol, stock_data, indicators, context
            )
            
            if not pattern_analysis or not pattern_analysis.get('success', False):
                logger.warning(f"[PATTERN_LLM] Pattern analysis failed for {symbol}")
                return self._build_failure_result(symbol, "Pattern analysis failed", start_time)
            
            # Step 2: Build LLM context from pattern results
            logger.info(f"[PATTERN_LLM] Step 2: Building LLM context for {symbol}")
            current_price = stock_data['close'].iloc[-1] if not stock_data.empty else None
            llm_context = self.context_builder.build_comprehensive_pattern_context(
                pattern_analysis, symbol, current_price
            )
            
            # Step 3: LLM synthesis and analysis
            logger.info(f"[PATTERN_LLM] Step 3: LLM synthesis for {symbol}")
            llm_result = await self._synthesize_with_llm(llm_context, symbol)
            
            # Step 4: Combine technical and LLM results
            processing_time = time.time() - start_time
            final_result = self._build_final_result(
                symbol, pattern_analysis, llm_result, processing_time, operation_id
            )
            
            logger.info(f"[PATTERN_LLM] Analysis completed for {symbol} in {processing_time:.2f}s (ID: {operation_id})")
            return final_result
            
        except asyncio.TimeoutError:
            processing_time = time.time() - start_time
            logger.error(f"[PATTERN_LLM] Timeout analyzing {symbol} after {processing_time:.2f}s")
            return self._build_failure_result(symbol, "Analysis timeout", processing_time)
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"[PATTERN_LLM] Analysis failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return self._build_failure_result(symbol, f"Analysis error: {str(e)}", processing_time)
    
    async def _execute_pattern_analysis(
        self, 
        symbol: str, 
        stock_data: pd.DataFrame, 
        indicators: Dict[str, Any], 
        context: str
    ) -> Dict[str, Any]:
        """
        Execute comprehensive pattern analysis using PatternAgentsOrchestrator.
        
        This runs all 4 pattern agents (reversal, continuation, recognition, technical)
        and returns aggregated technical analysis results (no LLM calls).
        """
        try:
            # Execute pattern analysis using the orchestrator
            # This runs all 4 pattern agents in parallel with technical analysis only
            aggregated_result = await self.pattern_orchestrator.analyze_patterns_comprehensive(
                symbol=symbol,
                stock_data=stock_data,
                indicators=indicators,
                context=context,
                chart_images=None  # Text-only analysis, no charts
            )
            
            logger.info(f"[PATTERN_LLM] Pattern orchestrator completed for {symbol}")
            return {
                'success': True,
                'analysis_data': aggregated_result,
                'symbol': symbol
            }
            
        except Exception as e:
            logger.error(f"[PATTERN_LLM] Pattern orchestrator failed for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e),
                'symbol': symbol
            }
    
    async def _synthesize_with_llm(self, pattern_context: str, symbol: str) -> Dict[str, Any]:
        """
        Send pattern analysis context to LLM for synthesis and trading insights.
        """
        if not self.llm_client:
            logger.error(f"[PATTERN_LLM] No LLM client available for {symbol}")
            return {
                'success': False,
                'error': 'No LLM client available',
                'analysis': 'LLM synthesis unavailable - technical analysis only'
            }
        
        try:
            # Build comprehensive prompt
            prompt = self._build_llm_prompt(pattern_context, symbol)
            
            logger.info(f"[PATTERN_LLM] Sending LLM request for {symbol} (context: {len(pattern_context)} chars)")
            
            # Call LLM (text-only, no images)
            llm_response = await self.llm_client.generate(
                prompt=prompt,
                images=None,  # Text-only analysis
                enable_code_execution=False  # No code execution needed for pattern synthesis
            )
            
            if not llm_response or not llm_response.strip():
                logger.warning(f"[PATTERN_LLM] Empty LLM response for {symbol}")
                return {
                    'success': False,
                    'error': 'Empty LLM response',
                    'analysis': 'LLM returned empty response'
                }
            
            logger.info(f"[PATTERN_LLM] LLM synthesis completed for {symbol} ({len(llm_response)} chars)")
            
            # Parse LLM response
            parsed_response = self._parse_llm_response(llm_response, symbol)
            
            return {
                'success': True,
                'raw_response': llm_response,
                'parsed_analysis': parsed_response,
                'response_length': len(llm_response)
            }
            
        except asyncio.TimeoutError:
            logger.error(f"[PATTERN_LLM] LLM request timeout for {symbol}")
            return {
                'success': False,
                'error': 'LLM request timeout',
                'analysis': 'LLM synthesis timed out'
            }
        except Exception as e:
            logger.error(f"[PATTERN_LLM] LLM synthesis failed for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e),
                'analysis': f'LLM synthesis failed: {str(e)}'
            }
    
    def _build_llm_prompt(self, pattern_context: str, symbol: str) -> str:
        """
        Build comprehensive LLM prompt for pattern analysis.
        
        This creates a detailed prompt that asks the LLM to analyze the pattern
        data and provide trading insights and recommendations.
        """
        
        prompt = f"""You are an expert technical analyst specializing in chart pattern recognition and trading strategy development.

## ANALYSIS TASK
Analyze the comprehensive pattern data for {symbol} and provide actionable trading insights based on the detected patterns.

## PATTERN ANALYSIS DATA
{pattern_context}

## ANALYSIS REQUIREMENTS

### 1. PATTERN SYNTHESIS & INTERPRETATION
- Summarize the most significant patterns detected across all agents
- Evaluate pattern reliability, completion status, and strength
- Identify key pattern confluence areas and resolve any conflicts
- Assess the overall market bias based on pattern analysis

### 2. TRADING STRATEGY & RECOMMENDATIONS
- Provide specific entry/exit levels based on the strongest patterns
- Calculate and justify risk-reward ratios for pattern-based trades
- Suggest appropriate position sizing based on pattern confidence
- Recommend timeframe for trade execution (short/medium/long term)

### 3. RISK ASSESSMENT & MANAGEMENT
- Evaluate pattern failure risks and invalidation scenarios
- Identify key stop-loss levels and risk management points  
- Assess false breakout/breakdown probabilities
- Provide overall risk rating for pattern-based trading

### 4. MARKET CONTEXT & OUTLOOK
- Analyze how different pattern types align or conflict
- Provide short-term vs long-term pattern-based outlook
- Identify critical price levels to watch for pattern evolution
- Suggest market conditions that would strengthen/weaken patterns

## OUTPUT FORMAT
Provide your analysis in the following structured format:

### PATTERN ANALYSIS SUMMARY
[Comprehensive summary of key patterns and their implications]

### TRADING RECOMMENDATIONS
- **Primary Signal**: [Bullish/Bearish/Neutral with confidence level]
- **Entry Strategy**: [Specific entry levels and conditions]
- **Exit Strategy**: [Stop-loss and target levels with rationale]
- **Position Size**: [Recommended position sizing based on pattern strength]
- **Time Horizon**: [Expected timeframe for pattern completion]

### RISK ANALYSIS
- **Pattern Failure Risk**: [Assessment of pattern breakdown probability]
- **Key Invalidation Levels**: [Price levels that would negate patterns]
- **Overall Risk Rating**: [High/Medium/Low with justification]

### KEY OBSERVATIONS
- [3-5 bullet points highlighting the most important pattern insights]
- [Focus on actionable intelligence for trading decisions]

Please ensure your analysis is practical and actionable for trading decisions while maintaining objectivity about pattern limitations and risks."""

        return prompt
    
    def _parse_llm_response(self, llm_response: str, symbol: str) -> Dict[str, Any]:
        """
        Parse LLM response and extract structured information.
        
        This is a simple parser - could be enhanced with more sophisticated
        extraction logic if needed.
        """
        try:
            # For now, we'll store the full response and do basic extraction
            parsed = {
                'full_analysis': llm_response,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
            
            # Try to extract key sections (basic implementation)
            if 'PATTERN ANALYSIS SUMMARY' in llm_response:
                sections = llm_response.split('###')
                for section in sections:
                    if 'PATTERN ANALYSIS SUMMARY' in section:
                        parsed['pattern_summary'] = section.replace('PATTERN ANALYSIS SUMMARY', '').strip()
                    elif 'TRADING RECOMMENDATIONS' in section:
                        parsed['trading_recommendations'] = section.replace('TRADING RECOMMENDATIONS', '').strip()
                    elif 'RISK ANALYSIS' in section:
                        parsed['risk_analysis'] = section.replace('RISK ANALYSIS', '').strip()
                    elif 'KEY OBSERVATIONS' in section:
                        parsed['key_observations'] = section.replace('KEY OBSERVATIONS', '').strip()
            
            return parsed
            
        except Exception as e:
            logger.warning(f"[PATTERN_LLM] Error parsing LLM response for {symbol}: {e}")
            return {
                'full_analysis': llm_response,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'parse_error': str(e)
            }
    
    def _build_final_result(
        self, 
        symbol: str, 
        pattern_analysis: Dict[str, Any], 
        llm_result: Dict[str, Any], 
        processing_time: float,
        operation_id: str
    ) -> Dict[str, Any]:
        """Build final comprehensive result combining technical and LLM analysis."""
        
        # Extract pattern analysis data
        technical_analysis = pattern_analysis.get('analysis_data', {})
        
        # Calculate overall confidence
        overall_confidence = 0.5  # Default
        if technical_analysis and 'overall_confidence' in technical_analysis:
            tech_confidence = technical_analysis['overall_confidence']
            llm_confidence = 0.7 if llm_result.get('success', False) else 0.3
            overall_confidence = (tech_confidence * 0.7) + (llm_confidence * 0.3)
        
        # Build comprehensive result
        result = {
            'agent_name': self.name,
            'operation_id': operation_id,
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'processing_time': processing_time,
            'success': True,
            'confidence_score': overall_confidence,
            
            # Technical pattern analysis (from PatternAgentsOrchestrator)
            'technical_analysis': {
                'pattern_agents_result': technical_analysis,
                'individual_agent_results': technical_analysis.get('individual_results', {}),
                'unified_analysis': technical_analysis.get('unified_analysis', {}),
                'consensus_signals': technical_analysis.get('consensus_signals', {}),
                'conflicting_signals': technical_analysis.get('conflicting_signals', [])
            },
            
            # LLM synthesis and insights
            'llm_synthesis': {
                'success': llm_result.get('success', False),
                'analysis': llm_result.get('parsed_analysis', {}),
                'raw_response': llm_result.get('raw_response', ''),
                'error': llm_result.get('error')
            },
            
            # Summary for other agents/decision makers
            'pattern_summary': self._build_summary_for_final_decision(technical_analysis, llm_result),
            
            # Performance metadata
            'metadata': {
                'successful_pattern_agents': technical_analysis.get('successful_agents', 0),
                'total_pattern_agents': (
                    technical_analysis.get('successful_agents', 0) + 
                    technical_analysis.get('failed_agents', 0)
                ),
                'llm_synthesis_success': llm_result.get('success', False),
                'context_length': len(llm_result.get('raw_response', '')),
                'version': self.version
            }
        }
        
        return result
    
    def _build_summary_for_final_decision(
        self, 
        technical_analysis: Dict[str, Any], 
        llm_result: Dict[str, Any]
    ) -> str:
        """
        Build concise summary for final decision agent consumption.
        
        This creates a human-readable summary similar to what volume/risk agents provide.
        """
        try:
            summary_parts = []
            
            # Technical summary
            if technical_analysis:
                total_patterns = technical_analysis.get('unified_analysis', {}).get('pattern_summary', {}).get('total_patterns_identified', 0)
                consensus = technical_analysis.get('consensus_signals', {})
                confidence = technical_analysis.get('overall_confidence', 0)
                
                summary_parts.append(f"Pattern Detection: {total_patterns} patterns identified (confidence: {confidence:.2f})")
                
                if consensus:
                    primary_signals = list(consensus.keys())[:2]  # Top 2 consensus signals
                    summary_parts.append(f"Primary Signals: {', '.join(primary_signals)}")
            
            # LLM insights summary
            if llm_result.get('success') and llm_result.get('parsed_analysis'):
                parsed = llm_result['parsed_analysis']
                if 'pattern_summary' in parsed:
                    # Extract first sentence of pattern summary
                    pattern_summary = parsed['pattern_summary']
                    first_sentence = pattern_summary.split('.')[0] if pattern_summary else ''
                    if first_sentence:
                        summary_parts.append(f"Key Insight: {first_sentence}")
            
            return ". ".join(summary_parts) if summary_parts else "Pattern analysis completed with mixed results."
            
        except Exception as e:
            logger.warning(f"[PATTERN_LLM] Error building summary: {e}")
            return "Pattern analysis completed - see full results for details."
    
    def _build_failure_result(self, symbol: str, error: str, processing_time: float) -> Dict[str, Any]:
        """Build failure result when analysis fails."""
        
        return {
            'agent_name': self.name,
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'processing_time': processing_time,
            'success': False,
            'error': error,
            'confidence_score': 0.0,
            'pattern_summary': f"Pattern analysis failed for {symbol}: {error}",
            'technical_analysis': {},
            'llm_synthesis': {
                'success': False,
                'error': error,
                'analysis': {}
            },
            'metadata': {
                'version': self.version,
                'failure_reason': error
            }
        }


# Test function for the Pattern LLM Agent
async def test_pattern_llm_agent():
    """Test the Pattern LLM Agent with sample data."""
    
    print("🧪 Testing Pattern LLM Agent")
    print("=" * 50)
    
    try:
        # Create pattern LLM agent (without real LLM client for testing)
        agent = PatternLLMAgent(gemini_client=None)
        print("✅ PatternLLMAgent created successfully")
        print(f"   Agent name: {agent.name}")
        print(f"   Version: {agent.version}")
        print(f"   Has LLM client: {agent.llm_client is not None}")
        print(f"   Has pattern orchestrator: {agent.pattern_orchestrator is not None}")
        print(f"   Has context builder: {agent.context_builder is not None}")
        
        # Test would require actual stock data and LLM client for full integration test
        # For now, we'll test the component initialization
        
        print("✅ Pattern LLM Agent component test completed successfully")
        print("   Ready for integration with analysis service")
        return True
        
    except Exception as e:
        print(f"❌ Pattern LLM Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_pattern_llm_agent())