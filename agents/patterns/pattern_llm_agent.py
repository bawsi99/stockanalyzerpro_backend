"""
Pattern LLM Agent

Main orchestrator for pattern analysis with LLM synthesis.
Coordinates pattern detection through PatternAgentsOrchestrator and provides
LLM-based analysis and trading insights.
"""

import logging
import asyncio
import time
from datetime import datetime
from typing import Dict, Any
import pandas as pd

from .patterns_agents import PatternAgentsOrchestrator
from .pattern_context_builder import PatternContextBuilder

logger = logging.getLogger(__name__)


class PatternLLMAgent:
    """
    Main Pattern LLM Agent that orchestrates comprehensive pattern analysis
    and provides LLM-based synthesis and trading insights.
    """
    
    def __init__(self, gemini_client=None):
        self.name = "pattern_llm"
        self.version = "1.0.0"
        self.pattern_orchestrator = PatternAgentsOrchestrator(gemini_client)
        self.context_builder = PatternContextBuilder()
        
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
        except Exception as e:
            logger.error(f"[PATTERN_LLM] Failed to initialize LLM client: {e}")
            return None
    
    async def analyze_patterns_with_llm(self, symbol: str, stock_data: pd.DataFrame, indicators: Dict[str, Any], context: str = "") -> Dict[str, Any]:
        """Main entry point for comprehensive pattern analysis with LLM synthesis."""
        start_time = time.time()
        operation_id = f"pattern_llm_{symbol}_{int(time.time())}"
        
        logger.info(f"[PATTERN_LLM] Starting comprehensive analysis for {symbol} (ID: {operation_id})")
        
        try:
            # Step 1: Technical pattern detection
            pattern_analysis = await self._execute_pattern_analysis(symbol, stock_data, indicators, context)
            
            if not pattern_analysis or not pattern_analysis.get('success', False):
                return self._build_failure_result(symbol, "Pattern analysis failed", time.time() - start_time)
            
            # Step 2: Build LLM context
            current_price = stock_data['close'].iloc[-1] if not stock_data.empty else None

            # Convert AggregatedPatternAnalysis to dict and strip non-text fields (e.g., images)
            analysis_data = pattern_analysis.get('analysis_data', {})
            try:
                from dataclasses import is_dataclass, asdict
                if is_dataclass(analysis_data):
                    analysis_data = asdict(analysis_data)
            except Exception as conv_err:
                logger.warning(f"[PATTERN_LLM] asdict conversion failed, using __dict__ if available: {conv_err}")
                if hasattr(analysis_data, '__dict__'):
                    analysis_data = analysis_data.__dict__

            # Remove embedded chart images from individual results to keep context lean
            try:
                indiv = analysis_data.get('individual_results', {})
                if isinstance(indiv, dict):
                    for k, v in list(indiv.items()):
                        if isinstance(v, dict):
                            v.pop('chart_image', None)
                        elif hasattr(v, '__dict__'):
                            # Convert nested dataclass and strip image
                            dv = v.__dict__
                            dv.pop('chart_image', None)
                            indiv[k] = dv
            except Exception as strip_err:
                logger.warning(f"[PATTERN_LLM] Failed to strip chart images: {strip_err}")

            llm_context = self.context_builder.build_comprehensive_pattern_context(
                analysis_data, symbol, current_price
            )
            
            # Step 3: LLM synthesis
            llm_result = await self._synthesize_with_llm(llm_context, symbol)
            
            # Step 4: Combine results
            processing_time = time.time() - start_time
            return self._build_final_result(symbol, pattern_analysis, llm_result, processing_time, operation_id)
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"[PATTERN_LLM] Analysis failed for {symbol}: {e}")
            return self._build_failure_result(symbol, f"Analysis error: {str(e)}", processing_time)
    
    async def _execute_pattern_analysis(self, symbol: str, stock_data: pd.DataFrame, indicators: Dict[str, Any], context: str) -> Dict[str, Any]:
        """Execute comprehensive pattern analysis using PatternAgentsOrchestrator."""
        try:
            aggregated_result = await self.pattern_orchestrator.analyze_patterns_comprehensive(
                symbol=symbol, stock_data=stock_data, indicators=indicators, context=context, chart_images=None
            )
            
            return {'success': True, 'analysis_data': aggregated_result, 'symbol': symbol}
            
        except Exception as e:
            logger.error(f"[PATTERN_LLM] Pattern orchestrator failed for {symbol}: {e}")
            return {'success': False, 'error': str(e), 'symbol': symbol}
    
    async def _synthesize_with_llm(self, pattern_context: str, symbol: str) -> Dict[str, Any]:
        """Send pattern analysis context to LLM for synthesis."""
        if not self.llm_client:
            return {'success': False, 'error': 'No LLM client available'}
        
        try:
            prompt = self._build_llm_prompt(pattern_context, symbol)
            
            llm_response = await self.llm_client.generate(
                prompt=prompt, images=None, enable_code_execution=False
            )
            
            if not llm_response:
                return {'success': False, 'error': 'Empty LLM response'}
            
            return {'success': True, 'raw_response': llm_response, 'parsed_analysis': {'full_analysis': llm_response}}
            
        except Exception as e:
            logger.error(f"[PATTERN_LLM] LLM synthesis failed for {symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _build_llm_prompt(self, pattern_context: str, symbol: str) -> str:
        """Build comprehensive LLM prompt for pattern analysis."""
        return f"""You are an expert technical analyst. Analyze the pattern data for {symbol} and provide actionable insights.

PATTERN ANALYSIS DATA:
{pattern_context}

Provide comprehensive analysis including:
1. Key patterns identified and their significance
2. Trading recommendations with entry/exit levels  
3. Risk assessment and management
4. Market outlook based on patterns

Format your response clearly with specific actionable recommendations."""

    def _build_final_result(self, symbol: str, pattern_analysis: Dict[str, Any], llm_result: Dict[str, Any], processing_time: float, operation_id: str) -> Dict[str, Any]:
        """Build final comprehensive result."""
        technical_analysis = pattern_analysis.get('analysis_data', {})
        overall_confidence = technical_analysis.get('overall_confidence', 0.5)
        
        return {
            'agent_name': self.name,
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'processing_time': processing_time,
            'success': True,
            'confidence_score': overall_confidence,
            'technical_analysis': technical_analysis,
            'llm_synthesis': llm_result,
            'pattern_summary': f"Pattern analysis completed for {symbol}",
            'metadata': {'version': self.version}
        }
    
    def _build_failure_result(self, symbol: str, error: str, processing_time: float) -> Dict[str, Any]:
        """Build failure result."""
        return {
            'agent_name': self.name,
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'processing_time': processing_time,
            'success': False,
            'error': error,
            'confidence_score': 0.0,
            'pattern_summary': f"Pattern analysis failed for {symbol}: {error}",
            'metadata': {'version': self.version}
        }


# Test function
async def test_pattern_llm_agent():
    print("🧪 Testing Pattern LLM Agent")
    print("=" * 50)
    
    try:
        agent = PatternLLMAgent(gemini_client=None)
        print("✅ PatternLLMAgent created successfully")
        print(f"   Agent name: {agent.name}")
        print(f"   Version: {agent.version}")
        print("✅ Ready for integration")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_pattern_llm_agent())
