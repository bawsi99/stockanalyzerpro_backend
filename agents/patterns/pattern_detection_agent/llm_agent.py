#!/usr/bin/env python3
"""
Pattern Detection LLM Agent

This module handles LLM integration for pattern detection analysis, providing:
- Intelligent pattern interpretation and insights
- Pattern confluence analysis using AI
- Trading recommendations based on detected patterns
- Risk assessment for pattern-based trading decisions
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import sys
import os

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from llm.gemini_llm import GeminiLLM

logger = logging.getLogger(__name__)

class PatternDetectionLLMAgent:
    """
    LLM Agent specialized in pattern detection analysis.
    
    Provides AI-powered insights for:
    - Pattern detection interpretation
    - Pattern confluence analysis
    - Trading strategy recommendations
    - Risk assessment and market outlook
    """
    
    def __init__(self):
        self.name = "pattern_detection_llm"
        self.version = "1.0.0"
        self.llm = GeminiLLM()
    
    async def generate_pattern_analysis(
        self, 
        stock_data: Dict[str, Any], 
        pattern_data: Dict[str, Any],
        symbol: str = "STOCK",
        market_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive LLM analysis for pattern detection results.
        
        Args:
            stock_data: Basic stock information and current state
            pattern_data: Pattern detection analysis results
            symbol: Stock symbol
            market_context: Optional market context information
            
        Returns:
            Dictionary containing LLM analysis and insights
        """
        try:
            logger.info(f"[PATTERN_DETECTION_LLM] Generating analysis for {symbol}")
            
            if not pattern_data.get('success', False):
                return self._build_error_result("Pattern detection analysis failed - no data to analyze")
            
            # Build comprehensive prompt
            analysis_prompt = self._build_pattern_analysis_prompt(
                stock_data, pattern_data, symbol, market_context
            )
            
            # Get LLM response
            llm_response = await self._get_llm_response(analysis_prompt, symbol)
            
            # Parse and structure the response
            structured_response = self._parse_llm_response(llm_response, pattern_data, symbol)
            
            return structured_response
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION_LLM] Analysis generation failed: {e}")
            return self._build_error_result(str(e))
    
    def _build_pattern_analysis_prompt(
        self, 
        stock_data: Dict[str, Any], 
        pattern_data: Dict[str, Any],
        symbol: str,
        market_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build comprehensive prompt for pattern detection analysis"""
        
        # Extract key information
        detected_patterns = pattern_data.get('detected_patterns', [])
        pattern_summary = pattern_data.get('pattern_summary', {})
        formation_stage = pattern_data.get('formation_stage', {})
        key_levels = pattern_data.get('key_levels', {})
        confidence_score = pattern_data.get('confidence_score', 0)
        
        current_price = key_levels.get('current_price', 0)
        total_patterns = pattern_summary.get('total_patterns', 0)
        dominant_pattern = pattern_summary.get('dominant_pattern', 'none')
        overall_bias = pattern_summary.get('overall_bias', 'neutral')
        
        prompt = f"""# Pattern Detection Analysis Report for {symbol}

## ANALYSIS CONTEXT
You are an expert technical analyst specializing in chart pattern recognition and interpretation. Analyze the detected patterns and provide comprehensive insights for trading and investment decisions.

## STOCK INFORMATION
- **Symbol**: {symbol}
- **Current Price**: ${current_price:.2f}
- **Analysis Confidence**: {confidence_score:.1%}

## DETECTED PATTERNS SUMMARY
- **Total Patterns Detected**: {total_patterns}
- **Dominant Pattern**: {dominant_pattern}
- **Overall Market Bias**: {overall_bias}
- **Pattern Confluence**: {pattern_summary.get('pattern_confluence', 'none')}
- **Formation Stage**: {formation_stage.get('pattern_maturity', 'unknown')}

## DETAILED PATTERN ANALYSIS

### Individual Patterns Detected:
"""
        
        if detected_patterns:
            for i, pattern in enumerate(detected_patterns, 1):
                pattern_name = pattern.get('pattern_name', f'Pattern {i}')
                pattern_type = pattern.get('pattern_type', 'unknown')
                completion = pattern.get('completion_percentage', 0)
                reliability = pattern.get('reliability', 'unknown')
                quality = pattern.get('pattern_quality', 'unknown')
                
                prompt += f"""
**{i}. {pattern_name.replace('_', ' ').title()}**
- Pattern Type: {pattern_type.title()}
- Completion: {completion}%
- Reliability: {reliability.title()}
- Quality: {quality.title()}
"""
                
                # Add pattern-specific data
                pattern_data_info = pattern.get('pattern_data', {})
                if pattern_data_info:
                    prompt += f"- Technical Details: {json.dumps(pattern_data_info, indent=2)}\n"
        else:
            prompt += "\nNo significant patterns detected in the current timeframe.\n"
        
        # Add key levels information
        prompt += f"""
## KEY PRICE LEVELS
- **Current Price**: ${current_price:.2f}
- **Nearest Resistance**: ${key_levels.get('nearest_resistance', 0):.2f}
- **Nearest Support**: ${key_levels.get('nearest_support', 0):.2f}
- **Key Breakout Level**: ${key_levels.get('breakout_level', current_price):.2f}

## FORMATION ANALYSIS
- **Primary Stage**: {formation_stage.get('primary_stage', 'unknown')}
- **Pattern Maturity**: {formation_stage.get('pattern_maturity', 'unknown')}
- **Breakout Potential**: {formation_stage.get('breakout_potential', 'unknown')}

## MARKET CONTEXT
"""
        
        if market_context:
            prompt += f"- Market Environment: {market_context.get('market_trend', 'unknown')}\n"
            prompt += f"- Sector Performance: {market_context.get('sector_performance', 'unknown')}\n"
            prompt += f"- Volume Profile: {market_context.get('volume_profile', 'unknown')}\n"
        else:
            prompt += "- No additional market context provided\n"
        
        prompt += f"""
## ANALYSIS REQUIREMENTS

Please provide a comprehensive analysis covering:

### 1. PATTERN INTERPRETATION
- Detailed analysis of each detected pattern
- Pattern significance and reliability assessment
- Historical success rates for detected pattern types
- Pattern confluence and reinforcement analysis

### 2. MARKET OUTLOOK
- Short-term price direction based on patterns
- Medium-term trend implications
- Key scenarios and probability assessments
- Critical levels to monitor

### 3. TRADING STRATEGY
- Entry points and timing considerations
- Stop-loss levels and risk management
- Profit targets and exit strategies  
- Position sizing recommendations

### 4. RISK ASSESSMENT
- Pattern failure scenarios and probabilities
- Key risk factors and warning signals
- Volatility expectations
- Maximum drawdown potential

### 5. KEY INSIGHTS
- Most significant findings from pattern analysis
- Confluence factors that strengthen/weaken signals
- Unique characteristics of current pattern setup
- Historical precedents and comparisons

## OUTPUT FORMAT
Please structure your response as a comprehensive analysis report with clear sections and actionable insights. Focus on practical trading applications while maintaining professional technical analysis standards.

**Important**: Base your analysis strictly on the provided pattern detection data. Avoid speculation beyond what the technical patterns suggest.
"""
        
        return prompt
    
    async def _get_llm_response(self, prompt: str, symbol: str) -> str:
        """Get response from LLM with error handling"""
        try:
            # Make async call to LLM
            response = await self.llm.generate_response_async(prompt)
            
            if not response or len(response.strip()) < 50:
                return f"Pattern detection analysis for {symbol} could not be completed due to insufficient LLM response."
            
            return response
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION_LLM] LLM request failed: {e}")
            return f"Pattern detection analysis for {symbol} encountered an error: {str(e)}"
    
    def _parse_llm_response(self, llm_response: str, pattern_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Parse and structure the LLM response"""
        try:
            # Basic response validation
            if "error:" in llm_response.lower() or len(llm_response.strip()) < 100:
                logger.warning(f"[PATTERN_DETECTION_LLM] Poor quality response for {symbol}")
                return self._build_fallback_analysis(pattern_data, symbol, llm_response)
            
            # Extract key sections from response (basic parsing)
            sections = self._extract_response_sections(llm_response)
            
            # Build structured result
            result = {
                'success': True,
                'agent_name': self.name,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'pattern_detection',
                'confidence_score': pattern_data.get('confidence_score', 0),
                
                # Core Analysis Content
                'pattern_interpretation': sections.get('pattern_interpretation', ''),
                'market_outlook': sections.get('market_outlook', ''),
                'trading_strategy': sections.get('trading_strategy', ''),
                'risk_assessment': sections.get('risk_assessment', ''),
                'key_insights': sections.get('key_insights', ''),
                
                # Full Response
                'full_analysis': llm_response,
                'total_patterns_analyzed': len(pattern_data.get('detected_patterns', [])),
                
                # Quality Metrics
                'response_length': len(llm_response),
                'analysis_quality': self._assess_response_quality(llm_response)
            }
            
            logger.info(f"[PATTERN_DETECTION_LLM] Analysis completed for {symbol} ({len(llm_response)} chars)")
            return result
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION_LLM] Response parsing failed: {e}")
            return self._build_fallback_analysis(pattern_data, symbol, llm_response)
    
    def _extract_response_sections(self, response: str) -> Dict[str, str]:
        """Extract different sections from the LLM response"""
        sections = {}
        
        try:
            # Define section headers to look for
            section_headers = {
                'pattern_interpretation': ['pattern interpretation', 'pattern analysis', '1. pattern'],
                'market_outlook': ['market outlook', 'price direction', '2. market'],
                'trading_strategy': ['trading strategy', 'entry points', '3. trading'],
                'risk_assessment': ['risk assessment', 'risk factors', '4. risk'],
                'key_insights': ['key insights', 'key findings', '5. key']
            }
            
            response_lower = response.lower()
            
            # Extract sections based on headers
            for section_name, headers in section_headers.items():
                section_content = ""
                
                for header in headers:
                    header_pos = response_lower.find(header)
                    if header_pos != -1:
                        # Find the start of content after header
                        content_start = header_pos + len(header)
                        
                        # Find next section or end
                        next_section_pos = len(response)
                        for other_section, other_headers in section_headers.items():
                            if other_section != section_name:
                                for other_header in other_headers:
                                    other_pos = response_lower.find(other_header, content_start + 50)
                                    if other_pos != -1 and other_pos < next_section_pos:
                                        next_section_pos = other_pos
                        
                        # Extract content
                        section_content = response[content_start:next_section_pos].strip()
                        break
                
                sections[section_name] = section_content[:2000]  # Limit length
            
            return sections
            
        except Exception as e:
            logger.error(f"[PATTERN_DETECTION_LLM] Section extraction failed: {e}")
            return {'full_content': response[:2000]}
    
    def _assess_response_quality(self, response: str) -> str:
        """Assess the quality of the LLM response"""
        try:
            if len(response) < 200:
                return 'poor'
            elif len(response) < 800:
                return 'fair'
            elif len(response) < 2000:
                return 'good'
            else:
                return 'excellent'
                
        except Exception:
            return 'unknown'
    
    def _build_fallback_analysis(self, pattern_data: Dict[str, Any], symbol: str, llm_response: str) -> Dict[str, Any]:
        """Build fallback analysis when LLM response is poor"""
        detected_patterns = pattern_data.get('detected_patterns', [])
        pattern_summary = pattern_data.get('pattern_summary', {})
        
        fallback_content = f"""
## Pattern Detection Analysis for {symbol}

### Summary
- {len(detected_patterns)} patterns detected
- Overall bias: {pattern_summary.get('overall_bias', 'neutral')}
- Confidence level: {pattern_data.get('confidence_score', 0):.1%}

### Detected Patterns
"""
        
        for i, pattern in enumerate(detected_patterns, 1):
            fallback_content += f"""
{i}. {pattern.get('pattern_name', 'Unknown').replace('_', ' ').title()}
   - Type: {pattern.get('pattern_type', 'unknown')}
   - Completion: {pattern.get('completion_percentage', 0)}%
   - Reliability: {pattern.get('reliability', 'unknown')}
"""
        
        fallback_content += f"""
### Key Levels
- Current Price: ${pattern_data.get('key_levels', {}).get('current_price', 0):.2f}
- Support: ${pattern_data.get('key_levels', {}).get('nearest_support', 0):.2f}
- Resistance: ${pattern_data.get('key_levels', {}).get('nearest_resistance', 0):.2f}

*Note: This is a simplified analysis due to limited AI response quality.*
"""
        
        return {
            'success': True,
            'agent_name': self.name,
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'pattern_detection_fallback',
            'confidence_score': pattern_data.get('confidence_score', 0),
            'pattern_interpretation': fallback_content,
            'market_outlook': 'Analysis quality limited - manual review recommended',
            'trading_strategy': 'Consult additional analysis before trading decisions',
            'risk_assessment': 'Standard pattern-based risks apply',
            'key_insights': f'{len(detected_patterns)} patterns detected with varying reliability',
            'full_analysis': fallback_content,
            'total_patterns_analyzed': len(detected_patterns),
            'response_length': len(fallback_content),
            'analysis_quality': 'fallback',
            'original_llm_response': llm_response
        }
    
    def _build_error_result(self, error_message: str) -> Dict[str, Any]:
        """Build error result dictionary"""
        return {
            'success': False,
            'agent_name': self.name,
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'pattern_detection',
            'confidence_score': 0.0
        }