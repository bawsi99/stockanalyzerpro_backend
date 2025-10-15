#!/usr/bin/env python3
"""
Integrated Market Structure Agent with Chart Generation and LLM Analysis

This module provides a comprehensive market structure analysis system that combines:
1. Optimized chart generation with multiple quality levels
2. Enhanced multimodal LLM analysis using generated charts
3. Structured response parsing and validation
4. Error handling and resilience features

Features:
- Automatic chart generation for market structure data
- Multimodal LLM analysis with visual and numerical data
- Quality-based fallback for chart generation failures
- Structured JSON response parsing with validation
- Performance optimization and caching support
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from PIL import Image
import traceback

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.append(parent_dir)

# Import resilient chart generator
try:
    from agents.patterns.market_structure_agent.resilient_chart_generator import ResilientChartGenerator
except ImportError:
    # Fallback import
    from resilient_chart_generator import ResilientChartGenerator

# Import LLM client
try:
    from backend.llm import get_llm_client
except ImportError:
    try:
        from llm import get_llm_client
    except ImportError:
        def get_llm_client(agent_name):
            return None

# Set up logging
logger = logging.getLogger(__name__)

class IntegratedMarketStructureAgent:
    """
    Main integrated agent that combines chart generation with LLM analysis.
    
    This agent coordinates the complete workflow:
    1. Generate optimized market structure charts
    2. Create enhanced prompts with visual context
    3. Execute multimodal LLM analysis
    4. Parse and validate structured responses
    5. Provide comprehensive market structure insights
    """
    
    def __init__(self, 
                 charts_output_dir: str = "integrated_charts",
                 results_output_dir: str = "integrated_analysis_results",
                 agent_name: str = "pattern_agent"):
        """
        Initialize the integrated market structure agent.
        
        Args:
            charts_output_dir: Directory for generated charts
            results_output_dir: Directory for analysis results
            agent_name: LLM agent configuration name
        """
        self.name = "integrated_market_structure_agent"
        self.version = "1.0.0"
        
        # Setup directories
        self.charts_dir = Path(charts_output_dir)
        self.results_dir = Path(results_output_dir)
        self.charts_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.chart_generator = ResilientChartGenerator(output_dir=str(self.charts_dir))
        
        # Initialize LLM client
        try:
            self.llm_client = get_llm_client(agent_name)
            logger.info(f"✅ {self.name} initialized with LLM agent: {agent_name}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM client: {e}")
            self.llm_client = None
        
        # Enhanced prompt template
        self.prompt_template = self._load_prompt_template()
        
        logger.info(f"📊 {self.name} v{self.version} initialized")
        
    def _load_prompt_template(self) -> str:
        """Load the enhanced prompt template for market structure analysis."""
        return """# Market Structure Analysis - Multimodal Enhanced Analysis

## CONTEXT
You are a Market Structure Analysis Expert with advanced multimodal capabilities. You have access to both a comprehensive market structure chart and detailed numerical analysis data.

## VISUAL CHART PROVIDED
The attached chart contains:
• **Price Action**: Candlestick-style visualization with market structure context
• **Swing Points**: Algorithmically identified with strength-based visual sizing
• **BOS/CHOCH Events**: Break of Structure and Change of Character marked with directional annotations
• **Support/Resistance Levels**: Key levels with strength indicators and touch counts
• **Volume Analysis**: Volume bars with trend correlation indicators
• **Trend Structure**: Multi-timeframe trend analysis and regime classification
• **Market Context**: Current market state and structural bias indicators

Chart Details:
- Symbol: {symbol}
- Scenario: {scenario_description}
- File Size: {chart_size_kb} KB
- Generated: {generation_timestamp}

## NUMERICAL ANALYSIS DATA

{analysis_context}

## YOUR ENHANCED ANALYSIS TASK

Using BOTH the visual chart and numerical data, provide comprehensive market structure analysis covering:

### 1. VISUAL CHART VALIDATION
- Validate algorithmic findings against visual patterns
- Identify any discrepancies between numerical and visual analysis
- Assess chart clarity and pattern definition quality
- Evaluate the strength of visual confirmations

### 2. SWING POINT ANALYSIS
- Confirm swing point identification accuracy
- Assess swing point quality and significance
- Evaluate swing density and structure integrity
- Identify key recent swing formations

### 3. STRUCTURAL BREAK ASSESSMENT
- Validate BOS/CHOCH event identification
- Evaluate break quality and strength
- Assess structural bias and trend implications
- Identify most significant recent structural changes

### 4. TREND STRUCTURE EVALUATION
- Confirm trend direction and strength
- Assess trend structure integrity and quality
- Evaluate higher-high/higher-low formations
- Identify potential trend reversal signals

### 5. SUPPORT/RESISTANCE LEVEL ANALYSIS
- Validate key level identification and strength
- Assess level respect and rejection patterns
- Evaluate current price position relative to levels
- Identify most significant upcoming levels

### 6. VOLUME CORRELATION ANALYSIS
- Assess volume-price relationship strength
- Evaluate volume patterns at key structural events
- Identify volume confirmation or divergence signals
- Assess volume trend correlation quality

### 7. MARKET STATE ASSESSMENT
- Determine current market regime and phase
- Assess structural bias and directional conviction
- Evaluate market condition quality for trading
- Identify key transition or inflection signals

## OUTPUT REQUIREMENTS

Provide your analysis in the following structured format:

**NARRATIVE ANALYSIS:**
[Provide comprehensive written analysis covering all aspects above]

**STRUCTURED JSON RESPONSE:**
```json
{
  "symbol": "{symbol}",
  "analysis_timestamp": "ISO_TIMESTAMP",
  "chart_validation": {
    "chart_clarity": "excellent|good|fair|poor",
    "visual_numerical_agreement": "strong|medium|weak",
    "pattern_definition_quality": "very_clear|clear|moderate|unclear",
    "visual_confidence_boost": 0-25
  },
  "swing_analysis": {
    "total_swing_points": 0,
    "swing_quality_score": 0-100,
    "swing_density": "optimal|high|medium|low",
    "recent_swing_significance": "high|medium|low",
    "structure_integrity": "intact|weakening|compromised"
  },
  "structural_breaks": {
    "bos_events_count": 0,
    "choch_events_count": 0,
    "recent_break_type": "bullish_bos|bearish_bos|bullish_choch|bearish_choch|none",
    "break_strength": "very_strong|strong|medium|weak|none",
    "structural_bias": "strongly_bullish|bullish|neutral|bearish|strongly_bearish",
    "break_quality": "clean|acceptable|messy"
  },
  "trend_structure": {
    "trend_direction": "strong_uptrend|uptrend|weak_uptrend|sideways|weak_downtrend|downtrend|strong_downtrend",
    "trend_strength_score": 0-100,
    "higher_highs_lows_present": true|false,
    "trend_structure_quality": "excellent|good|fair|poor",
    "reversal_probability": "very_low|low|moderate|high|very_high"
  },
  "key_levels": {
    "nearest_resistance": 0.0,
    "nearest_support": 0.0,
    "resistance_strength": "very_strong|strong|medium|weak",
    "support_strength": "very_strong|strong|medium|weak",
    "price_position": "above_resistance|near_resistance|mid_range|near_support|below_support",
    "level_respect_quality": "excellent|good|fair|poor"
  },
  "volume_analysis": {
    "volume_trend_correlation": "very_strong|strong|medium|weak|none",
    "volume_at_breaks": "confirmation|neutral|divergence",
    "volume_pattern": "increasing|stable|decreasing|irregular",
    "volume_strength": "very_high|high|medium|low|very_low"
  },
  "market_regime": {
    "current_regime": "trending|breakout|consolidation|reversal|transition",
    "regime_confidence": 0-100,
    "regime_quality": "very_clear|clear|moderate|unclear",
    "phase": "early|developing|mature|late"
  },
  "confidence_assessment": {
    "overall_confidence": 0-100,
    "visual_confirmation_boost": 0-25,
    "analysis_quality": "excellent|good|fair|poor",
    "key_uncertainty_factors": ["factor1", "factor2"]
  },
  "actionable_insights": {
    "primary_insight": "Main takeaway from analysis",
    "key_levels_to_watch": [0.0, 0.0],
    "structural_signals": ["signal1", "signal2"],
    "risk_factors": ["risk1", "risk2"]
  }
}
```

## ANALYSIS GUIDELINES

1. **Balance Visual and Numerical**: Give appropriate weight to both visual and algorithmic findings
2. **Highlight Confirmations**: Emphasize where visual and numerical analysis agree
3. **Identify Discrepancies**: Note any conflicts between visual and algorithmic findings
4. **Assess Quality**: Evaluate the reliability and clarity of patterns and signals
5. **Be Specific**: Provide precise assessments rather than vague generalizations
6. **Focus on Structure**: Emphasize market structure elements over price predictions
7. **Consider Context**: Evaluate patterns within broader market context

## IMPORTANT NOTES

- Base analysis strictly on provided data and visual chart
- Avoid making price predictions or trading recommendations
- Focus on structural analysis and pattern recognition
- Highlight both strengths and limitations in your assessment
- Use visual analysis to enhance rather than replace algorithmic findings
- Be objective about pattern quality and signal strength

Output your complete narrative analysis first, followed by the JSON response. Ensure the JSON is properly formatted and contains no trailing text.
"""

    async def analyze_market_structure(self,
                                     stock_data: Dict[str, Any],
                                     analysis_data: Dict[str, Any],
                                     symbol: str,
                                     scenario_description: str = "Market Analysis") -> Dict[str, Any]:
        """
        Complete integrated market structure analysis with chart generation and LLM analysis.
        
        Args:
            stock_data: Stock price and volume data
            analysis_data: Market structure analysis results
            symbol: Stock symbol
            scenario_description: Description of the market scenario
            
        Returns:
            Dictionary containing complete analysis results
        """
        try:
            logger.info(f"🚀 Starting integrated market structure analysis for {symbol}")
            
            # Step 1: Generate optimized chart
            chart_result = await self._generate_chart(stock_data, analysis_data, symbol, scenario_description)
            if not chart_result['success']:
                return self._build_error_result(f"Chart generation failed: {chart_result['error']}", symbol)
            
            # Step 2: Create enhanced prompt with chart context
            prompt, chart_metadata = await self._create_enhanced_prompt(
                analysis_data, symbol, scenario_description, chart_result['chart_path']
            )
            
            # Step 3: Execute multimodal LLM analysis
            llm_result = await self._execute_llm_analysis(prompt, chart_result['chart_path'])
            if not llm_result['success']:
                return self._build_error_result(f"LLM analysis failed: {llm_result['error']}", symbol)
            
            # Step 4: Parse and validate response
            structured_result = await self._parse_and_validate_response(
                llm_result['response'], analysis_data, symbol, chart_result
            )
            
            # Step 5: Create comprehensive result
            final_result = self._build_final_result(
                structured_result, chart_result, chart_metadata, analysis_data, symbol, scenario_description
            )
            
            # Step 6: Save results
            result_path = await self._save_analysis_result(final_result, symbol, scenario_description)
            final_result['result_path'] = result_path
            
            logger.info(f"✅ Integrated analysis completed for {symbol}: {result_path}")
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Integrated analysis failed for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return self._build_error_result(f"Analysis failed: {str(e)}", symbol)
    
    async def _generate_chart(self,
                            stock_data: Dict[str, Any],
                            analysis_data: Dict[str, Any],
                            symbol: str,
                            scenario_description: str) -> Dict[str, Any]:
        """Generate optimized market structure chart."""
        try:
            logger.info(f"📊 Generating chart for {symbol}")
            
            # Use resilient chart generator with LLM optimization
            result = await asyncio.to_thread(
                self.chart_generator.generate_resilient_chart,
                stock_data=stock_data,
                analysis_data=analysis_data,
                symbol=symbol,
                chart_title=f"{symbol} Market Structure - {scenario_description}",
                optimization_level="llm_optimized"  # Use best quality for LLM
            )
            
            if result['success']:
                logger.info(f"✅ Chart generated: {result['chart_path']}")
            else:
                logger.warning(f"⚠️ Chart generation issues: {result.get('warnings', [])}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Chart generation error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _create_enhanced_prompt(self,
                                    analysis_data: Dict[str, Any],
                                    symbol: str,
                                    scenario_description: str,
                                    chart_path: str) -> Tuple[str, Dict[str, Any]]:
        """Create enhanced prompt with chart context."""
        try:
            # Get chart metadata
            chart_metadata = self._get_chart_metadata(chart_path)
            
            # Format analysis context
            analysis_context = self._format_analysis_context(analysis_data)
            
            # Build enhanced prompt
            prompt = self.prompt_template.format(
                symbol=symbol,
                scenario_description=scenario_description,
                chart_size_kb=chart_metadata.get('size_kb', 0),
                generation_timestamp=datetime.now().isoformat(),
                analysis_context=analysis_context
            )
            
            logger.info(f"📝 Enhanced prompt created ({len(prompt)} chars)")
            return prompt, chart_metadata
            
        except Exception as e:
            logger.error(f"❌ Prompt creation error: {e}")
            return "", {}
    
    async def _execute_llm_analysis(self, prompt: str, chart_path: str) -> Dict[str, Any]:
        """Execute multimodal LLM analysis with chart."""
        try:
            if not self.llm_client:
                return {'success': False, 'error': 'LLM client not initialized'}
            
            logger.info(f"🤖 Executing LLM analysis with chart: {Path(chart_path).name}")
            
            # Load chart image
            chart_image = Image.open(chart_path)
            
            # Execute multimodal LLM analysis
            response = await self.llm_client.generate_with_images(
                prompt=prompt,
                images=[chart_image]
            )
            
            # Handle tuple response format
            if isinstance(response, tuple):
                response_text = response[0]
                token_usage = response[1] if len(response) > 1 else None
            else:
                response_text = response
                token_usage = None
            
            if not response_text or len(response_text.strip()) < 100:
                return {'success': False, 'error': 'Insufficient LLM response'}
            
            logger.info(f"✅ LLM analysis completed ({len(response_text)} chars)")
            return {
                'success': True,
                'response': response_text,
                'token_usage': token_usage
            }
            
        except Exception as e:
            logger.error(f"❌ LLM analysis error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _parse_and_validate_response(self,
                                         llm_response: str,
                                         analysis_data: Dict[str, Any],
                                         symbol: str,
                                         chart_result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate LLM response structure."""
        try:
            # Extract narrative and JSON sections
            narrative, json_data = self._extract_response_sections(llm_response)
            
            # Validate JSON structure
            if json_data:
                validation_result = self._validate_response_structure(json_data)
                if not validation_result['valid']:
                    logger.warning(f"⚠️ Response validation issues: {validation_result['issues']}")
            
            return {
                'success': True,
                'narrative_analysis': narrative,
                'structured_data': json_data,
                'full_response': llm_response,
                'validation_result': validation_result if json_data else None
            }
            
        except Exception as e:
            logger.error(f"❌ Response parsing error: {e}")
            return {
                'success': False,
                'error': str(e),
                'full_response': llm_response
            }
    
    def _extract_response_sections(self, response: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Extract narrative and JSON sections from response."""
        try:
            # Find JSON section
            json_start = response.rfind('```json')
            json_end = response.rfind('```')
            
            if json_start != -1 and json_end != -1 and json_start < json_end:
                # Extract JSON from code block
                json_text = response[json_start + 7:json_end].strip()
                narrative = response[:json_start].strip()
            else:
                # Try to find raw JSON at the end
                brace_start = response.rfind('{')
                if brace_start != -1:
                    json_text = response[brace_start:].strip()
                    narrative = response[:brace_start].strip()
                else:
                    # No JSON found
                    return response, None
            
            # Parse JSON
            try:
                json_data = json.loads(json_text)
                return narrative, json_data
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ JSON parsing failed: {e}")
                return response, None
                
        except Exception as e:
            logger.error(f"❌ Section extraction error: {e}")
            return response, None
    
    def _validate_response_structure(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the structure of the JSON response."""
        required_sections = [
            'symbol', 'chart_validation', 'swing_analysis', 'structural_breaks',
            'trend_structure', 'key_levels', 'volume_analysis', 'market_regime',
            'confidence_assessment', 'actionable_insights'
        ]
        
        issues = []
        valid = True
        
        for section in required_sections:
            if section not in json_data:
                issues.append(f"Missing section: {section}")
                valid = False
            elif not isinstance(json_data[section], dict):
                issues.append(f"Section {section} is not a dictionary")
                valid = False
        
        return {
            'valid': valid,
            'issues': issues,
            'completeness_score': (len(required_sections) - len(issues)) / len(required_sections)
        }
    
    def _get_chart_metadata(self, chart_path: str) -> Dict[str, Any]:
        """Get metadata for the generated chart."""
        try:
            chart_file = Path(chart_path)
            file_size_bytes = chart_file.stat().st_size
            file_size_kb = round(file_size_bytes / 1024, 1)
            
            return {
                'path': str(chart_path),
                'filename': chart_file.name,
                'size_bytes': file_size_bytes,
                'size_kb': file_size_kb,
                'exists': chart_file.exists()
            }
        except Exception as e:
            logger.error(f"❌ Chart metadata error: {e}")
            return {}
    
    def _format_analysis_context(self, analysis_data: Dict[str, Any]) -> str:
        """Format analysis data into readable context."""
        context_parts = []
        
        try:
            # Swing Points Analysis
            swing_points = analysis_data.get('swing_points', {})
            if swing_points:
                context_parts.append(f"""
### SWING POINT ANALYSIS
- Total Swing Points: {swing_points.get('total_swings', 0)}
- Swing Highs: {len(swing_points.get('swing_highs', []))}
- Swing Lows: {len(swing_points.get('swing_lows', []))}
- Swing Density: {swing_points.get('swing_density', 0):.3f}
- Quality Score: {swing_points.get('quality_score', 0)}/100""")
            
            # BOS/CHOCH Analysis
            bos_choch = analysis_data.get('bos_choch_analysis', {})
            if bos_choch:
                context_parts.append(f"""
### STRUCTURAL BREAK ANALYSIS
- BOS Events: {len(bos_choch.get('bos_events', []))}
- CHOCH Events: {len(bos_choch.get('choch_events', []))}
- Structural Bias: {bos_choch.get('structural_bias', 'neutral').title()}
- Recent Break Type: {bos_choch.get('recent_break_type', 'none').replace('_', ' ').title()}""")
            
            # Trend Analysis
            trend_analysis = analysis_data.get('trend_analysis', {})
            if trend_analysis:
                context_parts.append(f"""
### TREND STRUCTURE
- Direction: {trend_analysis.get('trend_direction', 'unknown').title()}
- Strength: {trend_analysis.get('trend_strength', 'unknown').title()}
- Quality: {trend_analysis.get('trend_quality', 'unknown').title()}
- Structure Score: {trend_analysis.get('structure_score', 0)}/100""")
            
            # Key Levels
            key_levels = analysis_data.get('key_levels', {})
            if key_levels:
                resistance = key_levels.get('nearest_resistance', {})
                support = key_levels.get('nearest_support', {})
                context_parts.append(f"""
### KEY LEVELS ANALYSIS
- Nearest Resistance: {resistance.get('level', 'N/A')} (Strength: {resistance.get('strength', 'N/A')})
- Nearest Support: {support.get('level', 'N/A')} (Strength: {support.get('strength', 'N/A')})
- Current Price: {key_levels.get('current_price', 'N/A')}
- Total Levels: {len(key_levels.get('levels', []))}""")
            
            # Volume Analysis
            volume_analysis = analysis_data.get('volume_analysis', {})
            if volume_analysis:
                context_parts.append(f"""
### VOLUME ANALYSIS
- Volume Trend: {volume_analysis.get('volume_trend', 'unknown').title()}
- Correlation with Price: {volume_analysis.get('correlation_strength', 'unknown').title()}
- Average Volume: {volume_analysis.get('average_volume', 'N/A')}
- Recent Volume Pattern: {volume_analysis.get('recent_pattern', 'unknown').title()}""")
            
            # Market Regime
            market_regime = analysis_data.get('market_regime', {})
            if market_regime:
                context_parts.append(f"""
### MARKET REGIME
- Current Regime: {market_regime.get('regime', 'unknown').title()}
- Confidence: {market_regime.get('confidence', 0):.0%}
- Volatility: {market_regime.get('volatility', 'unknown').title()}
- Trend Strength: {market_regime.get('trend_strength', 'unknown').title()}""")
            
        except Exception as e:
            logger.error(f"❌ Context formatting error: {e}")
            context_parts.append(f"### ERROR\nFailed to format analysis context: {str(e)}")
        
        return '\n'.join(context_parts) if context_parts else "No detailed analysis context available."
    
    def _build_final_result(self,
                          structured_result: Dict[str, Any],
                          chart_result: Dict[str, Any],
                          chart_metadata: Dict[str, Any],
                          analysis_data: Dict[str, Any],
                          symbol: str,
                          scenario_description: str) -> Dict[str, Any]:
        """Build comprehensive final result."""
        return {
            'success': True,
            'agent_name': self.name,
            'agent_version': self.version,
            'symbol': symbol,
            'scenario': scenario_description,
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'integrated_market_structure',
            
            # Chart Information
            'chart_info': {
                'generated': chart_result['success'],
                'chart_path': chart_result.get('chart_path'),
                'optimization_level': chart_result.get('optimization_level'),
                'generation_quality': chart_result.get('quality_level'),
                'metadata': chart_metadata
            },
            
            # LLM Analysis Results
            'llm_analysis': {
                'narrative_analysis': structured_result.get('narrative_analysis', ''),
                'structured_data': structured_result.get('structured_data', {}),
                'response_quality': self._assess_response_quality(structured_result.get('full_response', '')),
                'validation_result': structured_result.get('validation_result')
            },
            
            # Original Analysis Data
            'original_analysis': analysis_data,
            
            # Performance Metrics
            'performance_metrics': {
                'chart_generation_success': chart_result['success'],
                'llm_analysis_success': structured_result['success'],
                'overall_quality_score': self._calculate_overall_quality(structured_result, chart_result),
                'response_length': len(structured_result.get('full_response', '')),
                'has_structured_data': structured_result.get('structured_data') is not None
            },
            
            # Full Response
            'full_llm_response': structured_result.get('full_response', ''),
            'processing_errors': structured_result.get('error') or chart_result.get('error')
        }
    
    def _calculate_overall_quality(self,
                                 structured_result: Dict[str, Any],
                                 chart_result: Dict[str, Any]) -> int:
        """Calculate overall quality score (0-100)."""
        score = 0
        
        # Chart generation quality (30 points)
        if chart_result['success']:
            score += 20
            quality_level = chart_result.get('quality_level', 'minimal')
            if quality_level == 'full':
                score += 10
            elif quality_level == 'standard':
                score += 7
            elif quality_level == 'minimal':
                score += 5
        
        # LLM analysis quality (40 points)
        if structured_result['success']:
            score += 15
            if structured_result.get('structured_data'):
                score += 15
                validation = structured_result.get('validation_result', {})
                if validation and validation.get('valid', False):
                    score += 10
        
        # Response quality (30 points)
        response = structured_result.get('full_response', '')
        if len(response) > 1000:
            score += 10
        elif len(response) > 500:
            score += 7
        elif len(response) > 200:
            score += 5
        
        if structured_result.get('narrative_analysis'):
            score += 10
        
        if structured_result.get('structured_data'):
            score += 10
        
        return min(score, 100)
    
    def _assess_response_quality(self, response: str) -> str:
        """Assess the quality of the LLM response."""
        if len(response) < 500:
            return 'poor'
        elif len(response) < 1500:
            return 'fair'
        elif len(response) < 3000:
            return 'good'
        else:
            return 'excellent'
    
    async def _save_analysis_result(self,
                                  result: Dict[str, Any],
                                  symbol: str,
                                  scenario_description: str) -> str:
        """Save comprehensive analysis result."""
        try:
            # Create filename
            scenario_clean = scenario_description.replace(' ', '_').lower()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol}_{scenario_clean}_integrated_{timestamp}.json"
            filepath = self.results_dir / filename
            
            # Save result
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            logger.info(f"💾 Analysis result saved: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"❌ Save error: {e}")
            return ""
    
    def _build_error_result(self, error_message: str, symbol: str) -> Dict[str, Any]:
        """Build error result dictionary."""
        return {
            'success': False,
            'agent_name': self.name,
            'agent_version': self.version,
            'symbol': symbol,
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'integrated_market_structure_error'
        }

# Example usage and testing
async def main():
    """Example usage of the integrated market structure agent."""
    
    # Initialize agent
    agent = IntegratedMarketStructureAgent()
    
    # Example market data (this would come from your data pipeline)
    example_stock_data = {
        'prices': [100, 102, 101, 105, 108, 106, 110, 112, 109, 115],
        'volumes': [1000, 1200, 900, 1500, 1800, 1100, 2000, 2200, 1700, 2500],
        'timestamps': list(range(10))
    }
    
    example_analysis_data = {
        'swing_points': {
            'total_swings': 6,
            'swing_highs': [{'price': 112, 'index': 7}],
            'swing_lows': [{'price': 101, 'index': 2}],
            'swing_density': 0.6,
            'quality_score': 85
        },
        'bos_choch_analysis': {
            'bos_events': [{'type': 'bullish_bos', 'price': 108}],
            'choch_events': [],
            'structural_bias': 'bullish',
            'recent_break_type': 'bullish_bos'
        },
        'trend_analysis': {
            'trend_direction': 'uptrend',
            'trend_strength': 'strong',
            'trend_quality': 'good',
            'structure_score': 78
        }
    }
    
    # Execute integrated analysis
    result = await agent.analyze_market_structure(
        stock_data=example_stock_data,
        analysis_data=example_analysis_data,
        symbol="EXAMPLE",
        scenario_description="Strong Uptrend Example"
    )
    
    if result['success']:
        print("✅ Integrated analysis completed successfully!")
        print(f"📊 Chart: {result['chart_info']['chart_path']}")
        print(f"💾 Result: {result.get('result_path')}")
        print(f"⭐ Quality Score: {result['performance_metrics']['overall_quality_score']}/100")
    else:
        print(f"❌ Analysis failed: {result['error']}")

if __name__ == "__main__":
    asyncio.run(main())