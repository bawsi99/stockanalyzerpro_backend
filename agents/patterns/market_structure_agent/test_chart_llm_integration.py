#!/usr/bin/env python3
"""
Market Structure Chart + LLM Integration Demo

This module demonstrates how to integrate generated market structure charts 
with LLM analysis using multimodal capabilities.

Shows:
1. Chart generation with metadata
2. Enhanced prompt creation with visual context
3. Mock LLM analysis workflow
4. Chart information integration into prompts
"""

import os
import base64
import json
from pathlib import Path
from typing import Dict, Any, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChartLLMIntegrator:
    """
    Integrates market structure charts with LLM analysis for enhanced insights.
    """
    
    def __init__(self, charts_dir: str = "test_market_structure_charts"):
        self.charts_dir = Path(charts_dir)
        
        # Market structure prompt template with chart integration
        self.enhanced_prompt_template = """You are a Market Structure Analysis Specialist with multimodal visual analysis capabilities. 

## VISUAL CHART ANALYSIS

A comprehensive market structure chart has been provided showing:
{visual_elements}

Chart Details:
- File: {chart_filename}
- Size: {chart_size_kb} KB
- Scenario: {scenario_description}
- Symbol: {symbol}

## ANALYSIS CONTEXT

{analysis_context}

## YOUR ENHANCED TASK

Using BOTH the visual chart and the numerical data provided:

1. **Visual Pattern Recognition**: Identify market structure patterns visible in the chart
2. **Swing Point Validation**: Verify algorithmic swing points against visual patterns
3. **BOS/CHOCH Confirmation**: Validate structural breaks using visual analysis
4. **Level Quality Assessment**: Assess support/resistance level strength visually
5. **Trend Structure Verification**: Confirm trend analysis using visual trend lines
6. **Volume Correlation**: Analyze volume patterns in relation to price structure

## ENHANCED OUTPUT FORMAT

Provide your analysis in JSON format with visual insights:

```json
{{
  "visual_analysis": {{
    "chart_clarity": "excellent/good/fair/poor",
    "pattern_recognition": "Clear visual patterns identified",
    "visual_confirmation": "How visual analysis confirms/contradicts numerical data"
  }},
  "swing_points": {{
    "total_swing_points": 0,
    "visual_validation": "strong/medium/weak",
    "swing_quality": "strong/medium/weak", 
    "recent_swing_highs": 0,
    "recent_swing_lows": 0,
    "swing_density": "optimal/high/low"
  }},
  "structural_breaks": {{
    "bos_events": 0,
    "choch_events": 0,
    "visual_break_quality": "clean/messy/unclear",
    "recent_break_type": "bullish_bos/bearish_bos/bullish_choch/bearish_choch/none",
    "structural_bias": "bullish/bearish/neutral",
    "break_strength": "strong/medium/weak/none"
  }},
  "trend_structure": {{
    "trend_direction": "uptrend/downtrend/sideways",
    "trend_strength": "strong/medium/weak",
    "visual_trend_clarity": "very_clear/clear/unclear",
    "higher_highs_lows": "present/absent/mixed",
    "structure_integrity": "intact/weakening/broken"
  }},
  "key_levels": {{
    "nearest_resistance": 0.0,
    "nearest_support": 0.0,
    "level_visual_strength": "strong/medium/weak",
    "current_position": "near_support/near_resistance/mid_range",
    "level_count": 0
  }},
  "volume_analysis": {{
    "volume_trend_correlation": "strong/medium/weak/none",
    "volume_at_key_levels": "significant/normal/low",
    "volume_pattern": "increasing/decreasing/irregular"
  }},
  "current_state": "trending_up/trending_down/consolidating/testing_resistance/testing_support",
  "confidence_score": 0-100,
  "visual_confidence_boost": 0-20,
  "key_insight": "Enhanced insight combining visual and numerical analysis"
}}
```

## INSTRUCTIONS

1. Carefully examine the provided market structure chart
2. Cross-reference visual patterns with numerical data
3. Use visual analysis to enhance confidence in algorithmic findings
4. Identify any discrepancies between visual and numerical analysis
5. Provide actionable insights based on comprehensive analysis

Focus on market structure analysis and visual validation. Avoid trading recommendations.
"""

    def encode_chart_to_base64(self, chart_path: str) -> str:
        """Encode chart image to base64 for LLM integration"""
        try:
            with open(chart_path, 'rb') as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode chart: {e}")
            return None

    def get_chart_metadata(self, chart_path: str) -> Dict[str, Any]:
        """Extract chart metadata for prompt enhancement"""
        try:
            chart_file = Path(chart_path)
            file_size_bytes = chart_file.stat().st_size
            file_size_kb = round(file_size_bytes / 1024, 1)
            
            # Parse filename to extract scenario and symbol
            filename = chart_file.name
            parts = filename.replace('_market_structure.png', '').split('_')
            symbol = parts[0] if parts else 'UNKNOWN'
            scenario = '_'.join(parts[1:]) if len(parts) > 1 else 'unknown_scenario'
            
            return {
                'chart_filename': filename,
                'chart_path': str(chart_path),
                'chart_size_bytes': file_size_bytes,
                'chart_size_kb': file_size_kb,
                'symbol': symbol,
                'scenario': scenario.replace('_', ' ').title(),
                'visual_elements': [
                    'Price action with candlestick-style visualization',
                    'Swing points marked with strength-based sizing',
                    'BOS/CHOCH events with directional annotations',
                    'Support and resistance levels with strength indicators',
                    'Volume analysis with trend correlation',
                    'Trend structure indicators and market regime info',
                    'Comprehensive analysis summary metrics',
                    'Multi-panel layout with detailed breakdowns'
                ]
            }
        except Exception as e:
            logger.error(f"Failed to get chart metadata: {e}")
            return {}

    def build_enhanced_prompt(self, 
                            analysis_data: Dict[str, Any], 
                            chart_path: str) -> str:
        """Build enhanced prompt with chart integration"""
        
        # Get chart metadata
        chart_metadata = self.get_chart_metadata(chart_path)
        
        # Format visual elements
        visual_elements_text = '\n'.join([
            f"• {element}" for element in chart_metadata.get('visual_elements', [])
        ])
        
        # Build analysis context from numerical data
        analysis_context = self._format_analysis_context(analysis_data)
        
        # Build enhanced prompt
        enhanced_prompt = self.enhanced_prompt_template.format(
            visual_elements=visual_elements_text,
            chart_filename=chart_metadata.get('chart_filename', 'unknown.png'),
            chart_size_kb=chart_metadata.get('chart_size_kb', 0),
            scenario_description=chart_metadata.get('scenario', 'Unknown Scenario'),
            symbol=chart_metadata.get('symbol', 'UNKNOWN'),
            analysis_context=analysis_context
        )
        
        return enhanced_prompt, chart_metadata

    def _format_analysis_context(self, analysis_data: Dict[str, Any]) -> str:
        """Format analysis data into readable context"""
        
        context_parts = []
        
        # Swing points summary
        swing_points = analysis_data.get('swing_points', {})
        context_parts.append(f"""
SWING POINT ANALYSIS:
- Total Swing Points: {swing_points.get('total_swings', 0)}
- Swing Density: {swing_points.get('swing_density', 0):.3f}
- Swing Highs: {len(swing_points.get('swing_highs', []))}
- Swing Lows: {len(swing_points.get('swing_lows', []))}""")

        # BOS/CHOCH analysis
        bos_choch = analysis_data.get('bos_choch_analysis', {})
        context_parts.append(f"""
STRUCTURAL BREAK ANALYSIS:
- BOS Events: {len(bos_choch.get('bos_events', []))}
- CHOCH Events: {len(bos_choch.get('choch_events', []))}
- Structural Bias: {bos_choch.get('structural_bias', 'neutral').title()}""")

        # Trend analysis
        trend_analysis = analysis_data.get('trend_analysis', {})
        context_parts.append(f"""
TREND STRUCTURE:
- Direction: {trend_analysis.get('trend_direction', 'unknown').title()}
- Strength: {trend_analysis.get('trend_strength', 'unknown').title()}
- Quality: {trend_analysis.get('trend_quality', 'unknown').title()}""")

        # Structure quality
        quality = analysis_data.get('structure_quality', {})
        context_parts.append(f"""
STRUCTURE QUALITY:
- Quality Score: {quality.get('quality_score', 0)}/100
- Rating: {quality.get('quality_rating', 'unknown').title()}""")

        return '\n'.join(context_parts)

    def create_llm_analysis_package(self, 
                                  analysis_data: Dict[str, Any],
                                  chart_path: str) -> Dict[str, Any]:
        """Create complete package for LLM analysis"""
        
        # Build enhanced prompt
        enhanced_prompt, chart_metadata = self.build_enhanced_prompt(analysis_data, chart_path)
        
        # Encode chart for multimodal analysis
        chart_base64 = self.encode_chart_to_base64(chart_path)
        
        # Create analysis package
        analysis_package = {
            'prompt': enhanced_prompt,
            'chart_data': {
                'base64_image': chart_base64,
                'metadata': chart_metadata
            },
            'analysis_data': analysis_data,
            'multimodal_enabled': chart_base64 is not None,
            'analysis_type': 'market_structure_enhanced',
            'timestamp': '2024-10-15T21:16:00Z'
        }
        
        return analysis_package

    def simulate_llm_response(self, 
                            analysis_package: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate enhanced LLM response with visual analysis"""
        
        # This would normally call the actual LLM API
        # For demonstration, we'll create a mock enhanced response
        
        chart_metadata = analysis_package['chart_data']['metadata']
        symbol = chart_metadata.get('symbol', 'UNKNOWN')
        scenario = chart_metadata.get('scenario', 'Unknown')
        
        # Mock enhanced response based on scenario
        if 'uptrend' in scenario.lower():
            mock_response = {
                "visual_analysis": {
                    "chart_clarity": "excellent",
                    "pattern_recognition": "Clear ascending trend with well-defined swing structure",
                    "visual_confirmation": "Visual trend lines confirm algorithmic uptrend identification"
                },
                "swing_points": {
                    "total_swing_points": 10,
                    "visual_validation": "strong",
                    "swing_quality": "strong",
                    "recent_swing_highs": 2,
                    "recent_swing_lows": 2,
                    "swing_density": "optimal"
                },
                "structural_breaks": {
                    "bos_events": 3,
                    "choch_events": 0,
                    "visual_break_quality": "clean",
                    "recent_break_type": "bullish_bos",
                    "structural_bias": "bullish",
                    "break_strength": "strong"
                },
                "trend_structure": {
                    "trend_direction": "uptrend",
                    "trend_strength": "strong",
                    "visual_trend_clarity": "very_clear",
                    "higher_highs_lows": "present",
                    "structure_integrity": "intact"
                },
                "key_levels": {
                    "nearest_resistance": 177.3,
                    "nearest_support": 165.2,
                    "level_visual_strength": "strong",
                    "current_position": "near_resistance",
                    "level_count": 8
                },
                "volume_analysis": {
                    "volume_trend_correlation": "strong",
                    "volume_at_key_levels": "significant",
                    "volume_pattern": "increasing"
                },
                "current_state": "trending_up",
                "confidence_score": 92,
                "visual_confidence_boost": 15,
                "key_insight": f"Visual analysis strongly confirms {symbol} uptrend structure with clean BOS events and strong level respect"
            }
            
        elif 'downtrend' in scenario.lower():
            mock_response = {
                "visual_analysis": {
                    "chart_clarity": "good",
                    "pattern_recognition": "Clear descending pattern with multiple structural breaks",
                    "visual_confirmation": "Visual confirms strong bearish structure with clean BOS events"
                },
                "swing_points": {
                    "total_swing_points": 10,
                    "visual_validation": "strong",
                    "swing_quality": "medium",
                    "recent_swing_highs": 2,
                    "recent_swing_lows": 3,
                    "swing_density": "optimal"
                },
                "structural_breaks": {
                    "bos_events": 4,
                    "choch_events": 1,
                    "visual_break_quality": "clean",
                    "recent_break_type": "bearish_bos",
                    "structural_bias": "bearish",
                    "break_strength": "strong"
                },
                "trend_structure": {
                    "trend_direction": "downtrend",
                    "trend_strength": "strong",
                    "visual_trend_clarity": "clear",
                    "higher_highs_lows": "absent",
                    "structure_integrity": "broken"
                },
                "key_levels": {
                    "nearest_resistance": 145.8,
                    "nearest_support": 118.2,
                    "level_visual_strength": "medium",
                    "current_position": "mid_range",
                    "level_count": 8
                },
                "volume_analysis": {
                    "volume_trend_correlation": "medium",
                    "volume_at_key_levels": "normal",
                    "volume_pattern": "irregular"
                },
                "current_state": "trending_down",
                "confidence_score": 89,
                "visual_confidence_boost": 12,
                "key_insight": f"Visual analysis confirms {symbol} bearish structure with multiple BOS events indicating strong downtrend"
            }
            
        elif 'sideways' in scenario.lower():
            mock_response = {
                "visual_analysis": {
                    "chart_clarity": "good",
                    "pattern_recognition": "Clear ranging structure with defined boundaries",
                    "visual_confirmation": "Visual confirms consolidation pattern with multiple touches at key levels"
                },
                "swing_points": {
                    "total_swing_points": 12,
                    "visual_validation": "medium",
                    "swing_quality": "medium",
                    "recent_swing_highs": 3,
                    "recent_swing_lows": 3,
                    "swing_density": "high"
                },
                "structural_breaks": {
                    "bos_events": 0,
                    "choch_events": 2,
                    "visual_break_quality": "unclear",
                    "recent_break_type": "bearish_choch",
                    "structural_bias": "neutral",
                    "break_strength": "weak"
                },
                "trend_structure": {
                    "trend_direction": "sideways",
                    "trend_strength": "weak",
                    "visual_trend_clarity": "clear",
                    "higher_highs_lows": "mixed",
                    "structure_integrity": "intact"
                },
                "key_levels": {
                    "nearest_resistance": 155.7,
                    "nearest_support": 144.8,
                    "level_visual_strength": "strong",
                    "current_position": "mid_range",
                    "level_count": 6
                },
                "volume_analysis": {
                    "volume_trend_correlation": "weak",
                    "volume_at_key_levels": "low",
                    "volume_pattern": "decreasing"
                },
                "current_state": "consolidating",
                "confidence_score": 78,
                "visual_confidence_boost": 8,
                "key_insight": f"Visual analysis confirms {symbol} consolidation with well-defined range boundaries and decreasing volume"
            }
            
        else:  # Complex scenario
            mock_response = {
                "visual_analysis": {
                    "chart_clarity": "fair",
                    "pattern_recognition": "Complex multi-phase structure with trend transitions",
                    "visual_confirmation": "Visual shows multiple regime changes with mixed structural signals"
                },
                "swing_points": {
                    "total_swing_points": 12,
                    "visual_validation": "medium",
                    "swing_quality": "medium",
                    "recent_swing_highs": 2,
                    "recent_swing_lows": 2,
                    "swing_density": "optimal"
                },
                "structural_breaks": {
                    "bos_events": 4,
                    "choch_events": 2,
                    "visual_break_quality": "mixed",
                    "recent_break_type": "bullish_bos",
                    "structural_bias": "neutral",
                    "break_strength": "medium"
                },
                "trend_structure": {
                    "trend_direction": "uptrend",
                    "trend_strength": "medium",
                    "visual_trend_clarity": "unclear",
                    "higher_highs_lows": "mixed",
                    "structure_integrity": "weakening"
                },
                "key_levels": {
                    "nearest_resistance": 165.3,
                    "nearest_support": 132.7,
                    "level_visual_strength": "medium",
                    "current_position": "near_resistance",
                    "level_count": 6
                },
                "volume_analysis": {
                    "volume_trend_correlation": "medium",
                    "volume_at_key_levels": "normal",
                    "volume_pattern": "irregular"
                },
                "current_state": "testing_resistance",
                "confidence_score": 85,
                "visual_confidence_boost": 5,
                "key_insight": f"Visual analysis shows {symbol} in transitional phase with mixed signals requiring careful monitoring"
            }
        
        return {
            'response': mock_response,
            'analysis_type': 'market_structure_enhanced',
            'visual_analysis_enabled': True,
            'confidence_boost': mock_response['visual_confidence_boost'],
            'processing_time': 0.125  # Mock processing time
        }

    def save_analysis_result(self, 
                           analysis_package: Dict[str, Any],
                           llm_response: Dict[str, Any],
                           output_dir: str = "enhanced_analysis_results") -> str:
        """Save complete analysis result with chart integration"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        chart_metadata = analysis_package['chart_data']['metadata']
        symbol = chart_metadata.get('symbol', 'UNKNOWN')
        scenario = chart_metadata.get('scenario', 'unknown').replace(' ', '_').lower()
        
        # Create comprehensive result
        result = {
            'analysis_metadata': {
                'symbol': symbol,
                'scenario': scenario,
                'analysis_type': 'market_structure_enhanced_visual',
                'timestamp': analysis_package['timestamp'],
                'multimodal_enabled': analysis_package['multimodal_enabled']
            },
            'chart_info': chart_metadata,
            'enhanced_prompt': analysis_package['prompt'],
            'numerical_analysis': analysis_package['analysis_data'],
            'llm_response': llm_response['response'],
            'performance_metrics': {
                'confidence_score': llm_response['response']['confidence_score'],
                'visual_confidence_boost': llm_response['confidence_boost'],
                'processing_time': llm_response['processing_time']
            }
        }
        
        # Save result
        filename = f"{symbol}_{scenario}_enhanced_analysis.json"
        filepath = output_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Enhanced analysis result saved: {filepath}")
        return str(filepath)


def main():
    """Run chart + LLM integration demo"""
    
    logger.info("Starting Market Structure Chart + LLM Integration Demo...")
    
    # Initialize integrator
    integrator = ChartLLMIntegrator()
    
    # Find generated charts
    charts_dir = Path("test_market_structure_charts")
    chart_files = list(charts_dir.glob("*.png"))
    
    if not chart_files:
        logger.error("No charts found! Run test_chart_generation.py first.")
        return
    
    # Mock analysis data for each scenario
    from test_chart_generation import create_mock_data_scenarios
    scenarios = create_mock_data_scenarios()
    
    results = []
    
    for chart_file in chart_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing chart: {chart_file.name}")
        logger.info(f"{'='*60}")
        
        # Find matching scenario data
        chart_symbol = chart_file.name.split('_')[0]
        matching_scenario = None
        
        for stock_data, analysis_data, symbol, scenario in scenarios:
            if symbol == chart_symbol:
                matching_scenario = analysis_data
                break
        
        if not matching_scenario:
            logger.warning(f"No matching analysis data for {chart_symbol}")
            continue
        
        # Create analysis package
        analysis_package = integrator.create_llm_analysis_package(
            analysis_data=matching_scenario,
            chart_path=str(chart_file)
        )
        
        # Simulate LLM analysis
        llm_response = integrator.simulate_llm_response(analysis_package)
        
        # Save results
        result_path = integrator.save_analysis_result(analysis_package, llm_response)
        results.append(result_path)
        
        # Log key insights
        response_data = llm_response['response']
        logger.info(f"📊 Chart: {chart_file.name}")
        logger.info(f"🎯 Confidence: {response_data['confidence_score']}% (+{response_data['visual_confidence_boost']} visual boost)")
        logger.info(f"📈 Trend: {response_data['trend_structure']['trend_direction'].title()}")
        logger.info(f"⚡ State: {response_data['current_state'].replace('_', ' ').title()}")
        logger.info(f"💡 Insight: {response_data['key_insight']}")
        logger.info(f"💾 Saved: {result_path}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("ENHANCED ANALYSIS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Charts processed: {len(chart_files)}")
    logger.info(f"Analyses completed: {len(results)}")
    logger.info(f"Results saved in: enhanced_analysis_results/")
    
    logger.info(f"\n🎯 Key Benefits of Visual Integration:")
    logger.info(f"  • Enhanced pattern recognition capabilities")
    logger.info(f"  • Visual validation of algorithmic findings")
    logger.info(f"  • Improved confidence scoring with visual boost")
    logger.info(f"  • Better trend structure clarity assessment")
    logger.info(f"  • Volume correlation visual analysis")
    
    logger.info(f"\n✨ Chart + LLM integration demo completed!")


if __name__ == "__main__":
    main()