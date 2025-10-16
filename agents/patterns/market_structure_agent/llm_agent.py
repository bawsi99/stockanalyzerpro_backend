#!/usr/bin/env python3
"""
Market Structure Agent - LLM Integration Module

This module handles LLM integration for the Market Structure Agent using the new backend/llm system.
It manages all prompt processing internally, eliminating dependencies on backend/gemini components.
"""

import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# Import the new LLM system
from llm import get_llm_client


class MarketStructureLLMAgent:
    """
    LLM integration agent for Market Structure Analysis
    
    This agent handles all prompt processing internally and uses the new backend/llm system
    directly, eliminating dependencies on backend/gemini components like PromptManager,
    ContextEngineer, and GeminiClient.
    """
    
    def __init__(self):
        """Initialize the LLM agent with backend/llm client"""
        try:
            # Use the market_structure_agent configuration from llm_assignments.yaml
            self.llm_client = get_llm_client("market_structure_agent")
            print("✅ Market Structure LLM Agent initialized with backend/llm")
        except Exception as e:
            print(f"❌ Failed to initialize Market Structure LLM Agent: {e}")
            self.llm_client = None
            
        # Load the prompt template
        self.prompt_template = self._load_prompt_template()
    
    def _load_prompt_template(self) -> str:
        """Load the prompt template from the text file"""
        try:
            template_path = Path(__file__).parent / "market_structure_analysis.txt"
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"❌ Failed to load prompt template: {e}")
            return self._get_fallback_prompt_template()
    
    def _get_fallback_prompt_template(self) -> str:
        """Fallback prompt template if file loading fails"""
        return """You are a Market Structure Analysis Specialist. Your sole purpose is to analyze and report on market structure patterns and structural breaks.

## Your Specific Task:
Analyze market structure data to identify swing points, structural breaks, and provide a concise structural assessment report.

## Required Output Format:
Output ONLY a valid JSON object. NO PROSE, NO EXPLANATIONS OUTSIDE THE JSON.

## Analysis Context:
{context}
"""
    
    def build_analysis_prompt(self, analysis_data: Dict[str, Any], symbol: str, chart_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Build the complete analysis prompt with context and chart legend.
        
        Args:
            analysis_data: Processed data from MarketStructureProcessor
            symbol: Stock symbol being analyzed
            chart_metadata: Optional chart info (size_kb, path, generated)
        """
        context = self._build_analysis_context(analysis_data, symbol)

        # Chart metadata block (optional)
        chart_meta_block = ""
        if chart_metadata:
            chart_meta_block = (
                "## Chart Metadata\n"
                f"- Symbol: {symbol}\n"
                f"- File Size: {chart_metadata.get('size_kb', 0)} KB\n"
                f"- Generated: {chart_metadata.get('generation_timestamp', datetime.now().isoformat())}\n\n"
            )

        # Legend / How to read the chart
        legend_block = (
            "## Chart Legend / How to Read\n"
            "- Price: blue line with high/low shading\n"
            "- Swing High: red ▲ (label 'SH' for strong)\n"
            "- Swing Low: green ▼ (label 'SL' for strong)\n"
            "- BOS (Break of Structure): green 'BOS ↑' (bullish), red 'BOS ↓' (bearish)\n"
            "- CHoCH (Change of Character): annotated 'CHoCH' in trend color\n"
            "- Support Levels: green horizontal lines (weak : / medium -- / strong —)\n"
            "- Resistance Levels: red horizontal lines (weak : / medium -- / strong —)\n"
            "- Volume: bars (20D MA in blue) in the bottom subplot\n\n"
        )

        # Inject context into template if placeholder exists; else append at end
        template = self.prompt_template or ""
        if "{context}" in template:
            body = template.replace("{context}", context)
        else:
            body = f"{template}\n\n## Analysis Context:\n{context}"

        # Prepend chart info and legend so the LLM sees it first
        full_prompt = f"{chart_meta_block}{legend_block}{body}".strip()
        return full_prompt
    
    def _build_analysis_context(self, analysis_data: Dict[str, Any], symbol: str) -> str:
        """
        Build the analysis context from processor data.
        
        This replaces the context engineering functionality from backend/gemini
        by directly processing the analysis data into a structured context.
        """
        try:
            # Extract key components for context summary
            swing_points = analysis_data.get('swing_points', {})
            bos_choch_analysis = analysis_data.get('bos_choch_analysis', {})
            trend_analysis = analysis_data.get('trend_analysis', {})
            key_levels = analysis_data.get('key_levels', {})
            structure_quality = analysis_data.get('structure_quality', {})
            current_state = analysis_data.get('current_state', {})
            fractal_analysis = analysis_data.get('fractal_analysis', {})
            
            # Build structured context
            context = f"""Stock: {symbol}
Analysis Timestamp: {datetime.now().isoformat()}

MARKET STRUCTURE ANALYSIS DATA:
{json.dumps(analysis_data, indent=2, default=str)}

KEY METRICS SUMMARY:
SWING POINT ANALYSIS:
- Total Swing Points: {swing_points.get('total_swings', 0)}
- Swing Highs: {len(swing_points.get('swing_highs', []))}
- Swing Lows: {len(swing_points.get('swing_lows', []))}
- Swing Density: {swing_points.get('swing_density', 0):.4f}
- Analysis Method: {swing_points.get('analysis_method', 'unknown')}

BOS/CHOCH ANALYSIS:
- Total BOS Events: {bos_choch_analysis.get('total_bos_events', 0)}
- Total CHOCH Events: {bos_choch_analysis.get('total_choch_events', 0)}
- Structural Bias: {bos_choch_analysis.get('structural_bias', 'unknown')}
- Recent Structural Break: {bos_choch_analysis.get('recent_structural_break', {}).get('type', 'None') if bos_choch_analysis.get('recent_structural_break') else 'None'}

TREND STRUCTURE:
- Trend Direction: {trend_analysis.get('trend_direction', 'unknown')}
- Trend Strength: {trend_analysis.get('trend_strength', 'unknown')}
- Trend Quality: {trend_analysis.get('trend_quality', 'unknown')}
- Current Price: {trend_analysis.get('current_price', 0):.2f}
- Range High: {trend_analysis.get('range_high', 0):.2f}
- Range Low: {trend_analysis.get('range_low', 0):.2f}
- Price Position in Range: {trend_analysis.get('price_position_in_range', 0):.2f}

KEY LEVELS:
- Total Support Levels: {len(key_levels.get('support_levels', []))}
- Total Resistance Levels: {len(key_levels.get('resistance_levels', []))}
- Nearest Support: {key_levels.get('nearest_support', {}).get('level', 'None') if key_levels.get('nearest_support') else 'None'}
- Nearest Resistance: {key_levels.get('nearest_resistance', {}).get('level', 'None') if key_levels.get('nearest_resistance') else 'None'}

STRUCTURE QUALITY ASSESSMENT:
- Quality Score: {structure_quality.get('quality_score', 0)}/100
- Quality Rating: {structure_quality.get('quality_rating', 'unknown')}
- Quality Factors: {', '.join(structure_quality.get('quality_factors', []))}

CURRENT MARKET STATE:
- Structure State: {current_state.get('structure_state', 'unknown')}
- Price Position Description: {current_state.get('price_position_description', 'unknown')}
- Price Momentum (10d): {current_state.get('price_momentum_10d', 0):.2f}%
- Position in Recent Range: {current_state.get('position_in_recent_range', 0):.2f}
- Trend Alignment: {current_state.get('trend_alignment', 'unknown')}

FRACTAL ANALYSIS:
- Timeframe Alignment: {fractal_analysis.get('timeframe_alignment', 'unknown')}
- Trend Consensus: {fractal_analysis.get('trend_consensus', 'unknown')}"""
            
            # Add details for recent swing points
            swing_highs = swing_points.get('swing_highs', [])
            swing_lows = swing_points.get('swing_lows', [])
            
            if swing_highs:
                context += "\n\nRECENT SWING HIGHS:"
                for i, swing_high in enumerate(swing_highs[-3:], 1):  # Show last 3 swing highs
                    context += f"\n  {i}. {swing_high.get('date', 'N/A')}: {swing_high.get('price', 0):.2f} ({swing_high.get('strength', 'unknown')} strength)"
            
            if swing_lows:
                context += "\n\nRECENT SWING LOWS:"
                for i, swing_low in enumerate(swing_lows[-3:], 1):  # Show last 3 swing lows
                    context += f"\n  {i}. {swing_low.get('date', 'N/A')}: {swing_low.get('price', 0):.2f} ({swing_low.get('strength', 'unknown')} strength)"
            
            # Add BOS/CHOCH events details
            bos_events = bos_choch_analysis.get('bos_events', [])
            choch_events = bos_choch_analysis.get('choch_events', [])
            
            if bos_events:
                context += "\n\nRECENT BOS EVENTS:"
                for i, bos in enumerate(bos_events[-3:], 1):  # Show last 3 BOS events
                    context += f"\n  {i}. {bos.get('date', 'N/A')}: {bos.get('type', 'unknown')} at {bos.get('break_price', 0):.2f} ({bos.get('strength', 'unknown')} strength)"
            
            if choch_events:
                context += "\n\nRECENT CHOCH EVENTS:"
                for i, choch in enumerate(choch_events[-2:], 1):  # Show last 2 CHOCH events
                    context += f"\n  {i}. {choch.get('date', 'N/A')}: {choch.get('type', 'unknown')} - {choch.get('description', 'N/A')}"
            
            # Add key levels details
            support_levels = key_levels.get('support_levels', [])
            resistance_levels = key_levels.get('resistance_levels', [])
            
            if support_levels:
                context += "\n\nKEY SUPPORT LEVELS:"
                for i, support in enumerate(support_levels[-3:], 1):  # Show last 3 support levels
                    context += f"\n  {i}. {support.get('level', 0):.2f} ({support.get('strength', 'unknown')} - from {support.get('date', 'N/A')})"
            
            if resistance_levels:
                context += "\n\nKEY RESISTANCE LEVELS:"
                for i, resistance in enumerate(resistance_levels[-3:], 1):  # Show last 3 resistance levels
                    context += f"\n  {i}. {resistance.get('level', 0):.2f} ({resistance.get('strength', 'unknown')} - from {resistance.get('date', 'N/A')})"
            
            context += f"\n\nDATA QUALITY METRICS:"
            data_quality = analysis_data.get('data_quality', {})
            context += f"\n- Data Length: {data_quality.get('data_length', 0)} periods"
            context += f"\n- Length Quality: {data_quality.get('length_quality', 'unknown')}"
            context += f"\n- Overall Quality Score: {data_quality.get('overall_quality_score', 0)}/100"
            context += f"\n- Analysis Confidence: {analysis_data.get('confidence_score', 0):.2f}"
            
            context += f"\n\nPlease analyze this comprehensive market structure data to provide detailed insights about structural elements, trend quality, swing point significance, BOS/CHOCH implications, and actionable trading insights for {symbol}."
            
            return context
            
        except Exception as e:
            # Fallback context if processing fails
            return f"""Stock: {symbol}
Analysis Timestamp: {datetime.now().isoformat()}

MARKET STRUCTURE ANALYSIS DATA:
{json.dumps(analysis_data, indent=2, default=str)}

Error processing context: {str(e)}

Please analyze this market structure data to identify key structural elements and provide trading insights."""
    
    async def analyze_market_structure(
            self, 
            chart_image: bytes,
            analysis_data: Dict[str, Any], 
            symbol: str
        ) -> Optional[str]:
            """
            Main analysis method using backend/llm system.
            
            This replaces the old GeminiClient.analyze_agent_specific() method
            by calling backend/llm directly with the complete prompt built internally.
            
            Args:
                chart_image: Chart image bytes from MarketStructureCharts
                analysis_data: Processed data from MarketStructureProcessor
                symbol: Stock symbol being analyzed
                
            Returns:
                LLM analysis response as JSON string, or error response
            """
            if not self.llm_client:
                error_msg = "LLM client not initialized"
                print(f"[MARKET_STRUCTURE_LLM] {error_msg}")
                return f'{{"error": "{error_msg}", "agent": "market_structure", "status": "failed"}}'
            
            try:
                # Build the complete prompt with chart metadata
                chart_metadata = {
                    "size_kb": round(len(chart_image) / 1024, 1) if chart_image else 0,
                    "generation_timestamp": datetime.now().isoformat(),
                }
                prompt = self.build_analysis_prompt(analysis_data, symbol, chart_metadata=chart_metadata)
                
                print(f"[MARKET_STRUCTURE_LLM] Sending analysis request for {symbol}")
                print(f"[MARKET_STRUCTURE_LLM] Prompt length: {len(prompt)} characters")
                print(f"[MARKET_STRUCTURE_LLM] Chart image size: {len(chart_image)} bytes")
                
                # Call backend/llm with image and prompt
                response = await self.llm_client.generate(
                    prompt=prompt,
                    images=[chart_image],
                    enable_code_execution=False  # Market structure doesn't need code execution
                )
                
                print(f"[MARKET_STRUCTURE_LLM] Analysis completed for {symbol}")
                print(f"[MARKET_STRUCTURE_LLM] Response length: {len(response) if response else 0} characters")
                
                # Validate response is not empty
                if not response or not response.strip():
                    error_msg = "Empty response from LLM"
                    print(f"[MARKET_STRUCTURE_LLM] {error_msg}")
                    return f'{{"error": "{error_msg}", "agent": "market_structure", "status": "failed"}}'
                
                return response
                
            except asyncio.TimeoutError:
                error_msg = f"LLM request timed out for {symbol}"
                print(f"[MARKET_STRUCTURE_LLM] {error_msg}")
                return f'{{"error": "{error_msg}", "agent": "market_structure", "status": "timeout"}}'
                
            except Exception as e:
                error_msg = f"Market structure LLM analysis failed for {symbol}: {str(e)}"
                print(f"[MARKET_STRUCTURE_LLM] {error_msg}")
                import traceback
                print(f"[MARKET_STRUCTURE_LLM] Traceback: {traceback.format_exc()}")
                return f'{{"error": "{error_msg}", "agent": "market_structure", "status": "failed"}}'
    
    def get_client_info(self) -> str:
        """Get information about the LLM client being used"""
        if self.llm_client:
            return self.llm_client.get_provider_info()
        return "No LLM client initialized"


# Test function for the new LLM agent
async def test_market_structure_llm_agent():
    """Test the Market Structure LLM Agent"""
    print("🧪 Testing Market Structure LLM Agent")
    print("=" * 50)
    
    try:
        # Create agent
        llm_agent = MarketStructureLLMAgent()
        print(f"✅ Agent created successfully")
        print(f"   Client info: {llm_agent.get_client_info()}")
        
        # Test prompt building
        sample_analysis_data = {
            "swing_points": {
                "swing_highs": [
                    {"date": "2024-01-15", "price": 150.25, "strength": "strong", "type": "swing_high"},
                    {"date": "2024-01-22", "price": 155.80, "strength": "medium", "type": "swing_high"}
                ],
                "swing_lows": [
                    {"date": "2024-01-12", "price": 145.30, "strength": "strong", "type": "swing_low"},
                    {"date": "2024-01-25", "price": 148.75, "strength": "medium", "type": "swing_low"}
                ],
                "total_swings": 4,
                "swing_density": 0.05,
                "analysis_method": "local_extrema"
            },
            "bos_choch_analysis": {
                "bos_events": [
                    {"type": "bullish_bos", "date": "2024-01-20", "break_price": 152.50, "strength": "strong"}
                ],
                "choch_events": [],
                "total_bos_events": 1,
                "total_choch_events": 0,
                "structural_bias": "bullish"
            },
            "trend_analysis": {
                "trend_direction": "uptrend",
                "trend_strength": "strong",
                "trend_quality": "good",
                "current_price": 151.45,
                "range_high": 156.00,
                "range_low": 144.50,
                "price_position_in_range": 0.65
            },
            "key_levels": {
                "support_levels": [{"level": 148.75, "strength": "strong", "date": "2024-01-25"}],
                "resistance_levels": [{"level": 155.80, "strength": "medium", "date": "2024-01-22"}],
                "nearest_support": {"level": 148.75, "strength": "strong"},
                "nearest_resistance": {"level": 155.80, "strength": "medium"},
                "current_price": 151.45
            },
            "structure_quality": {
                "quality_score": 82,
                "quality_rating": "good",
                "quality_factors": ["optimal_swing_density", "high_trend_consistency", "strong_bos_present"]
            },
            "current_state": {
                "structure_state": "trending_up",
                "price_position_description": "mid_range",
                "price_momentum_10d": 3.2,
                "position_in_recent_range": 0.65,
                "trend_alignment": "uptrend"
            },
            "fractal_analysis": {
                "timeframe_alignment": "aligned",
                "trend_consensus": "uptrend"
            },
            "confidence_score": 0.85,
            "data_quality": {
                "data_length": 60,
                "length_quality": "excellent",
                "overall_quality_score": 95
            }
        }
        
        prompt = llm_agent.build_analysis_prompt(sample_analysis_data, "TEST_STOCK")
        print(f"✅ Prompt built successfully")
        print(f"   Prompt length: {len(prompt)} characters")
        print(f"   Contains required sections: {all(section in prompt for section in ['Analysis Context', 'Stock: TEST_STOCK', 'SWING POINT ANALYSIS'])}")
        
        # Test would require actual chart image for full test
        print(f"✅ Market Structure LLM Agent test completed")
        print(f"   Ready for integration with patterns agents orchestrator")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run test
    asyncio.run(test_market_structure_llm_agent())