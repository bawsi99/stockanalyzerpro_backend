#!/usr/bin/env python3
"""
Volume Confirmation LLM Agent - Migrated to backend/llm

This module provides the LLM analysis capabilities for the Volume Confirmation Agent
using the new backend/llm framework instead of the legacy Gemini backend.

The agent now handles all prompt processing internally:
- Template loading and formatting
- Context engineering specific to volume confirmation
- Fully formatted prompt construction
- Direct LLM calls via backend/llm

Migration from backend/gemini:
- No longer uses PromptManager, ContextEngineer, or GeminiClient
- All prompt logic is self-contained within this agent
- Uses backend/llm.LLMClient for clean LLM calls
"""

import os
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

class VolumeConfirmationLLMAgent:
    """
    LLM Agent for Volume Confirmation Analysis
    
    Handles all prompt processing and LLM interactions for volume confirmation analysis.
    This replaces the previous dependency on backend/gemini services.
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize the Volume Confirmation LLM Agent
        
        Args:
            llm_client: Optional LLMClient instance. If None, creates one for 'volume_confirmation_agent'
        """
        if llm_client is None:
            from llm import get_llm_client
            self.llm_client = get_llm_client("volume_confirmation_agent")
        else:
            self.llm_client = llm_client
            
        # Load prompt template once during initialization
        self.prompt_template = self._load_prompt_template()
        
    def _load_prompt_template(self) -> str:
        """
        Load the volume confirmation prompt template.
        
        Previously handled by PromptManager, now handled directly by the agent.
        """
        # Get the prompts directory relative to this file
        prompts_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
            'prompts'
        )
        template_path = os.path.join(prompts_dir, 'volume_confirmation_analysis.txt')
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template = f.read()
                return template
        except FileNotFoundError:
            # Fallback template if file not found
            return self._get_fallback_template()
    
    def _get_fallback_template(self) -> str:
        """
        Fallback prompt template if the file can't be loaded.
        """
        return """You are a Volume Confirmation Specialist. Your sole purpose is to determine whether price movements are backed by legitimate volume support.

## Analysis Context:
{context}

## Your Specific Task:
Analyze the price-volume relationship chart to determine if current price movements have proper volume confirmation.

## Key Analysis Points:
- Volume confirmation during significant price moves
- Price-volume correlation strength and direction
- Trend continuation volume support

## Instructions:
1. Assess volume response to price movements
2. Evaluate price-volume correlation strength
3. Determine trend volume backing

## Required Output Format:

Output ONLY a valid JSON object. NO PROSE, NO EXPLANATIONS OUTSIDE THE JSON.

```json
{
  "volume_confirmation_status": "confirmed/diverging/neutral",
  "confirmation_strength": "strong/medium/weak",
  "price_volume_correlation": 0.0,
  "trend_volume_support": "strong/medium/weak/none",
  "confidence_score": 0-100,
  "key_insight": "Brief insight about volume-price relationship"
}
```

Focus only on price-volume confirmation. Ignore anomalies, institutional activity, or support/resistance levels."""
    
    def _build_context(self, analysis_data: Dict[str, Any], symbol: str) -> str:
        """
        Build context for the prompt using volume confirmation analysis data.
        
        This replaces the context engineering that was previously done by backend/gemini.
        
        Args:
            analysis_data: Processed analysis data from VolumeConfirmationProcessor
            symbol: Stock symbol
            
        Returns:
            Formatted context string
        """
        try:
            context_parts = [
                f"Stock Symbol: {symbol}",
                f"Analysis Timestamp: {datetime.now().isoformat()}",
                ""
            ]
            
            # Add overall assessment if available
            if 'overall_assessment' in analysis_data:
                assessment = analysis_data['overall_assessment']
                context_parts.extend([
                    "## Overall Volume Confirmation Assessment:",
                    f"Status: {assessment.get('confirmation_status', 'unknown')}",
                    f"Strength: {assessment.get('confirmation_strength', 'unknown')}",
                    f"Confidence Score: {assessment.get('confidence_score', 0)}%",
                    ""
                ])
            
            # Add price-volume correlation data
            if 'price_volume_correlation' in analysis_data:
                correlation = analysis_data['price_volume_correlation']
                context_parts.extend([
                    "## Price-Volume Correlation Analysis:",
                    f"Correlation Coefficient: {correlation.get('correlation_coefficient', 0.0):.3f}",
                    f"Correlation Strength: {correlation.get('correlation_strength', 'unknown')}",
                    f"Correlation Direction: {correlation.get('correlation_direction', 'unknown')}",
                    f"Trend: {correlation.get('correlation_trend', 'unknown')}",
                    ""
                ])
            
            # Add volume averages for context
            if 'volume_averages' in analysis_data:
                volume_data = analysis_data['volume_averages']
                context_parts.extend([
                    "## Volume Context:",
                    f"Current Volume: {volume_data.get('current_volume', 0):,}",
                    f"20-Day Average: {volume_data.get('volume_20d_avg', 0):,}",
                    f"Volume Ratio vs 20D: {volume_data.get('volume_vs_20d', 1.0):.2f}x",
                    ""
                ])
            
            # Add recent movements/confirmations
            if 'recent_movements' in analysis_data and analysis_data['recent_movements']:
                movements = analysis_data['recent_movements'][:3]  # Top 3 most recent
                context_parts.extend([
                    "## Recent Volume Confirmation Signals:",
                ])
                
                for i, movement in enumerate(movements, 1):
                    if isinstance(movement, dict) and 'date' in movement:
                        context_parts.extend([
                            f"Signal {i} ({movement.get('date', 'unknown')}):",
                            f"  Price Change: {movement.get('price_change_pct', 0):.2f}% {movement.get('price_move', 'unknown')}",
                            f"  Volume Response: {movement.get('volume_response', 'unknown')}",
                            f"  Volume Ratio: {movement.get('volume_ratio', 1.0):.2f}x",
                            f"  Significance: {movement.get('significance', 'unknown')}",
                        ])
                context_parts.append("")
            
            # Add trend support information
            if 'trend_support' in analysis_data:
                trend_data = analysis_data['trend_support']
                context_parts.extend([
                    "## Trend Support Analysis:",
                    f"Current Trend: {trend_data.get('current_trend', 'unknown')}",
                    f"Uptrend Volume Support: {trend_data.get('uptrend_volume_support', 'unknown')}",
                    f"Downtrend Volume Support: {trend_data.get('downtrend_volume_support', 'unknown')}",
                    f"Trend Consistency: {trend_data.get('trend_consistency', 0):.2%}",
                    ""
                ])
            
            # Add S/R band confirmations if available
            if 'sr_band_confirmations' in analysis_data and analysis_data['sr_band_confirmations']:
                confirmations = analysis_data['sr_band_confirmations']
                context_parts.extend([
                    "## Support/Resistance Band Confirmations:",
                ])
                for conf in confirmations[:3]:  # Top 3 confirmations
                    context_parts.extend([
                        f"Type: {conf.get('type', 'unknown')}",
                        f"Volume Ratio: {conf.get('volume_ratio', 1.0)}x",
                        f"Date: {conf.get('date', 'unknown')}",
                        ""
                    ])
            
            # Add data quality information
            context_parts.extend([
                "## Analysis Metadata:",
                f"Data Period: {analysis_data.get('data_period', 'unknown')}",
                f"Data Quality: {analysis_data.get('data_quality', 'unknown')}",
                f"Data Range: {analysis_data.get('data_range', 'unknown')}"
            ])
            
            return "\n".join(context_parts)
            
        except Exception as e:
            # Fallback context in case of errors
            return f"""Stock Symbol: {symbol}
Analysis Data: {json.dumps(analysis_data, indent=2, default=str)}

Error building structured context: {str(e)}
Please analyze the raw data above for volume confirmation patterns."""
    
    def _format_prompt(self, context: str) -> str:
        """
        Format the final prompt by inserting context into template.
        
        This replaces the PromptManager.format_prompt() functionality.
        
        Args:
            context: Built context string
            
        Returns:
            Complete formatted prompt
        """
        try:
            # Simple template substitution
            formatted_prompt = self.prompt_template.replace('{context}', context)
            
            # Add the solving line that was previously added by PromptManager
            if not formatted_prompt.endswith("Let me solve this by .."):
                formatted_prompt += "\n\nLet me solve this by .."
            
            return formatted_prompt
            
        except Exception as e:
            # Fallback to basic prompt if formatting fails
            return f"""You are a Volume Confirmation Specialist. Analyze this data:

{context}

Provide volume confirmation analysis in JSON format with these fields:
- volume_confirmation_status: confirmed/diverging/neutral
- confirmation_strength: strong/medium/weak
- price_volume_correlation: numeric value
- trend_volume_support: strong/medium/weak/none
- confidence_score: 0-100
- key_insight: brief insight

Let me solve this by .."""
    
    async def analyze_with_chart(self, 
                                analysis_data: Dict[str, Any], 
                                symbol: str, 
                                chart_image: bytes) -> str:
        """
        Perform volume confirmation analysis with chart image.
        
        This is the main method that replaces the previous Gemini backend workflow.
        
        Args:
            analysis_data: Processed volume confirmation data
            symbol: Stock symbol
            chart_image: Chart image bytes
            
        Returns:
            LLM analysis response
        """
        try:
            # Build context (replaces context engineering)
            context = self._build_context(analysis_data, symbol)
            
            # Format prompt (replaces prompt manager)
            formatted_prompt = self._format_prompt(context)
            
            # Log chart dimensions
            try:
                from PIL import Image
                import io as _io
                _w, _h = Image.open(_io.BytesIO(chart_image)).size
                print(f"[VOLUME_CONFIRMATION_LLM] Chart image dimensions: {_w}x{_h}px")
            except Exception:
                print(f"[VOLUME_CONFIRMATION_LLM] Chart image dimensions: unknown")
            
            # Make LLM call (replaces GeminiClient.analyze_volume_agent_specific)
            response = await self.llm_client.generate(
                prompt=formatted_prompt,
                images=[chart_image],
                enable_code_execution=True
            )
            
            return response
            
        except Exception as e:
            # Return structured error response
            error_response = {
                "error": f"Volume confirmation LLM analysis failed: {str(e)}",
                "agent": "volume_confirmation",
                "status": "failed",
                "fallback_used": True
            }
            return json.dumps(error_response)
    
    async def analyze_without_chart(self, 
                                   analysis_data: Dict[str, Any], 
                                   symbol: str) -> str:
        """
        Perform volume confirmation analysis without chart image (text-only).
        
        Args:
            analysis_data: Processed volume confirmation data  
            symbol: Stock symbol
            
        Returns:
            LLM analysis response
        """
        try:
            # Build context
            context = self._build_context(analysis_data, symbol)
            
            # Format prompt
            formatted_prompt = self._format_prompt(context)
            
            # Make text-only LLM call
            response = await self.llm_client.generate_text(
                prompt=formatted_prompt,
                enable_code_execution=True
            )
            
            return response
            
        except Exception as e:
            # Return structured error response
            error_response = {
                "error": f"Volume confirmation LLM analysis failed: {str(e)}",
                "agent": "volume_confirmation", 
                "status": "failed",
                "fallback_used": True
            }
            return json.dumps(error_response)
    
    def get_client_info(self) -> str:
        """Get information about the LLM client being used."""
        try:
            return self.llm_client.get_provider_info()
        except:
            return "Unknown LLM client"
    
    def get_provider_info(self) -> str:
        """Alias for get_client_info for compatibility."""
        return self.get_client_info()
    
    async def analyze_with_llm(self, stock_data, symbol, chart_image, context=""):
        """Compatibility method for volume orchestrator integration."""
        # First process the data using the processor
        from .processor import VolumeConfirmationProcessor
        processor = VolumeConfirmationProcessor()
        analysis_data = processor.process_volume_confirmation_data(stock_data)
        
        # Then perform LLM analysis
        if chart_image:
            llm_response = await self.analyze_with_chart(analysis_data, symbol, chart_image)
        else:
            llm_response = await self.analyze_without_chart(analysis_data, symbol)
            
        return {
            'success': True,
            'technical_analysis': analysis_data,
            'llm_analysis': llm_response,
            'chart_image': chart_image,
            'has_llm_analysis': True
        }


# Helper function for easy instantiation
def create_volume_confirmation_llm_agent(llm_client=None) -> VolumeConfirmationLLMAgent:
    """
    Create a Volume Confirmation LLM Agent instance.
    
    Args:
        llm_client: Optional LLMClient instance
        
    Returns:
        VolumeConfirmationLLMAgent instance
    """
    return VolumeConfirmationLLMAgent(llm_client=llm_client)


# Test function
async def test_volume_confirmation_llm_agent():
    """Test the Volume Confirmation LLM Agent."""
    print("🧪 Testing Volume Confirmation LLM Agent")
    print("=" * 50)
    
    try:
        # Create a mock LLM client for testing
        class MockLLMClient:
            def get_provider_info(self):
                return "mock:test-model"
            async def generate(self, prompt, images=None, **kwargs):
                return '{"volume_confirmation_status": "confirmed", "confidence_score": 75}'
            async def generate_text(self, prompt, **kwargs):
                return '{"volume_confirmation_status": "confirmed", "confidence_score": 75}'
        
        # Create agent with mock client
        agent = create_volume_confirmation_llm_agent(MockLLMClient())
        print(f"✅ Agent created using: {agent.get_client_info()}")
        
        # Test context building
        sample_data = {
            'overall_assessment': {
                'confirmation_status': 'volume_confirms_price',
                'confirmation_strength': 'medium',
                'confidence_score': 75
            },
            'price_volume_correlation': {
                'correlation_coefficient': 0.65,
                'correlation_strength': 'medium',
                'correlation_direction': 'positive',
                'correlation_trend': 'stable'
            },
            'volume_averages': {
                'current_volume': 2500000,
                'volume_20d_avg': 2000000,
                'volume_vs_20d': 1.25
            },
            'recent_movements': [
                {
                    'date': '2024-01-15',
                    'price_change_pct': 2.5,
                    'price_move': 'up',
                    'volume_response': 'confirming',
                    'volume_ratio': 1.8,
                    'significance': 'high'
                }
            ],
            'trend_support': {
                'current_trend': 'uptrend',
                'uptrend_volume_support': 'strong',
                'downtrend_volume_support': 'weak',
                'trend_consistency': 0.85
            },
            'data_period': '60 days',
            'data_quality': 'excellent',
            'data_range': '2023-11-15 to 2024-01-15'
        }
        
        context = agent._build_context(sample_data, "TESTSTOCK")
        print(f"✅ Context built: {len(context)} characters")
        
        # Test prompt formatting
        formatted_prompt = agent._format_prompt(context)
        print(f"✅ Prompt formatted: {len(formatted_prompt)} characters")
        print(f"   Contains solving line: {'Let me solve this by' in formatted_prompt}")
        
        # Show sample of formatted prompt
        print(f"\n📄 Sample prompt (first 300 chars):")
        print(formatted_prompt[:300] + "...")
        
        print(f"\n✅ Volume Confirmation LLM Agent test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_volume_confirmation_llm_agent())