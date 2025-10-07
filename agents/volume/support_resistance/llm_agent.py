#!/usr/bin/env python3
"""
Support/Resistance LLM Agent

Provides LLM-powered analysis for volume-based support and resistance levels.
Migrated from backend/gemini to backend/llm framework.
"""

import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from llm import get_llm_client
from .processor import SupportResistanceProcessor

logger = logging.getLogger(__name__)

class SupportResistanceLLMAgent:
    """
    LLM-powered analysis for support/resistance data using backend/llm framework.
    
    This agent handles all prompt processing internally, replacing the functionality
    that was previously provided by backend/gemini's context_engineer and prompt_manager.
    """
    
    def __init__(self):
        self.agent_name = "support_resistance_agent"
        self.processor = SupportResistanceProcessor()
        
        # Initialize LLM client using backend/llm framework
        try:
            self.llm_client = get_llm_client("support_resistance_agent")  # Use support_resistance_agent config
            logger.info(f"Support/Resistance LLM agent initialized with {self.llm_client.get_provider_info()}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.llm_client = None
    
    async def analyze_with_llm(self, 
                              stock_data,
                              symbol: str,
                              chart_image: Optional[bytes] = None,
                              context: str = "") -> Dict[str, Any]:
        """
        Perform comprehensive support/resistance analysis with LLM insights.
        
        Args:
            stock_data: DataFrame with OHLCV data
            symbol: Stock symbol
            chart_image: Optional chart image for multi-modal analysis
            context: Additional context for analysis
            
        Returns:
            Dict containing technical analysis and LLM insights
        """
        try:
            # Step 1: Perform technical analysis using processor
            technical_analysis = self.processor.process_support_resistance_data(stock_data)
            
            if 'error' in technical_analysis:
                return {
                    'success': False,
                    'error': technical_analysis['error'],
                    'technical_analysis': technical_analysis
                }
            
            # Step 2: Generate LLM analysis if client is available
            llm_analysis = None
            if self.llm_client:
                try:
                    # Build comprehensive prompt with our own prompt processing
                    prompt = self._build_comprehensive_prompt(
                        symbol, technical_analysis, context
                    )
                    
                    # Call LLM with or without image
                    if chart_image:
                        llm_analysis = await self.llm_client.generate(
                            prompt=prompt,
                            images=[chart_image],
                            enable_code_execution=True,
                            timeout=90
                        )
                    else:
                        llm_analysis = await self.llm_client.generate_text(
                            prompt=prompt,
                            enable_code_execution=True,
                            timeout=90
                        )
                        
                    logger.info(f"LLM analysis completed for {symbol} support/resistance")
                    
                except Exception as llm_error:
                    logger.warning(f"LLM analysis failed for {symbol}: {llm_error}")
                    llm_analysis = None
            
            # Step 3: Combine technical and LLM analysis
            return {
                'success': True,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'technical_analysis': technical_analysis,
                'llm_analysis': llm_analysis,
                'has_llm_analysis': llm_analysis is not None,
                'agent_info': {
                    'agent_name': self.agent_name,
                    'llm_provider': self.llm_client.get_provider_info() if self.llm_client else None
                }
            }
            
        except Exception as e:
            logger.error(f"Support/resistance analysis failed for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e),
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
    
    def _build_comprehensive_prompt(self, 
                                   symbol: str, 
                                   technical_analysis: Dict[str, Any], 
                                   context: str) -> str:
        """
        Build comprehensive prompt for LLM analysis.
        
        This replaces the functionality from backend/gemini's context_engineer and prompt_manager.
        All prompt processing is now handled within the agent itself.
        """
        
        # Extract key data for prompt context
        validated_levels = technical_analysis.get('validated_levels', [])
        current_position = technical_analysis.get('current_position_analysis', {})
        quality_assessment = technical_analysis.get('quality_assessment', {})
        trading_implications = technical_analysis.get('trading_implications', {})
        
        # Build structured prompt with intelligent context curation
        prompt_sections = []
        
        # Section 1: Analysis Context
        prompt_sections.append(f"""# Support/Resistance Analysis for {symbol}

## Analysis Context
- **Symbol**: {symbol}
- **Analysis Timestamp**: {datetime.now().isoformat()}
- **Analysis Type**: Volume-based support and resistance levels
- **Data Quality Score**: {quality_assessment.get('overall_score', 'N/A')}/100
""")
        
        # Section 2: Technical Analysis Summary
        support_levels = [level for level in validated_levels if level.get('type') in ['support', 'both']]
        resistance_levels = [level for level in validated_levels if level.get('type') in ['resistance', 'both']]
        
        prompt_sections.append(f"""## Technical Analysis Summary
- **Total Validated Levels**: {len(validated_levels)}
- **Support Levels**: {len(support_levels)}
- **Resistance Levels**: {len(resistance_levels)}
- **Current Price**: {current_position.get('current_price', 'N/A')}
- **Price Position**: {current_position.get('range_position_classification', 'Unknown')}
""")
        
        # Section 3: Key Support/Resistance Levels
        if support_levels or resistance_levels:
            prompt_sections.append("## Key Levels Analysis")
            
            if support_levels:
                prompt_sections.append("### Support Levels:")
                for i, level in enumerate(support_levels[:5], 1):  # Top 5 levels
                    distance = current_position.get('support_distance_percentage', 0)
                    prompt_sections.append(f"""**{i}. ${level.get('price', 'N/A'):.2f}**
   - Strength: {level.get('reliability', 'Unknown')}
   - Test Count: {level.get('total_tests', 0)}
   - Success Rate: {level.get('success_rate', 0):.1%}
   - Volume Confirmation: {level.get('volume_confirmation', 'Unknown')}""")
            
            if resistance_levels:
                prompt_sections.append("### Resistance Levels:")
                for i, level in enumerate(resistance_levels[:5], 1):  # Top 5 levels
                    distance = current_position.get('resistance_distance_percentage', 0)
                    prompt_sections.append(f"""**{i}. ${level.get('price', 'N/A'):.2f}**
   - Strength: {level.get('reliability', 'Unknown')}
   - Test Count: {level.get('total_tests', 0)}
   - Success Rate: {level.get('success_rate', 0):.1%}
   - Volume Confirmation: {level.get('volume_confirmation', 'Unknown')}""")
        
        # Section 4: Trading Implications
        if trading_implications:
            risk_reward = trading_implications.get('risk_reward_ratio', 'N/A')
            prompt_sections.append(f"""## Current Trading Setup
- **Risk/Reward Ratio**: {risk_reward}
- **Nearest Support**: ${current_position.get('nearest_support', {}).get('price', 'N/A')}
- **Nearest Resistance**: ${current_position.get('nearest_resistance', {}).get('price', 'N/A')}
- **Support Distance**: {current_position.get('support_distance_percentage', 0):.1f}%
- **Resistance Distance**: {current_position.get('resistance_distance_percentage', 0):.1f}%
""")
        
        # Section 5: Raw Data (for LLM to analyze)
        prompt_sections.append(f"""## Detailed Technical Data
```json
{json.dumps(technical_analysis, indent=2, default=str)}
```""")
        
        # Section 6: Additional Context
        if context:
            prompt_sections.append(f"""## Additional Context
{context}""")
        
        # Section 7: Analysis Instructions
        prompt_sections.append("""## Analysis Instructions

Please analyze this volume-based support and resistance data and provide comprehensive insights on:

### 1. Level Validation & Strength Assessment
- Evaluate the reliability and strength of identified levels
- Assess volume confirmation for each key level
- Identify any levels that may be false or weak

### 2. Current Price Position Analysis  
- Analyze the current price position relative to key levels
- Assess the significance of the current setup
- Identify potential trading opportunities or risks

### 3. Breakout/Breakdown Probability
- Evaluate the likelihood of price breaking key levels
- Assess volume patterns that might support breakouts
- Identify confluence factors that strengthen/weaken levels

### 4. Trading Strategy Recommendations
- Provide specific entry and exit levels
- Suggest stop-loss and target prices
- Assess risk/reward ratios for potential trades

### 5. Risk Assessment
- Identify key risks in the current setup
- Suggest risk management strategies
- Highlight any warning signals from the data

### 6. Key Insights Summary
- Provide 3-5 key takeaways from the analysis
- Highlight the most important levels to watch
- Give an overall assessment of the support/resistance structure

Please provide detailed, actionable analysis based on the technical data provided.""")
        
        # Combine all sections
        return "\n\n".join(prompt_sections)
    
    def extract_key_insights(self, analysis_result: Dict[str, Any]) -> List[str]:
        """
        Extract key insights from the analysis result.
        """
        insights = []
        
        if not analysis_result.get('success', False):
            return [f"Analysis failed: {analysis_result.get('error', 'Unknown error')}"]
        
        technical = analysis_result.get('technical_analysis', {})
        validated_levels = technical.get('validated_levels', [])
        current_position = technical.get('current_position_analysis', {})
        
        # Level count insight
        if len(validated_levels) >= 5:
            insights.append(f"Strong level structure with {len(validated_levels)} validated levels")
        elif len(validated_levels) >= 3:
            insights.append(f"Adequate level structure with {len(validated_levels)} validated levels")
        else:
            insights.append(f"Limited level structure with only {len(validated_levels)} validated levels")
        
        # Current position insight
        position = current_position.get('range_position_classification', 'unknown')
        if position == 'near_support':
            insights.append("Price near key support - potential bounce opportunity")
        elif position == 'near_resistance':
            insights.append("Price near key resistance - potential reversal risk")
        elif position == 'middle_range':
            insights.append("Price in middle range - direction unclear")
        
        # Quality assessment
        quality_score = technical.get('quality_assessment', {}).get('overall_score', 0)
        if quality_score >= 80:
            insights.append(f"High-quality analysis with {quality_score}/100 confidence")
        elif quality_score >= 60:
            insights.append(f"Good analysis quality with {quality_score}/100 confidence")
        else:
            insights.append(f"Moderate analysis quality - use with caution")
        
        # LLM analysis available
        if analysis_result.get('has_llm_analysis', False):
            insights.append("Enhanced with AI-powered pattern analysis")
        
        return insights
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            'agent_name': self.agent_name,
            'version': '2.0.0',
            'description': 'Volume-based support/resistance analysis with LLM insights',
            'llm_framework': 'backend/llm',
            'capabilities': [
                'Volume-based level identification',
                'Level strength assessment',
                'Trading setup analysis',
                'AI-powered pattern recognition',
                'Multi-modal chart analysis'
            ],
            'llm_provider': self.llm_client.get_provider_info() if self.llm_client else None
        }