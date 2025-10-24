#!/usr/bin/env python3
"""
Volume Momentum LLM Agent

Self-contained LLM integration for volume momentum analysis.
Migrated from orchestrator-based to agent-internal LLM handling.
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from llm import get_llm_client

logger = logging.getLogger(__name__)

class VolumeMomentumLLMAgent:
    """
    Self-contained LLM agent for volume momentum analysis.
    
    Handles all prompt processing internally, replacing orchestrator-based LLM calls.
    Uses backend/llm framework with agent-specific prompt engineering.
    """
    
    def __init__(self):
        self.agent_name = "volume_momentum_llm_agent"
        
        # Initialize LLM client using backend/llm
        try:
            self.llm_client = get_llm_client("volume_momentum_agent")
            logger.info(f"Volume Momentum LLM agent initialized with {self.llm_client.get_provider_info()}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.llm_client = None
        
        # Load prompt template
        self.prompt_template = self._load_prompt_template()
    
    async def analyze_with_chart(self,
                                analysis_data: Dict[str, Any],
                                symbol: str,
                                chart_image: bytes,
                                context: str = "") -> Optional[str]:
        """
        Perform LLM analysis with chart image for volume momentum.
        
        Args:
            analysis_data: Technical analysis data from processor
            symbol: Stock symbol
            chart_image: Chart image bytes
            context: Additional context
            
        Returns:
            LLM analysis response string or None if failed
        """
        if not self.llm_client:
            logger.warning("LLM client not available for volume momentum analysis")
            return None
        
        try:
            # Build comprehensive prompt
            prompt = self._build_comprehensive_prompt(symbol, analysis_data, context)
            
            logger.info(f"Sending LLM request for {symbol} volume momentum analysis")
            
            # Make LLM call with image
            response = await self.llm_client.generate(
                prompt=prompt,
                images=[chart_image],
                enable_code_execution=True,
                timeout=90
            )
            
            logger.info(f"LLM analysis completed for {symbol}")
            return response
            
        except asyncio.TimeoutError:
            logger.error(f"LLM request timed out for {symbol}")
            return None
        except Exception as e:
            logger.error(f"LLM analysis failed for {symbol}: {e}")
            return None
    
    def _load_prompt_template(self) -> str:
        """
        Load the volume momentum prompt template.
        
        Returns template from file or fallback template.
        """
        try:
            # Load from agent's own directory
            agent_dir = os.path.dirname(__file__)
            template_path = os.path.join(agent_dir, 'volume_trend_momentum.txt')
            
            if os.path.exists(template_path):
                with open(template_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                logger.warning(f"Template file not found: {template_path}, using fallback")
                return self._get_fallback_template()
                
        except Exception as e:
            logger.warning(f"Failed to load template: {e}, using fallback")
            return self._get_fallback_template()
    
    def _get_fallback_template(self) -> str:
        """Fallback prompt template for volume momentum analysis."""
        return """You are a Volume Momentum Specialist. Your sole purpose is to assess volume momentum sustainability and acceleration for momentum trading decisions.

## Analysis Context:
{context}

## Your Specific Task:
Analyze volume momentum patterns to determine trend sustainability and acceleration.

## Key Analysis Points:
- Volume momentum direction and acceleration
- Momentum sustainability factors  
- Trend continuation probability
- Volume acceleration/deceleration signals
- Momentum exhaustion indicators

## Instructions:
1. Assess volume momentum strength and direction
2. Evaluate momentum sustainability indicators
3. Determine trend continuation probability
4. Identify acceleration/deceleration signals
5. Assess momentum exhaustion risk

## Required Output Format:

Output ONLY a valid JSON object. NO PROSE, NO EXPLANATIONS OUTSIDE THE JSON.

```json
{
  "momentum_direction": "increasing/decreasing/neutral",
  "momentum_strength": "strong/moderate/weak",
  "trend_sustainability": "high/medium/low",
  "acceleration_signal": "accelerating/decelerating/steady",
  "momentum_indicators": {
    "current_momentum_score": 0-100,
    "momentum_change_rate": 0.0,
    "sustainability_score": 0-100
  },
  "trend_continuation": {
    "continuation_probability": 0-100,
    "exhaustion_warning": true/false,
    "next_phase_prediction": "acceleration/deceleration/reversal/continuation"
  },
  "volume_patterns": {
    "volume_trend": "increasing/decreasing/stable",
    "acceleration_pattern": "consistent/volatile/weakening",
    "cycle_position": "early/mid/late"
  },
  "confidence_score": 0-100,
  "key_insight": "Brief actionable insight about volume momentum"
}
```

Focus exclusively on volume momentum patterns. Ignore price levels, institutional activity, or anomalies."""
    
    def _build_comprehensive_prompt(self, 
                                   symbol: str, 
                                   analysis_data: Dict[str, Any], 
                                   context: str) -> str:
        """
        Build comprehensive prompt for volume momentum analysis.
        
        This replaces the orchestrator's prompt building with agent-internal processing.
        """
        # Build structured context
        context_sections = []
        
        # Header
        context_sections.append(f"""Stock: {symbol}
Analysis Timestamp: {datetime.now().isoformat()}

VOLUME MOMENTUM ANALYSIS DATA:""")
        
        # Key metrics summary
        volume_trend = analysis_data.get('volume_trend', {})
        momentum_signals = analysis_data.get('momentum_signals', {})
        momentum_indicators = analysis_data.get('volume_momentum_indicators', {})
        
        context_sections.append(f"""
Key Metrics Summary:
- Volume Trend Direction: {volume_trend.get('trend_direction', 'unknown')}
- Volume Trend Strength: {volume_trend.get('trend_strength', 'unknown')}
- Momentum Signal: {momentum_signals.get('acceleration_signal', 'neutral')}
- Signal Strength: {momentum_signals.get('signal_strength', 'unknown')}
- Current Momentum Score: {momentum_indicators.get('current_momentum_score', 0):.1f}""")
        
        # Volume trend details
        if volume_trend:
            context_sections.append(f"""
Volume Trend Analysis:
- Primary Direction: {volume_trend.get('primary_trend_direction', 'unknown')}
- Strength Classification: {volume_trend.get('trend_strength_classification', 'unknown')}
- Trend Agreement Score: {volume_trend.get('trend_agreement_score', 0):.2f}
- Volume Change Rate: {volume_trend.get('volume_change_rate', 0):+.2f}%""")
        
        # Momentum analysis details
        if momentum_signals:
            context_sections.append(f"""
Momentum Signal Analysis:
- Acceleration Signal: {momentum_signals.get('acceleration_signal', 'neutral')}
- Signal Strength: {momentum_signals.get('signal_strength', 'unknown')}
- Momentum Direction: {momentum_signals.get('momentum_direction', 'unknown')}
- Signal Reliability: {momentum_signals.get('signal_reliability', 0):.0f}%""")
        
        # Volume momentum indicators
        if momentum_indicators:
            context_sections.append(f"""
Volume Momentum Indicators:
- Current Momentum Score: {momentum_indicators.get('current_momentum_score', 0):.1f}/100
- Momentum Change Rate: {momentum_indicators.get('momentum_change_rate', 0):+.3f}
- Momentum Acceleration: {momentum_indicators.get('momentum_acceleration', 'unknown')}
- Trend Alignment: {momentum_indicators.get('trend_alignment', 'unknown')}""")
        
        # Trend continuation analysis
        trend_continuation = analysis_data.get('trend_continuation', {})
        if trend_continuation:
            context_sections.append(f"""
Trend Continuation Analysis:
- Continuation Probability: {trend_continuation.get('continuation_probability', 0):.0f}%
- Exhaustion Warning: {trend_continuation.get('exhaustion_warning', False)}
- Next Phase Prediction: {trend_continuation.get('next_phase_prediction', 'unknown')}
- Sustainability Timeframe: {trend_continuation.get('sustainability_timeframe', 'unknown')}""")
        
        # Cycle analysis
        cycle_analysis = analysis_data.get('cycle_analysis', {})
        if cycle_analysis:
            context_sections.append(f"""
Momentum Cycle Analysis:
- Current Phase: {cycle_analysis.get('current_phase', 'unknown')}
- Cycle Position: {cycle_analysis.get('cycle_position', 'unknown')}
- Cycle Count: {cycle_analysis.get('cycle_count', 0)}
- Average Cycle Length: {cycle_analysis.get('average_cycle_length', 0):.1f} days
- Cycle Regularity: {cycle_analysis.get('cycle_regularity', 'unknown')}""")
        
        # Future implications
        future_implications = analysis_data.get('future_implications', {})
        if future_implications:
            context_sections.append(f"""
Future Implications:
- Trend Continuation Probability: {future_implications.get('trend_continuation_probability', 0):.1f}%
- Momentum Exhaustion Warning: {future_implications.get('momentum_exhaustion_warning', False)}
- Volume Acceleration Signal: {future_implications.get('volume_acceleration_signal', 'unknown')}
- Predicted Momentum Phase: {future_implications.get('predicted_momentum_phase', 'unknown')}
- Confidence Level: {future_implications.get('confidence_level', 'unknown')}""")
        
        # Sustainability assessment
        sustainability = analysis_data.get('sustainability_assessment', {})
        if sustainability:
            context_sections.append(f"""
Sustainability Assessment:
- Overall Sustainability: {sustainability.get('overall_sustainability', 'unknown')}
- Sustainability Score: {sustainability.get('sustainability_score', 0)}/100
- Timeframe: {sustainability.get('sustainability_timeframe', 'unknown')}
- Supporting Factors: {', '.join(sustainability.get('supporting_factors', [])[:3])}
- Risk Factors: {', '.join(sustainability.get('risk_factors', [])[:3])}""")
        
        # Additional context
        if context:
            context_sections.append(f"""
Additional Context:
{context}""")
        
        # Raw data
        context_sections.append(f"""
Complete Analysis Data:
{json.dumps(analysis_data, indent=2, default=str)}

Please analyze this volume momentum data to assess trend sustainability and momentum patterns.""")
        
        # Combine context and template
        full_context = "\n".join(context_sections)
        return self.prompt_template.replace('{context}', full_context)
    
    def get_client_info(self) -> str:
        """Get information about the LLM client being used."""
        if self.llm_client:
            return self.llm_client.get_provider_info()
        return "No LLM client initialized"