#!/usr/bin/env python3
"""
Institutional Activity LLM Agent

Self-contained LLM integration for institutional trading activity analysis.
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

class InstitutionalActivityLLMAgent:
    """
    Self-contained LLM agent for institutional activity analysis.
    
    Handles all prompt processing internally, replacing orchestrator-based LLM calls.
    Uses backend/llm framework with agent-specific prompt engineering.
    """
    
    def __init__(self):
        self.agent_name = "institutional_activity_llm_agent"
        
        # Initialize LLM client using backend/llm
        try:
            self.llm_client = get_llm_client("institutional_activity_agent")
            logger.info(f"Institutional Activity LLM agent initialized with {self.llm_client.get_provider_info()}")
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
        Perform LLM analysis with chart image for institutional activity.
        
        Args:
            analysis_data: Technical analysis data from processor
            symbol: Stock symbol
            chart_image: Chart image bytes
            context: Additional context
            
        Returns:
            LLM analysis response string or None if failed
        """
        if not self.llm_client:
            logger.warning("LLM client not available for institutional activity analysis")
            return None
        
        try:
            # Build comprehensive prompt
            prompt = self._build_comprehensive_prompt(symbol, analysis_data, context)
            
            logger.info(f"Sending LLM request for {symbol} institutional activity analysis")
            
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
        Load the institutional activity prompt template.
        
        Returns template from file or fallback template.
        """
        try:
            # Load from agent's own directory
            agent_dir = os.path.dirname(__file__)
            template_path = os.path.join(agent_dir, 'institutional_activity_analysis.txt')
            
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
        """Fallback prompt template for institutional activity analysis."""
        return """You are an Institutional Activity Detection Specialist. Your sole purpose is to identify and analyze institutional trading patterns and smart money flow.

## Analysis Context:
{context}

## Your Specific Task:
Analyze the volume and price data to identify institutional trading activity, large block transactions, and smart money flow patterns.

## Key Analysis Points:
- Large block transaction detection and classification
- Institutional accumulation/distribution patterns
- Smart money flow identification
- Volume profile analysis for institutional footprints
- Pattern recognition for institutional behavior

## Instructions:
1. Identify large block transactions and institutional patterns
2. Analyze volume profile for institutional activity signatures  
3. Assess smart money flow direction and strength
4. Evaluate institutional accumulation vs distribution
5. Classify institutional activity level and confidence

## Required Output Format:

Output ONLY a valid JSON object. NO PROSE, NO EXPLANATIONS OUTSIDE THE JSON.

```json
{
  "institutional_activity_level": "high/medium/low/minimal",
  "primary_activity": "accumulation/distribution/balanced/unknown",
  "large_block_analysis": {
    "total_large_blocks": 0,
    "institutional_blocks": 0,
    "average_block_size": 0,
    "block_frequency": "high/medium/low"
  },
  "smart_money_flow": {
    "direction": "inflow/outflow/neutral",
    "strength": "strong/medium/weak",
    "confidence": 0-100
  },
  "volume_profile_insights": {
    "institutional_footprint": "strong/moderate/weak",
    "price_level_concentration": "high/medium/low",
    "volume_distribution": "institutional/retail/mixed"
  },
  "pattern_classification": {
    "accumulation_score": 0-100,
    "distribution_score": 0-100,
    "dominant_pattern": "accumulation/distribution/rotation/unknown"
  },
  "confidence_score": 0-100,
  "key_insight": "Brief insight about institutional activity patterns"
}
```

Focus only on institutional activity patterns. Ignore retail anomalies, support/resistance levels, or momentum analysis."""
    
    def _build_comprehensive_prompt(self, 
                                   symbol: str, 
                                   analysis_data: Dict[str, Any], 
                                   context: str) -> str:
        """
        Build comprehensive prompt for institutional activity analysis.
        
        This replaces the orchestrator's prompt building with agent-internal processing.
        """
        # Build structured context
        context_sections = []
        
        # Header
        context_sections.append(f"""Stock: {symbol}
Analysis Timestamp: {datetime.now().isoformat()}

INSTITUTIONAL ACTIVITY ANALYSIS DATA:""")
        
        # Key metrics summary
        activity_level = analysis_data.get('institutional_activity_level', 'unknown')
        primary_activity = analysis_data.get('primary_activity', 'unknown')
        large_block_analysis = analysis_data.get('large_block_analysis', {})
        volume_profile = analysis_data.get('volume_profile', {})
        
        context_sections.append(f"""
Key Metrics Summary:
- Activity Level: {activity_level}
- Primary Activity: {primary_activity}
- Large Block Transactions: {large_block_analysis.get('total_large_blocks', 0)}
- Institutional Blocks: {large_block_analysis.get('institutional_block_count', 0)}
- Block Detection Confidence: {large_block_analysis.get('detection_confidence', 0):.1f}%""")
        
        # Volume profile details
        if volume_profile and 'error' not in volume_profile:
            highest_vol = volume_profile.get('highest_volume_level', {})
            context_sections.append(f"""
Volume Profile Analysis:
- Highest Volume Node: ${highest_vol.get('price_level', 0):.2f}
- Volume Distribution: {volume_profile.get('volume_distribution', 'unknown')}
- Price Level Count: {len(volume_profile.get('volume_at_price', []))}""")
        
        # Large block details
        if large_block_analysis:
            context_sections.append(f"""
Large Block Transaction Analysis:
- Total Large Blocks: {large_block_analysis.get('total_large_blocks', 0)}
- Institutional Block Count: {large_block_analysis.get('institutional_block_count', 0)}
- Average Block Size: {large_block_analysis.get('average_block_size', 0):,.0f}
- Block Frequency Pattern: {large_block_analysis.get('block_frequency_pattern', 'unknown')}
- Detection Quality Score: {large_block_analysis.get('detection_quality_score', 0)}/100""")
        
        # Smart money flow indicators
        smart_money = analysis_data.get('smart_money_indicators', {})
        if smart_money:
            context_sections.append(f"""
Smart Money Flow Indicators:
- Flow Direction: {smart_money.get('flow_direction', 'unknown')}
- Flow Strength: {smart_money.get('flow_strength', 'unknown')}
- Volume Weighted Direction: {smart_money.get('volume_weighted_direction', 'unknown')}
- Institutional Signature Strength: {smart_money.get('institutional_signature_strength', 0)}/100""")
        
        # Pattern analysis
        pattern_analysis = analysis_data.get('pattern_analysis', {})
        if pattern_analysis:
            context_sections.append(f"""
Pattern Analysis:
- Accumulation Score: {pattern_analysis.get('accumulation_score', 0)}/100
- Distribution Score: {pattern_analysis.get('distribution_score', 0)}/100  
- Dominant Pattern: {pattern_analysis.get('dominant_pattern', 'unknown')}
- Pattern Confidence: {pattern_analysis.get('pattern_confidence', 0)}/100""")
        
        # Additional context
        if context:
            context_sections.append(f"""
Additional Context:
{context}""")
        
        # Raw data
        context_sections.append(f"""
Complete Analysis Data:
{json.dumps(analysis_data, indent=2, default=str)}

Please analyze this institutional activity data to identify institutional trading patterns and smart money flow.""")
        
        # Combine context and template
        full_context = "\n".join(context_sections)
        return self.prompt_template.replace('{context}', full_context)
    
    def get_client_info(self) -> str:
        """Get information about the LLM client being used."""
        if self.llm_client:
            return self.llm_client.get_provider_info()
        return "No LLM client initialized"