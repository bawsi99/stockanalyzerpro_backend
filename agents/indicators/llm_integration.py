#!/usr/bin/env python3
"""
Indicator LLM Integration

Handles LLM integration specifically for indicator agents using the new backend/llm system.
Replaces the GeminiClient usage with the new LLM-agnostic approach.

Features:
- Uses backend/llm for LLM calls 
- Indicator-specific prompt management
- Indicator-specific context engineering
- Enhanced conflict detection and resolution
- Market regime awareness
"""

import json
import logging
import time
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

# Import the new LLM system
from backend.llm import get_llm_client

# Import indicator-specific components
from .prompt_manager import indicator_prompt_manager
from .context_engineer import indicator_context_engineer

logger = logging.getLogger(__name__)


class IndicatorLLMIntegration:
    """
    LLM integration specifically for indicator agents.
    
    Handles:
    - LLM calls using backend/llm system
    - Indicator-specific prompt formatting
    - Indicator-specific context engineering
    - Response parsing and validation
    - Error handling and fallbacks
    """
    
    def __init__(self):
        # Initialize LLM client for indicator agent
        self.llm_client = get_llm_client("indicator_agent")
        
        # Initialize indicator-specific components
        self.prompt_manager = indicator_prompt_manager
        self.context_engineer = indicator_context_engineer
        
        logger.info("IndicatorLLMIntegration initialized with backend/llm system")
    
    async def generate_indicator_summary(self,
                                       curated_data: Dict[str, Any],
                                       symbol: str,
                                       period: int,
                                       interval: str,
                                       knowledge_context: str = "",
                                       return_debug: bool = False) -> Tuple[str, Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Generate indicator summary using LLM with indicator-specific processing.
        
        Args:
            curated_data: Curated indicator data from agents
            symbol: Stock symbol
            period: Analysis period in days
            interval: Analysis interval
            knowledge_context: Additional context (MTF, sector, etc.)
            return_debug: Whether to return debug information
            
        Returns:
            Tuple of (markdown_summary, parsed_json, debug_info)
        """
        start_time = time.time()
        
        try:
            # Build timeframe string
            timeframe = f"{period} days, {interval}"
            
            # Enhance conflict detection using our context engineer
            enhanced_curated_data = self._enhance_conflict_detection(curated_data)
            
            # Build context using indicator-specific context engineer
            context = self.context_engineer.build_indicator_context(
                curated_data=enhanced_curated_data,
                symbol=symbol,
                timeframe=timeframe,
                knowledge_context=knowledge_context
            )
            
            # Format prompt using indicator-specific prompt manager
            prompt = self.prompt_manager.format_indicator_summary_prompt(context)
            
            logger.info(f"[INDICATOR_LLM] Generating summary for {symbol} with {len(context)} chars context")
            
            # Make LLM call using backend/llm system
            response = await self.llm_client.generate(
                prompt=prompt,
                enable_code_execution=True
            )
            
            processing_time = time.time() - start_time
            logger.info(f"[INDICATOR_LLM] Generated summary for {symbol} in {processing_time:.2f}s")
            
            if not response or not response.strip():
                logger.warning(f"[INDICATOR_LLM] Empty response for {symbol}, using fallback")
                fallback_result = self._create_fallback_response()
                
                if return_debug:
                    return "Analysis completed with fallback data due to empty response.", fallback_result, {
                        "raw_text": response or "",
                        "json_blob": json.dumps(fallback_result),
                        "processing_time": processing_time,
                        "context_length": len(context),
                        "prompt_length": len(prompt)
                    }
                return "Analysis completed with fallback data due to empty response.", fallback_result, None
            
            # Extract markdown and JSON from response
            markdown_part, json_blob = self._extract_markdown_and_json(response)
            
            # Parse JSON response
            try:
                parsed_result = json.loads(json_blob)
                
                # Validate and enhance the parsed result
                validated_result = self._validate_and_enhance_result(parsed_result)
                
                debug_info = None
                if return_debug:
                    debug_info = {
                        "raw_text": response,
                        "json_blob": json_blob,
                        "processing_time": processing_time,
                        "context_length": len(context),
                        "prompt_length": len(prompt),
                        "conflict_detected": enhanced_curated_data.get('detected_conflicts', {}).get('has_conflicts', False),
                        "conflict_count": enhanced_curated_data.get('detected_conflicts', {}).get('conflict_count', 0)
                    }
                
                return markdown_part, validated_result, debug_info
                
            except json.JSONDecodeError as e:
                logger.error(f"[INDICATOR_LLM] JSON parsing error for {symbol}: {e}")
                fallback_result = self._create_fallback_response()
                
                if return_debug:
                    return markdown_part, fallback_result, {
                        "raw_text": response,
                        "json_blob": json_blob,
                        "parsing_error": str(e),
                        "processing_time": processing_time
                    }
                return markdown_part, fallback_result, None
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"[INDICATOR_LLM] Analysis failed for {symbol}: {e}")
            
            # Return fallback response
            fallback_result = self._create_fallback_response()
            error_message = f"Analysis completed with fallback data due to error: {str(e)}"
            
            if return_debug:
                return error_message, fallback_result, {
                    "error": str(e),
                    "processing_time": processing_time
                }
            return error_message, fallback_result, None
    
    def _enhance_conflict_detection(self, curated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance conflict detection using indicator-specific context engineer.
        
        Args:
            curated_data: Original curated data
            
        Returns:
            Enhanced curated data with improved conflict analysis
        """
        try:
            enhanced_data = curated_data.copy()
            key_indicators = enhanced_data.get('key_indicators', {})
            
            # Use our context engineer for enhanced conflict detection
            enhanced_conflicts = self.context_engineer.detect_indicator_conflicts(key_indicators)
            
            # Replace the conflicts with our enhanced version
            enhanced_data['detected_conflicts'] = enhanced_conflicts
            
            logger.debug(f"[INDICATOR_LLM] Enhanced conflict detection: {enhanced_conflicts.get('conflict_count', 0)} conflicts, severity: {enhanced_conflicts.get('conflict_severity', 'none')}")
            
            return enhanced_data
            
        except Exception as e:
            logger.error(f"[INDICATOR_LLM] Error in conflict enhancement: {e}")
            return curated_data
    
    def _extract_markdown_and_json(self, llm_response: str) -> Tuple[str, str]:
        """
        Extract markdown summary and JSON code block from LLM response.
        
        Args:
            llm_response: Raw LLM response
            
        Returns:
            Tuple of (markdown_part, json_blob)
        """
        import re
        
        try:
            # Look for JSON code block
            match = re.search(r"```json\s*(\{[\s\S]+?\})\s*```", llm_response)
            if match:
                json_blob = match.group(1)
                markdown_part = llm_response[:match.start()].strip()
                
                logger.debug(f"[INDICATOR_LLM] Found JSON block: {len(json_blob)} chars")
                return markdown_part, json_blob
            else:
                # Try to find any JSON object in the output
                any_match = re.search(r"(\{[\s\S]+\})", llm_response)
                if any_match:
                    json_blob = any_match.group(1)
                    markdown_part = llm_response.replace(json_blob, "").strip()
                    
                    # Validate it's actually JSON
                    try:
                        json.loads(json_blob)
                        logger.debug(f"[INDICATOR_LLM] Found JSON object: {len(json_blob)} chars")
                        return markdown_part, json_blob
                    except json.JSONDecodeError:
                        pass
                
                logger.warning("[INDICATOR_LLM] No JSON found in response, using fallback")
                fallback_json = json.dumps(self._create_fallback_response())
                return llm_response, fallback_json
                
        except Exception as e:
            logger.error(f"[INDICATOR_LLM] Error extracting JSON: {e}")
            fallback_json = json.dumps(self._create_fallback_response())
            return llm_response, fallback_json
    
    def _validate_and_enhance_result(self, parsed_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and enhance the parsed LLM result.
        
        Args:
            parsed_result: Parsed JSON from LLM
            
        Returns:
            Validated and enhanced result
        """
        try:
            # Ensure required fields exist with defaults
            validated = {
                "trend_analysis": parsed_result.get("trend_analysis", {
                    "direction": "neutral",
                    "strength": "weak",
                    "confidence": 50,
                    "key_signals": []
                }),
                "momentum": parsed_result.get("momentum", {
                    "rsi_signal": "neutral",
                    "macd_signal": "neutral",
                    "momentum_strength": "weak"
                }),
                "critical_levels": parsed_result.get("critical_levels", {
                    "nearest_support": 0.0,
                    "nearest_resistance": 0.0,
                    "key_levels": []
                }),
                "trading_recommendation": parsed_result.get("trading_recommendation", {
                    "bias": "neutral",
                    "confidence": 50,
                    "entry_price": 0.0,
                    "target_price": 0.0,
                    "stop_loss": 0.0,
                    "rationale": "Analysis completed with default recommendation"
                }),
                "conflicts_resolved": parsed_result.get("conflicts_resolved", "No conflicts detected"),
                "confidence_score": parsed_result.get("confidence_score", 50)
            }
            
            # Add metadata about the analysis
            validated["analysis_metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "agent": "indicator_agent",
                "llm_system": "backend_llm",
                "context_engineer": "indicator_specific"
            }
            
            return validated
            
        except Exception as e:
            logger.error(f"[INDICATOR_LLM] Error validating result: {e}")
            return self._create_fallback_response()
    
    def _create_fallback_response(self) -> Dict[str, Any]:
        """Create a safe fallback response when LLM analysis fails."""
        return {
            "trend_analysis": {
                "direction": "neutral",
                "strength": "weak",
                "confidence": 50,
                "key_signals": ["Fallback analysis - insufficient data"]
            },
            "momentum": {
                "rsi_signal": "neutral",
                "macd_signal": "neutral",
                "momentum_strength": "weak"
            },
            "critical_levels": {
                "nearest_support": 0.0,
                "nearest_resistance": 0.0,
                "key_levels": []
            },
            "trading_recommendation": {
                "bias": "neutral",
                "confidence": 50,
                "entry_price": 0.0,
                "target_price": 0.0,
                "stop_loss": 0.0,
                "rationale": "Fallback recommendation due to analysis error"
            },
            "conflicts_resolved": "Analysis fallback - no conflicts assessed",
            "confidence_score": 50,
            "analysis_metadata": {
                "generated_at": datetime.now().isoformat(),
                "agent": "indicator_agent",
                "llm_system": "backend_llm",
                "fallback_used": True,
                "fallback_reason": "LLM analysis failed or returned empty"
            }
        }
    
    def get_llm_info(self) -> Dict[str, Any]:
        """Get information about the LLM configuration being used."""
        try:
            return {
                "provider": self.llm_client.get_provider_info(),
                "agent_config": self.llm_client.get_config(),
                "context_engineer": "indicator_specific",
                "prompt_manager": "indicator_specific"
            }
        except Exception as e:
            return {
                "error": f"Could not get LLM info: {e}",
                "agent_config": "unknown"
            }


# Global instance for indicators - using lazy initialization to avoid import-time API key issues
_indicator_llm_integration_instance = None

def get_indicator_llm_integration():
    """Get or create the global indicator LLM integration instance."""
    global _indicator_llm_integration_instance
    if _indicator_llm_integration_instance is None:
        _indicator_llm_integration_instance = IndicatorLLMIntegration()
    return _indicator_llm_integration_instance

# For backwards compatibility - this will be a function call now
indicator_llm_integration = get_indicator_llm_integration
