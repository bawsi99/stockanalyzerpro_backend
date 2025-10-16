#!/usr/bin/env python3
"""
Final Decision Agent

Synthesizes all upstream analyses (indicators, charts, MTF, sector, risk, ML)
into a single final JSON via optimized_final_decision prompt.

MIGRATED TO USE backend/llm SYSTEM:
- Moved all prompt processing logic to prompt_processor.py
- Uses new LLM backend for clean API calls only
- No dependencies on Gemini backend for prompt management
"""

import json
import logging
import time
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional

# New LLM backend import
try:
    from llm import get_llm_client
except ImportError:
    # Fallback for development/testing
    def get_llm_client(*args, **kwargs):
        raise ImportError("backend.llm not available. Please ensure it's properly installed.")

# Internal prompt processor
from .prompt_processor import FinalDecisionPromptProcessor

logger = logging.getLogger(__name__)


class FinalDecisionProcessor:
    """Agent that produces the final decision JSON."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.agent_name = "final_decision"
        
        # Initialize new LLM client
        try:
            self.llm_client = get_llm_client(
                agent_name="final_decision_agent",
                # Pass API key if provided for compatibility
                **(dict(api_key=api_key) if api_key else {})
            )
        except Exception as e:
            logger.warning(f"Failed to initialize LLM client, falling back to direct client: {e}")
            # Direct fallback configuration
            self.llm_client = get_llm_client(
                provider="gemini",
                model="gemini-2.5-pro",
                timeout=90,
                max_retries=3,
                enable_code_execution=True,
                **(dict(api_key=api_key) if api_key else {})
            )
        
        # Initialize prompt processor
        self.prompt_processor = FinalDecisionPromptProcessor()

    def _inject_context_blocks(
        self,
        knowledge_context: str,
        mtf_context: Optional[Dict[str, Any]],
        sector_bullets: Optional[str],
        risk_bullets: Optional[str],
        pattern_insights: Optional[str] = None,
        advanced_digest: Optional[Dict[str, Any]] = None,
        volume_analysis: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Inject labeled JSON blocks and synthesis sections into knowledge_context.
        Now delegated to the prompt processor.
        """
        return self.prompt_processor.inject_context_blocks(
            knowledge_context=knowledge_context,
            mtf_context=mtf_context,
            sector_bullets=sector_bullets,
            risk_bullets=risk_bullets,
            pattern_insights=pattern_insights,
            advanced_digest=advanced_digest,
            volume_analysis=volume_analysis
        )

    async def analyze_async(
        self,
        symbol: str,
        ind_json: Dict[str, Any],
        mtf_context: Optional[Dict[str, Any]] = None,
        sector_bullets: Optional[str] = None,
        advanced_digest: Optional[Dict[str, Any]] = None,
        risk_bullets: Optional[str] = None,
        pattern_insights: Optional[str] = None,
        chart_insights: str = "",
        knowledge_context: str = "",
        volume_analysis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            # Build knowledge context with injected blocks
            kc = self._inject_context_blocks(knowledge_context, mtf_context, sector_bullets, risk_bullets, pattern_insights, advanced_digest, volume_analysis)

            # Add existing trading strategy to indicator JSON (for consistency rule) only when dict input is provided
            if isinstance(ind_json, dict):
                enhanced_ind_json = deepcopy(ind_json)
                try:
                    existing_strategy = self.prompt_processor.extract_existing_trading_strategy(enhanced_ind_json)
                    if existing_strategy:
                        enhanced_ind_json["existing_trading_strategy"] = existing_strategy
                except Exception:
                    pass
            else:
                # Try to parse JSON string to inject existing trading strategy for consistency
                try:
                    parsed = json.loads(ind_json)
                    if isinstance(parsed, dict):
                        try:
                            existing_strategy = self.prompt_processor.extract_existing_trading_strategy(parsed)
                        except Exception:
                            existing_strategy = {}
                        if existing_strategy:
                            parsed["existing_trading_strategy"] = existing_strategy
                        # Pass back as a compact JSON string for context inclusion
                        enhanced_ind_json = json.dumps(parsed)
                    else:
                        enhanced_ind_json = ind_json  # keep original if not dict
                except Exception:
                    enhanced_ind_json = ind_json  # pass through raw JSON blob string

            # Build comprehensive context for final decision
            context = self.prompt_processor.build_comprehensive_context(enhanced_ind_json, chart_insights or "", kc)

            # Build the final decision prompt
            prompt = self.prompt_processor.format_prompt("optimized_final_decision", context=context)

            # Append ML guidance if present in knowledge context
            try:
                ml_block = self.prompt_processor.extract_labeled_json_block(kc or "", label="MLSystemValidation:")
                ml_guidance_text = self.prompt_processor.build_ml_guidance_text(ml_block) if ml_block else ""
                if ml_guidance_text:
                    prompt += "\n\n" + ml_guidance_text
            except Exception:
                pass

            # Add solving line to the prompt
            prompt += self.prompt_processor.solving_line

            # Call LLM using new backend
            text_response = await self.llm_client.generate(
                prompt=prompt,
                enable_code_execution=True
            )
            
            # For compatibility, we'll simulate the old response format
            # The new LLM backend doesn't separate code results, so we set them as empty
            code_results = []
            execution_results = []

            if not text_response or not str(text_response).strip():
                logger.warning("[FINAL_DECISION] Empty response; using fallback JSON")
                result = json.loads(self.prompt_processor._create_fallback_json())
                result.setdefault('analysis_metadata', {})
                result['analysis_metadata']['fallback_reason'] = 'empty_final_decision_response'
            else:
                try:
                    result = json.loads(text_response.strip())
                except json.JSONDecodeError:
                    try:
                        _, json_blob = self.prompt_processor.extract_markdown_and_json(text_response)
                        result = json.loads(json_blob)
                    except Exception:
                        logger.warning("[FINAL_DECISION] Unparsable response; using fallback JSON")
                        result = json.loads(self.prompt_processor._create_fallback_json())
                        result.setdefault('analysis_metadata', {})
                        result['analysis_metadata']['fallback_reason'] = 'unparsable_final_decision_response'

            # Enhance with calculation metadata if available
            if code_results or execution_results:
                result = self.prompt_processor.enhance_final_decision_with_calculations(result, code_results, execution_results)

            # Ensure timestamp exists
            result.setdefault("timestamp", datetime.now().isoformat())

            return {
                "agent_name": self.agent_name,
                "symbol": symbol,
                "analysis_timestamp": datetime.now().isoformat(),
                "result": result,
            }
        except Exception as e:  # noqa: E722
            logger.error(f"[FINAL_DECISION] Failed: {e}")
            # Return minimal fallback wrapper
            try:
                fallback = json.loads(self.prompt_processor._create_fallback_json())
            except Exception:
                fallback = {
                    "trend": "neutral",
                    "confidence_pct": 50,
                    "short_term": {"signal": "hold"},
                    "medium_term": {"signal": "hold"},
                    "long_term": {"signal": "hold"},
                    "risks": ["error"],
                    "must_watch_levels": [],
                    "timestamp": datetime.now().isoformat(),
                }
            return {
                "agent_name": self.agent_name,
                "symbol": symbol,
                "analysis_timestamp": datetime.now().isoformat(),
                "error": str(e),
                "result": fallback,
            }
