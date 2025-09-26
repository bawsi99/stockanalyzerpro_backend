#!/usr/bin/env python3
"""
Final Decision Agent

Synthesizes all upstream analyses (indicators, charts, MTF, sector, risk, ML)
into a single final JSON via optimized_final_decision prompt.
"""

import json
import logging
import time
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional

from backend.gemini.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class FinalDecisionProcessor:
    """Agent that produces the final decision JSON."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.agent_name = "final_decision"
        self.client = GeminiClient(api_key=api_key)

    def _inject_context_blocks(
        self,
        knowledge_context: str,
        mtf_context: Optional[Dict[str, Any]],
        sector_bullets: Optional[str],
        risk_bullets: Optional[str],
    ) -> str:
        """
        Inject labeled JSON blocks and synthesis sections into knowledge_context so the
        final decision template and Gemini helpers can leverage them.
        """
        parts = [knowledge_context or ""]

        # Add MTF context as labeled JSON block for downstream extraction
        if mtf_context and isinstance(mtf_context, dict):
            try:
                parts.append("MultiTimeframeContext:\n" + json.dumps(mtf_context))
            except Exception:
                pass

        # Add Sector and Risk synthesis sections for human-readable context
        if sector_bullets and sector_bullets.strip():
            parts.append("SECTOR CONTEXT\n" + sector_bullets.strip())
        if risk_bullets and risk_bullets.strip():
            parts.append("RISK CONTEXT\n" + risk_bullets.strip())

        return "\n\n".join([p for p in parts if p])

    async def analyze_async(
        self,
        symbol: str,
        ind_json: Dict[str, Any],
        mtf_context: Optional[Dict[str, Any]] = None,
        sector_bullets: Optional[str] = None,
        risk_bullets: Optional[str] = None,
        chart_insights: str = "",
        knowledge_context: str = "",
    ) -> Dict[str, Any]:
        try:
            # Build knowledge context with injected blocks
            kc = self._inject_context_blocks(knowledge_context, mtf_context, sector_bullets, risk_bullets)

            # Add existing trading strategy to indicator JSON (for consistency rule)
            enhanced_ind_json = deepcopy(ind_json) if isinstance(ind_json, dict) else {}
            try:
                existing_strategy = self.client._extract_existing_trading_strategy(enhanced_ind_json)
                if existing_strategy:
                    enhanced_ind_json["existing_trading_strategy"] = existing_strategy
            except Exception:
                pass

            # Build comprehensive context for final decision
            context = self.client._build_comprehensive_context(enhanced_ind_json, chart_insights or "", kc)

            # Build the final decision prompt
            prompt = self.client.prompt_manager.format_prompt("optimized_final_decision", context=context)

            # Append ML guidance if present in knowledge context
            try:
                ml_block = self.client._extract_labeled_json_block(kc or "", label="MLSystemValidation:")
                ml_guidance_text = self.client._build_ml_guidance_text(ml_block) if ml_block else ""
                if ml_guidance_text:
                    prompt += "\n\n" + ml_guidance_text
            except Exception:
                pass

            # Call LLM with code execution and parse JSON strictly
            text_response, code_results, execution_results = await self.client.core.call_llm_with_code_execution(prompt)

            if not text_response or not str(text_response).strip():
                logger.warning("[FINAL_DECISION] Empty response; using fallback JSON")
                result = json.loads(self.client._create_fallback_json())
                result.setdefault('analysis_metadata', {})
                result['analysis_metadata']['fallback_reason'] = 'empty_final_decision_response'
            else:
                try:
                    result = json.loads(text_response.strip())
                except json.JSONDecodeError:
                    try:
                        _, json_blob = self.client.extract_markdown_and_json(text_response)
                        result = json.loads(json_blob)
                    except Exception:
                        logger.warning("[FINAL_DECISION] Unparsable response; using fallback JSON")
                        result = json.loads(self.client._create_fallback_json())
                        result.setdefault('analysis_metadata', {})
                        result['analysis_metadata']['fallback_reason'] = 'unparsable_final_decision_response'

            # Enhance with calculation metadata if available
            if code_results or execution_results:
                result = self.client._enhance_final_decision_with_calculations(result, code_results, execution_results)

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
                fallback = json.loads(self.client._create_fallback_json())
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
