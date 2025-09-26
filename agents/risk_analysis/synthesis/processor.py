#!/usr/bin/env python3
"""
Risk Synthesis Agent

Lightweight wrapper that takes the structured risk digest (from the risk
orchestrator and agents) and produces exactly 5 clear bullets using the
risk_synthesis_template via the Gemini client.
"""

import logging
from datetime import datetime
from typing import Any, Dict

from gemini.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class RiskSynthesisProcessor:
    """Agent wrapper to produce 5-bullet risk synthesis via LLM."""

    def __init__(self, api_key: str | None = None) -> None:
        self.agent_name = "risk_synthesis"
        # Allow DI of api key (falls back to env in GeminiCore if None)
        self.client = GeminiClient(api_key=api_key)

    async def analyze_async(self, risk_digest: Dict[str, Any], context: str = "") -> Dict[str, Any]:
        """
        Create a human-consumable risk synthesis from a structured risk digest.

        Args:
            risk_digest: Combined risk output (e.g., from RiskAgentsOrchestrator)
            context: Optional additional context to include
        Returns:
            Dict with bullets, timestamp, and minimal metadata
        """
        try:
            # Use the same Gemini synthesis already implemented in the client
            bullets_text = await self.client.synthesize_risk_summary(risk_digest or {})

            if not bullets_text or not bullets_text.strip():
                logger.warning("[RISK_SYNTHESIS] Empty synthesis; returning placeholder")
                bullets_text = (
                    "• Unable to synthesize risk summary; using defaults\n"
                    "• Monitor market volatility and sudden moves (Medium impact)\n"
                    "• Watch liquidity conditions; size positions conservatively\n"
                    "• Check correlation breakdown risk during stress periods\n"
                    "• Use disciplined stops; reassess if regime changes"
                )

            return {
                "agent_name": self.agent_name,
                "analysis_timestamp": datetime.now().isoformat(),
                "bullets": bullets_text,
                "source": "risk_agents_digest",
                "context_included": bool(context),
            }
        except Exception as e:  # noqa: E722
            logger.error(f"[RISK_SYNTHESIS] Synthesis failed: {e}")
            return {
                "agent_name": self.agent_name,
                "analysis_timestamp": datetime.now().isoformat(),
                "error": str(e),
                "bullets": (
                    "• Risk synthesis error occurred; fall back to standard precautions\n"
                    "• Maintain conservative sizing due to uncertainty\n"
                    "• Monitor volatility and correlation clusters\n"
                    "• Focus on high-quality signals only\n"
                    "• Re-evaluate if drawdown exceeds thresholds"
                ),
            }
