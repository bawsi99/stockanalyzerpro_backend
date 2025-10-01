#!/usr/bin/env python3
"""
Sector Synthesis Agent

Generates 4 actionable sector bullets using the sector_synthesis_template via
Gemini, from either structured sector metrics or a sector knowledge_context.
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

from gemini.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class SectorSynthesisProcessor:
    """Agent that synthesizes sector context into 4 bullets."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.agent_name = "sector_synthesis"
        self.client = GeminiClient(api_key=api_key)

    def _build_sector_context(self, symbol: str, sector_data: Optional[Dict[str, Any]], base_context: str) -> str:
        """Build a sector knowledge_context block with SECTOR CONTEXT header and key lines.
        The Gemini client uses specific line prefixes to construct the synthesis prompt.
        """
        lines = []
        extras = []
        if sector_data:
            # Expected optional fields; we accept any and degrade gracefully
            so = sector_data.get("sector_outperformance_pct")
            mo = sector_data.get("market_outperformance_pct")
            sb = sector_data.get("sector_beta")
            mb = sector_data.get("market_beta")
            rot = sector_data.get("rotation_stage")
            rot_mom = sector_data.get("rotation_momentum")
            sector_name = sector_data.get("sector_name")
            
            # Additional metrics
            sc = sector_data.get("sector_correlation")
            mc = sector_data.get("market_correlation")
            ss = sector_data.get("sector_sharpe")
            ms = sector_data.get("market_sharpe")
            sv = sector_data.get("sector_volatility")
            mv = sector_data.get("market_volatility")
            sr = sector_data.get("sector_return")
            mr = sector_data.get("market_return")

            # Core metrics (used by GeminiClient for extraction)
            if so is not None:
                lines.append(f"- Sector Outperformance: {so}")
            if mo is not None:
                lines.append(f"- Market Outperformance: {mo}")
            if sb is not None:
                lines.append(f"- Sector Beta: {sb}")
            
            # Additional Context metrics
            if mb is not None:
                extras.append(f"- Market Beta: {mb}")
            if sector_name:
                extras.append(f"- Sector: {sector_name}")
            if rot:
                extras.append(f"- Rotation Stage: {rot}")
            if rot_mom is not None:
                extras.append(f"- Rotation Momentum: {rot_mom}")
            
            # Enhanced metrics
            if sc is not None:
                extras.append(f"- Sector Correlation: {sc}%")
            if mc is not None:
                extras.append(f"- Market Correlation: {mc}%")
            if ss is not None:
                extras.append(f"- Sector Sharpe: {ss}")
            if ms is not None:
                extras.append(f"- Market Sharpe: {ms}")
            if sv is not None:
                extras.append(f"- Sector Volatility: {sv}%")
            if mv is not None:
                extras.append(f"- Market Volatility: {mv}%")
            if sr is not None:
                extras.append(f"- Sector Return: {sr}%")
            if mr is not None:
                extras.append(f"- Market Return: {mr}%")

        # Build the final knowledge context
        header = [f"SECTOR CONTEXT for {symbol}".strip()]
        body = "\n".join(lines + extras)
        kc = "\n".join(["\n".join(header), body]).strip()

        # Merge with any base_context provided upstream
        if base_context:
            return f"{kc}\n\n{base_context}" if kc else base_context
        return kc

    async def analyze_async(self, symbol: str, sector_data: Optional[Dict[str, Any]] = None, knowledge_context: str = "") -> Dict[str, Any]:
        try:
            # Construct knowledge context with a SECTOR CONTEXT block if structured data provided
            sector_kc = self._build_sector_context(symbol, sector_data, knowledge_context)
            bullets = await self.client.synthesize_sector_summary(sector_kc)

            if not bullets or not bullets.strip():
                logger.warning("[SECTOR_SYNTHESIS] Empty response; using defaults")
                bullets = (
                    "• Relative performance vs sector unclear; monitor next 1-2 weeks\n"
                    "• Beta profile indeterminate; size positions prudently\n"
                    "• Sector rotation visibility low; wait for clearer momentum\n"
                    "• No strong sector tailwinds/headwinds identified"
                )

            return {
                "agent_name": self.agent_name,
                "symbol": symbol,
                "analysis_timestamp": datetime.now().isoformat(),
                "bullets": bullets,
                "context_block": sector_kc,
                "used_structured_metrics": bool(sector_data),
            }
        except Exception as e:  # noqa: E722
            logger.error(f"[SECTOR_SYNTHESIS] Failed: {e}")
            return {
                "agent_name": self.agent_name,
                "symbol": symbol,
                "analysis_timestamp": datetime.now().isoformat(),
                "error": str(e),
                "bullets": (
                    "• Sector summary unavailable; default to neutral stance\n"
                    "• Manage risk via conservative sizing\n"
                    "• Watch for sector leadership changes\n"
                    "• Reassess with updated sector metrics"
                ),
            }
