#!/usr/bin/env python3
"""
Sector Synthesis Agent

Generates 4 actionable sector bullets using the sector_synthesis_template via
backend/llm, from either structured sector metrics or a sector knowledge_context.

Migrated to use the new backend/llm system for provider-agnostic LLM calls.
"""

import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional

from llm import get_llm_client

logger = logging.getLogger(__name__)


class SectorSynthesisProcessor:
    """Agent that synthesizes sector context into 4 bullets."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.agent_name = "sector_synthesis"
        # Use new backend/llm system
        self.client = get_llm_client("sector_agent")
        
        # Load prompt template
        self.prompt_template = self._load_prompt_template()

    def _build_sector_context(self, symbol: str, sector_data: Optional[Dict[str, Any]], base_context: str) -> str:
        """Build a sector knowledge_context block with SECTOR CONTEXT header and key lines.
        This method formats sector data for processing by the LLM client.
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

            # Core metrics for LLM analysis
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
    
    def _load_prompt_template(self) -> str:
        """Load the sector synthesis prompt template from local file."""
        try:
            template_path = os.path.join(os.path.dirname(__file__), "sector_synthesis_template.txt")
            with open(template_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"Prompt template not found at {template_path}")
            # Fallback template
            return """You are a sector rotation analyst. Analyze the following sector context and create 4 actionable bullet points:

## Analysis Context:
{context}

## Instructions:
Return exactly 4 bullet points that cover:
• Performance Positioning
• Beta Analysis  
• Rotation Status
• Strategic Implication

Use format: • [insight]"""
    
    def _build_sector_analysis_prompt(self, knowledge_context: str) -> str:
        """Build the complete prompt for sector analysis using extracted metrics.
        
This method builds the complete prompt locally using the migrated context engineering logic.
        """
        try:
            ctx = knowledge_context or ""
            
            # Extract key metrics from knowledge_context using regex pattern matching
            def extract(pattern: str) -> str | None:
                m = re.search(pattern, ctx, re.IGNORECASE)
                return m.group(1).strip() if m else None

            sector_out = extract(r"-\s*Sector\s*Outperformance:\s*([+-]?[0-9]+(?:\.[0-9]+)?)")
            market_out = extract(r"-\s*Market\s*Outperformance:\s*([+-]?[0-9]+(?:\.[0-9]+)?)")
            sector_beta = extract(r"-\s*Sector\s*Beta:\s*([+-]?[0-9]+(?:\.[0-9]+)?)")
            market_beta = extract(r"-\s*Market\s*Beta:\s*([+-]?[0-9]+(?:\.[0-9]+)?)")
            rotation_stage = extract(r"-\s*Rotation\s*Stage:\s*([A-Za-z]+)")
            rotation_mom = extract(r"-\s*Rotation\s*Momentum:\s*([+-]?[0-9]+(?:\.[0-9]+)?)")
            sector_name = extract(r"-\s*Sector:\s*(.+)")
            
            # Extract additional metrics
            sector_corr = extract(r"-\s*Sector\s*Correlation:\s*([+-]?[0-9]+(?:\.[0-9]+)?)") 
            market_corr = extract(r"-\s*Market\s*Correlation:\s*([+-]?[0-9]+(?:\.[0-9]+)?)") 
            sector_sharpe = extract(r"-\s*Sector\s*Sharpe:\s*([+-]?[0-9]+(?:\.[0-9]+)?)") 
            market_sharpe = extract(r"-\s*Market\s*Sharpe:\s*([+-]?[0-9]+(?:\.[0-9]+)?)") 
            sector_vol = extract(r"-\s*Sector\s*Volatility:\s*([+-]?[0-9]+(?:\.[0-9]+)?)") 
            market_vol = extract(r"-\s*Market\s*Volatility:\s*([+-]?[0-9]+(?:\.[0-9]+)?)") 
            sector_ret = extract(r"-\s*Sector\s*Return:\s*([+-]?[0-9]+(?:\.[0-9]+)?)") 
            market_ret = extract(r"-\s*Market\s*Return:\s*([+-]?[0-9]+(?:\.[0-9]+)?)")

            # Build concise, explicit context with timeframes
            metrics_lines = []
            if sector_out is not None:
                metrics_lines.append(f"- Sector Outperformance (12m): {sector_out}%")
            if market_out is not None:
                metrics_lines.append(f"- Market Outperformance (12m): {market_out}%")
            if sector_beta is not None:
                metrics_lines.append(f"- Sector Beta (12m): {sector_beta}")
            if rotation_stage is not None:
                metrics_lines.append(f"- Rotation Stage (3m): {rotation_stage}")
            if rotation_mom is not None:
                metrics_lines.append(f"- Rotation Momentum (3m): {rotation_mom}%")

            additional_lines = []
            if sector_name:
                additional_lines.append(f"- Sector: {sector_name}")
            if market_beta is not None:
                additional_lines.append(f"- Market Beta (12m): {market_beta}")
            
            # Enhanced metrics in additional context
            if sector_corr is not None:
                additional_lines.append(f"- Sector Correlation: {sector_corr}%")
            if market_corr is not None:
                additional_lines.append(f"- Market Correlation: {market_corr}%")
            if sector_sharpe is not None:
                additional_lines.append(f"- Sector Sharpe: {sector_sharpe}")
            if market_sharpe is not None:
                additional_lines.append(f"- Market Sharpe: {market_sharpe}")
            if sector_vol is not None:
                additional_lines.append(f"- Sector Volatility: {sector_vol}%")
            if market_vol is not None:
                additional_lines.append(f"- Market Volatility: {market_vol}%")
            if sector_ret is not None:
                additional_lines.append(f"- Sector Return: {sector_ret}%")
            if market_ret is not None:
                additional_lines.append(f"- Market Return: {market_ret}%")

            # Build the structured context
            concise_context = f"""[Source: SectorContext]
Timeframes: Relative performance and beta = 12m; Rotation = 3m

Sector Metrics:
{chr(10).join(metrics_lines) if metrics_lines else 'N/A'}

Additional Context (if available):
{chr(10).join(additional_lines) if additional_lines else 'None'}"""

            # Format the complete prompt using the template
            prompt = self.prompt_template.format(context=concise_context)
            
            # Add solving line (equivalent to prompt_manager.SOLVING_LINE)
            prompt += "\n\nLet me solve this by analyzing the sector context and generating actionable insights..."
            
            return prompt
            
        except Exception as e:
            logger.error(f"Error building sector analysis prompt: {e}")
            # Fallback to simple context formatting
            fallback_context = knowledge_context or "No sector context provided"
            return self.prompt_template.format(context=fallback_context)

    async def analyze_async(self, symbol: str, sector_data: Optional[Dict[str, Any]] = None, knowledge_context: str = "") -> Dict[str, Any]:
        try:
            # Construct knowledge context with a SECTOR CONTEXT block if structured data provided
            sector_kc = self._build_sector_context(symbol, sector_data, knowledge_context)
            
            # Build the complete prompt with context engineering
            prompt = self._build_sector_analysis_prompt(sector_kc)
            
            # Use new backend/llm system for the LLM call
            bullets = await self.client.generate(prompt)

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
                "llm_provider": self.client.get_provider_info(),  # New: track which LLM was used
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
                "llm_provider": "error_fallback",
            }
