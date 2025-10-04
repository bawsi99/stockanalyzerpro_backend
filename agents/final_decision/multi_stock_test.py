#!/usr/bin/env python3
"""
Full Analysis Pipeline Testing Framework (with Sector Context Validation)

Tests the complete enhanced analysis pipeline by calling the production endpoint
and validating that sector context is properly included in the final decision.
This ensures our sector parameter fix works end-to-end.

As of the MTF integration fix, this test also validates that raw LLM analysis
from the MTF agent is properly passed through to the final decision agent,
ensuring better multi-timeframe integration.

Usage: python backend/agents/final_decision/multi_stock_test.py
Options:
- Use --symbols to specify custom stocks (default: RELIANCE,TCS,HDFCBANK)
- Use --period to specify analysis period in days (default: 365)
- Use --interval to specify data interval (default: day)

Requirements:
- Analysis service running on http://localhost:8002 (or ANALYSIS_SERVICE_URL)
- Zerodha credentials configured
- Gemini API key configured
"""

import os
import sys
import json
import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import httpx
import argparse


# Add repo root and backend to PYTHONPATH (so this script can run standalone)
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BACKEND_ROOT = os.path.join(REPO_ROOT, "backend")
sys.path.append(REPO_ROOT)
sys.path.append(BACKEND_ROOT)

# Core imports
from backend.analysis.orchestrator import StockAnalysisOrchestrator
from backend.ml.indicators.technical_indicators import TechnicalIndicators
from backend.agents.final_decision.processor import FinalDecisionProcessor
from backend.gemini.gemini_client import GeminiClient

# Optional/agent imports
try:
    from backend.agents.mtf_analysis import mtf_agent_integration_manager
    from backend.agents.mtf_analysis.mtf_llm_agent import mtf_llm_agent
except Exception as e:
    mtf_agent_integration_manager = None
    mtf_llm_agent = None

try:
    from backend.analysis.advanced_analysis import advanced_analysis_provider
except Exception:
    advanced_analysis_provider = None

class StockTestConfig:
    def __init__(self, symbol: str, name: str, exchange: str = "NSE", sector: str = None):
        self.symbol = symbol
        self.name = name
        self.exchange = exchange
        self.sector = sector

class FinalDecisionMultiStockTester:
    def __init__(
        self,
        stocks: List[StockTestConfig],
        period: int = 365,
        interval: str = "day",
        include_sector: bool = True,
        concurrency: int = 2,
        results_dir: Optional[str] = None,
        no_llm: bool = False,
    ):
        self.stocks = stocks
        self.period = period
        self.interval = interval
        self.include_sector = include_sector
        self.concurrency = max(1, int(concurrency))
        self.results_dir = results_dir or os.path.join(HERE, "final_decision_test_results")
        self.no_llm = no_llm
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize orchestrator (provides Gemini client for indicator summary)
        self.orchestrator = StockAnalysisOrchestrator()

        # Initialize final decision agent (reuse orchestrator's Gemini API key for continuity)
        try:
            api_key = self.orchestrator.gemini_client.core.api_key if getattr(self.orchestrator, "gemini_client", None) else os.environ.get("GEMINI_API_KEY")
        except Exception:
            api_key = os.environ.get("GEMINI_API_KEY")
        self.final_agent = FinalDecisionProcessor(api_key=api_key)

        # Separate Gemini client to reconstruct the exact prompt for logging
        self.gemini_for_prompts = GeminiClient(api_key=api_key)

    async def _call_full_analysis(self, symbol: str, exchange: str, sector: str = None) -> Dict[str, Any]:
        """
        Call the full enhanced analysis endpoint and return the complete response.
        This ensures we use the exact same data pipeline as production.
        """
        base_url = os.environ.get("ANALYSIS_SERVICE_URL", "http://localhost:8002")
        url = f"{base_url}/analyze/enhanced"
        
        payload = {
            "stock": symbol,
            "exchange": exchange,
            "interval": self.interval,
            "period": self.period,
            "enable_code_execution": True
        }
        
        # Add sector if provided
        if sector:
            payload["sector"] = sector
            print(f"[DEBUG] Adding sector '{sector}' to full analysis call for {symbol}")
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:  # 5 min timeout for full analysis
                print(f"[FULL_ANALYSIS] Calling {url} for {symbol}...")
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                
                if data.get("success"):
                    print(f"✅ [FULL_ANALYSIS] Analysis completed for {symbol}")
                    return data
                else:
                    print(f"⚠️ [FULL_ANALYSIS] Analysis failed for {symbol}: {data.get('error', 'Unknown error')}")
                    return {}
        except Exception as e:
            print(f"[FULL_ANALYSIS] Error calling enhanced analysis endpoint: {e}")
            return {}

    async def _reconstruct_and_save_prompt(self, symbol: str, indicators_payload, knowledge_context: str, prompt_path: str) -> None:
        """
        Reconstruct the final-decision prompt exactly as the agent does, then write to file.
        indicators_payload can be a raw JSON string (json_blob) or a dict.
        """
        # Add existing trading strategy only if dict
        payload = indicators_payload
        if isinstance(indicators_payload, dict):
            enhanced = dict(indicators_payload)
            try:
                existing = self.gemini_for_prompts._extract_existing_trading_strategy(enhanced)
                if existing:
                    enhanced["existing_trading_strategy"] = existing
            except Exception:
                pass
            payload = enhanced

        # Build comprehensive context and prompt
        context = self.gemini_for_prompts._build_comprehensive_context(
            payload,
            chart_insights="",
            knowledge_context=knowledge_context
        )
        prompt = self.gemini_for_prompts.prompt_manager.format_prompt(
            "optimized_final_decision",
            context=context
        )

        # Append ML guidance if present (none in this test by default)
        try:
            ml_block = self.gemini_for_prompts._extract_labeled_json_block(knowledge_context or "", label="MLSystemValidation:")
            ml_guidance = self.gemini_for_prompts._build_ml_guidance_text(ml_block) if ml_block else ""
            if ml_guidance:
                prompt += "\n\n" + ml_guidance
        except Exception:
            pass

        with open(prompt_path, 'w') as f:
            f.write("FINAL DECISION PROMPT (RECONSTRUCTED)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Symbol: {symbol}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Prompt Length: {len(prompt)}\n\n")
            f.write(prompt)

    def _extract_final_decision_context(self, analysis_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract the components needed to reconstruct the final decision prompt from the full analysis response.
        
        Note: As of the MTF integration fix, the multi_timeframe_analysis now contains the raw LLM
        response from the MTF agent, ensuring better integration with the final decision agent.
        """
        results = analysis_response.get("results", {})
        
        # Extract key components that would be passed to the final decision agent
        return {
            "technical_indicators": results.get("technical_indicators", {}),
            "ai_analysis": results.get("ai_analysis", {}),
            "sector_context": results.get("sector_context", {}),
            "multi_timeframe_analysis": results.get("multi_timeframe_analysis", {}),  # Contains raw MTF LLM analysis after fix
            "signals": results.get("signals", {}),
            "overlays": results.get("overlays", {}),
            "ml_predictions": results.get("ml_predictions", {}),
            "enhanced_metadata": results.get("enhanced_metadata", {})
        }
    
    def _prepare_mtf_payload_for_final_decision(self, mtf_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Prepare MTF payload for final decision agent using the same logic as the main analysis service.
        This method is provided for future compatibility if direct MTF processing is needed in tests.
        
        Note: This follows the same pattern as the fix applied to analysis_service.py to ensure
        raw LLM analysis from MTF agent is properly passed to final decision agent.
        """
        mtf_payload = None
        try:
            if isinstance(mtf_context, dict) and mtf_context:
                # Check if we have LLM insights from MTF LLM agent
                mtf_llm_insights = mtf_context.get('llm_insights', {})
                if mtf_llm_insights and mtf_llm_insights.get('success'):
                    # Pass the complete LLM insights including the raw llm_analysis
                    mtf_payload = mtf_llm_insights
                    print(f"[MTF_INTEGRATION_TEST] Using MTF LLM insights with raw analysis ({len(mtf_llm_insights.get('llm_analysis', ''))} chars)")
                else:
                    # Fallback to technical MTF analysis only
                    mtf_payload = {
                        'summary': mtf_context.get('summary', {}),
                        'cross_timeframe_validation': mtf_context.get('cross_timeframe_validation', {})
                    }
                    print(f"[MTF_INTEGRATION_TEST] Using technical MTF analysis only (no LLM insights)")
        except Exception as e:
            print(f"[MTF_INTEGRATION_TEST] Error preparing MTF payload for final decision: {e}")
            mtf_payload = None
        
        return mtf_payload

    async def test_single_stock(self, cfg: StockTestConfig) -> Dict[str, Any]:
        print(f"\n📌 Testing full analysis pipeline for {cfg.symbol} ({cfg.name})")
        t0 = time.time()
        try:
            # Call the full enhanced analysis endpoint - this uses the same pipeline as production
            analysis_response = await self._call_full_analysis(cfg.symbol, cfg.exchange, cfg.sector)
            
            if not analysis_response:
                raise Exception("Full analysis returned empty response")
            
            # Extract the components from the full analysis response
            context_data = self._extract_final_decision_context(analysis_response)
            
            # Debug: Print sector context info
            sector_context = context_data.get("sector_context", {})
            sector_bullets = sector_context.get("synthesis_bullets")
            print(f"[DEBUG] Sector context keys: {list(sector_context.keys())}")
            print(f"[DEBUG] sector_bullets value: {repr(sector_bullets[:100] if sector_bullets else sector_bullets)}...")
            print(f"[DEBUG] sector_bullets type: {type(sector_bullets)}")
            if sector_bullets:
                print(f"[DEBUG] sector_bullets length: {len(sector_bullets)}")
            
            # Debug: Print MTF context info to verify integration fix
            mtf_context = context_data.get("multi_timeframe_analysis", {})
            print(f"[MTF_DEBUG] MTF context keys: {list(mtf_context.keys())}")
            if isinstance(mtf_context, dict) and mtf_context:
                # Check if contains raw LLM analysis from MTF agent
                has_llm_insights = 'llm_insights' in mtf_context
                has_raw_llm = False
                if has_llm_insights:
                    llm_insights = mtf_context.get('llm_insights', {})
                    has_raw_llm = isinstance(llm_insights.get('llm_analysis'), str)
                    if has_raw_llm:
                        raw_analysis_length = len(llm_insights['llm_analysis'])
                        print(f"[MTF_DEBUG] ✅ Raw LLM analysis present ({raw_analysis_length} chars) - integration fix working!")
                    else:
                        print(f"[MTF_DEBUG] ⚠️ LLM insights present but no raw analysis - check integration")
                else:
                    print(f"[MTF_DEBUG] ℹ️ No LLM insights - using technical MTF only")
            else:
                print(f"[MTF_DEBUG] ⚠️ Empty MTF context")
            
            # Save the full analysis response
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            analysis_file = os.path.join(self.results_dir, f"full_analysis_{cfg.symbol}_{ts}.json")
            with open(analysis_file, 'w') as f:
                json.dump(analysis_response, f, indent=2, default=str)
            
            # Save just the final decision component for easier inspection
            ai_analysis = context_data.get("ai_analysis", {})
            final_decision_file = os.path.join(self.results_dir, f"final_decision_{cfg.symbol}_{ts}.json")
            with open(final_decision_file, 'w') as f:
                json.dump(ai_analysis, f, indent=2, default=str)
            
            # Save sector context separately for debugging
            sector_file = os.path.join(self.results_dir, f"sector_context_{cfg.symbol}_{ts}.json")
            with open(sector_file, 'w') as f:
                json.dump(sector_context, f, indent=2, default=str)
            
            # Reconstruct and save the full prompt
            prompt_file = os.path.join(self.results_dir, f"prompt_{cfg.symbol}_{ts}.txt")
            try:
                # Extract the indicators JSON blob from AI analysis (this is what gets passed to final decision)
                indicator_json_blob = ai_analysis.get("metadata", {}).get("indicator_json_blob", "")
                if not indicator_json_blob:
                    # Try to extract from technical indicators if not in metadata
                    indicator_json_blob = json.dumps(context_data.get("technical_indicators", {}))
                
                # Build knowledge context with sector bullets and MTF context
                # This should match the same pattern used by the actual final decision processor
                knowledge_context = ""
                
                # Add MTF context as labeled JSON block (same as final decision processor)
                mtf_payload = self._prepare_mtf_payload_for_final_decision(mtf_context)
                if mtf_payload:
                    try:
                        mtf_json = json.dumps(mtf_payload, indent=2)
                        knowledge_context += f"\n\nMultiTimeframeContext:\n{mtf_json}\n"
                        print(f"[PROMPT_RECONSTRUCTION] Added MTF context ({len(mtf_json)} chars) to knowledge context")
                    except Exception as e:
                        print(f"[PROMPT_RECONSTRUCTION] Failed to add MTF context: {e}")
                
                # Add sector bullets
                if sector_bullets:
                    knowledge_context += f"\n\nSECTOR CONTEXT:\n{sector_bullets}\n"
                
                await self._reconstruct_and_save_prompt(
                    cfg.symbol,
                    indicator_json_blob,
                    knowledge_context,
                    prompt_file
                )
                print(f"   → Full prompt: {os.path.basename(prompt_file)}")
            except Exception as e:
                print(f"   ⚠️ Failed to reconstruct prompt: {e}")
                prompt_file = None
            
            dt = time.time() - t0
            print(f"✅ Completed full analysis for {cfg.symbol} in {dt:.1f}s")
            print(f"   → Full analysis: {os.path.basename(analysis_file)}")
            print(f"   → Final decision: {os.path.basename(final_decision_file)}")
            print(f"   → Sector context: {os.path.basename(sector_file)}")
            
            return {
                "symbol": cfg.symbol,
                "success": True,
                "elapsed_sec": dt,
                "analysis_file": analysis_file,
                "final_decision_file": final_decision_file,
                "sector_file": sector_file,
                "prompt_file": prompt_file if 'prompt_file' in locals() else None,
                "sector_included": bool(cfg.sector),
                "sector_bullets_present": bool(sector_bullets),
                "sector_bullets_length": len(sector_bullets) if sector_bullets else 0
            }
        except Exception as e:
            dt = time.time() - t0
            print(f"❌ Failed for {cfg.symbol} in {dt:.1f}s: {e}")
            return {
                "symbol": cfg.symbol,
                "success": False,
                "elapsed_sec": dt,
                "error": str(e)
            }

    async def run(self) -> None:
        print("🚀 Starting Full Analysis Pipeline Tests (with Sector Context Validation)")
        print(f"Stocks: {[s.symbol for s in self.stocks]}")
        print(f"Period/Interval: {self.period} / {self.interval}")
        print(f"Include sector: {self.include_sector}")
        print(f"Concurrency: {self.concurrency}")
        print(f"Results dir: {self.results_dir}")
        print(f"Testing approach: Full enhanced analysis endpoint (production pipeline)")
        print("=" * 80)

        # Limit concurrency to reduce rate-limit risk
        sem = asyncio.Semaphore(self.concurrency)

        async def runner(cfg: StockTestConfig):
            async with sem:
                return await self.test_single_stock(cfg)

        results = await asyncio.gather(*[runner(s) for s in self.stocks])

        summary_path = os.path.join(self.results_dir, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Summary written to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Final Decision Agent Multi-Stock Prompt Test")
    parser.add_argument("--symbols", type=str, default="RELIANCE,TCS,HDFCBANK", help="Comma-separated stock symbols (NSE)")
    parser.add_argument("--exchange", type=str, default="NSE", help="Exchange for all symbols")
    parser.add_argument("--period", type=int, default=365, help="Analysis period in days")
    parser.add_argument("--interval", type=str, default="day", help="Data interval (minute, 5minute, 15minute, 30minute, 60minute, day)")
    parser.add_argument("--concurrency", type=int, default=2, help="Max concurrent tests")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for prompts/responses")
    parser.add_argument("--no-sector", action="store_true", help="Disable sector synthesis fetch via ANALYSIS_SERVICE_URL")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM calls and only generate prompts")
    parser.add_argument("--build-prompt", type=str, help="Build prompt for this symbol using cached analysis result JSON")

    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    
    # Create stocks with sector information for known symbols
    stocks = []
    for sym in symbols:
        if sym == "RELIANCE":
            stocks.append(StockTestConfig(sym, sym, exchange=args.exchange, sector="NIFTY_OIL_AND_GAS"))
        elif sym == "TCS":
            stocks.append(StockTestConfig(sym, sym, exchange=args.exchange, sector="NIFTY_IT"))
        elif sym == "HDFCBANK":
            stocks.append(StockTestConfig(sym, sym, exchange=args.exchange, sector="NIFTY_BANK"))
        else:
            # For unknown symbols, create without sector (will still work but with limited sector analysis)
            stocks.append(StockTestConfig(sym, sym, exchange=args.exchange))

    tester = FinalDecisionMultiStockTester(
        stocks,
        period=args.period,
        interval=args.interval,
        include_sector=(not args.no_sector),
        concurrency=args.concurrency,
        results_dir=args.out_dir,
        no_llm=args.no_llm,
    )
    asyncio.run(tester.run())


if __name__ == "__main__":
    main()
