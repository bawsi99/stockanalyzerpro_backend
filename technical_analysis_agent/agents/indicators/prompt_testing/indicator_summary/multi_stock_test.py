#!/usr/bin/env python3
"""
Indicator_summary prompt generator (new agent only)

Location: backend/agents/indicators/prompt testing/indicator_summary/multi_stock_test.py

What it does
- For each symbol provided, it:
  1) Retrieves stock data using the existing orchestrator
  2) Computes indicators deterministically (for agent inputs)
  3) Runs the new IndicatorAgentsOrchestrator and adapts its unified output to the curated shape
  4) Builds the prompt using the exact production template
  5) Writes the prompt and a summary to disk (no old/legacy comparison)

It does NOT call the LLM. It only builds the prompts that would be sent.

Usage
  python backend/agents/indicators/prompt\ testing/indicator_summary/multi_stock_test.py \
    --symbols RELIANCE,TCS,INFY --period 365 --interval day --exchange NSE

Notes
- Make sure environment/data access for retrieve_stock_data works in your setup.
- If you want to add custom static context lines, use --context.
"""

import os
import sys
import argparse
import re
from datetime import datetime
from typing import Tuple, Dict, Any, List

# Ensure backend/ is on sys.path so imports like `gemini.*` resolve
# File path is backend/agents/indicators/prompt testing/indicator_summary/multi_stock_test.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))  # up to backend/
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Imports from backend
from analysis.technical_indicators import TechnicalIndicators
from gemini.context_engineer import ContextEngineer, AnalysisType
from gemini.prompt_manager import PromptManager
from agents.indicators import indicators_orchestrator
from zerodha.client import ZerodhaDataClient


def normalize_prompt(text: str) -> str:
    """Remove dynamic values like timestamps and collapse whitespace for fair comparison."""
    if not isinstance(text, str):
        return ""
    s = text
    # Remove ISO datetimes
    s = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?", "<DATETIME>", s)
    # Remove integer epoch-like values
    s = re.sub(r"\b\d{10}(?:\d{3})?\b", "<EPOCH>", s)
    # Normalize multiple spaces and newlines
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n\n", s)
    return s.strip()


def curate_from_agents_unified(unified: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert IndicatorAgentsOrchestrator unified_analysis into the curated structure
    expected by ContextEngineer.structure_context(INDICATOR_SUMMARY).
    """
    indicator_summary = (unified or {}).get('indicator_summary', {})
    signal_consensus = (unified or {}).get('signal_consensus', {})

    key_indicators: Dict[str, Any] = {}

    # Map Trend block
    trend = indicator_summary.get('trend') if isinstance(indicator_summary, dict) else None
    if isinstance(trend, dict):
        key_indicators["trend_indicators"] = {
            "direction": trend.get('direction', 'neutral'),
            "strength": trend.get('strength', 'weak'),
            "confidence": trend.get('confidence', 0.0),
        }

    # Map Momentum block
    momentum = indicator_summary.get('momentum') if isinstance(indicator_summary, dict) else None
    if isinstance(momentum, dict):
        key_indicators.setdefault("momentum_indicators", {})
        key_indicators["momentum_indicators"].update({
            "rsi_status": momentum.get('rsi_signal', 'neutral'),
            "direction": momentum.get('direction', 'neutral'),
            "strength": momentum.get('strength', 'weak'),
            "confidence": momentum.get('confidence', 0.0),
        })

    # Conflicts
    detected_conflicts = {
        "has_conflicts": False,
        "conflict_count": 0,
        "conflict_list": []
    }
    if isinstance(signal_consensus, dict) and signal_consensus.get('consensus') == 'mixed':
        detected_conflicts.update({
            "has_conflicts": True,
            "conflict_count": 1,
            "conflict_list": ["Mixed consensus across indicator agents"],
        })

    curated = {
        "analysis_focus": "technical_indicators_summary",
        "key_indicators": key_indicators,
        "critical_levels": {},
        "conflict_analysis_needed": detected_conflicts["has_conflicts"],
        "detected_conflicts": detected_conflicts,
    }
    return curated


async def _fetch_stock_df(symbol: str, exchange: str, interval: str, period: int):
    z = ZerodhaDataClient()
    # Prefer async API if available
    if hasattr(z, 'get_historical_data_async'):
        return await z.get_historical_data_async(symbol=symbol, exchange=exchange, interval=interval, period=period)
    # Fallback to sync in executor
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, z.get_historical_data, symbol, exchange, interval, None, None, period)


def _curate_for_prompt_merge(unified: Dict[str, Any], indicators: Dict[str, Any], stock_df) -> Dict[str, Any]:
    indicator_summary = (unified or {}).get('indicator_summary', {})
    key_indicators: Dict[str, Any] = {}
    # Base blocks
    trend = indicator_summary.get('trend') if isinstance(indicator_summary, dict) else None
    if isinstance(trend, dict):
        key_indicators["trend_indicators"] = {
            "direction": trend.get('direction', 'neutral'),
            "strength": trend.get('strength', 'weak'),
            "confidence": trend.get('confidence', 0.0),
        }
    else:
        key_indicators["trend_indicators"] = {"direction": "neutral", "strength": "weak", "confidence": 0.0}
    momentum = indicator_summary.get('momentum') if isinstance(indicator_summary, dict) else None
    key_indicators.setdefault("momentum_indicators", {})
    if isinstance(momentum, dict):
        key_indicators["momentum_indicators"].update({
            "rsi_status": momentum.get('rsi_signal', 'neutral'),
            "direction": momentum.get('direction', 'neutral'),
            "strength": momentum.get('strength', 'weak'),
            "confidence": momentum.get('confidence', 0.0),
        })
    # Merge numerics from optimized indicators
    mov = (indicators or {}).get('moving_averages') or {}
    if isinstance(mov, dict):
        for k in ['sma_20','sma_50','sma_200','ema_20','ema_50','price_to_sma_200','sma_20_to_sma_50','golden_cross','death_cross']:
            v = mov.get(k)
            if v is not None:
                if isinstance(v, bool):
                    key_indicators['trend_indicators'][k] = v
                elif isinstance(v, (int, float)):
                    key_indicators['trend_indicators'][k] = float(v)
                else:
                    key_indicators['trend_indicators'][k] = v
    rsi = (indicators or {}).get('rsi')
    if isinstance(rsi, dict) and 'rsi_14' in rsi:
        try:
            key_indicators['momentum_indicators']['rsi_current'] = round(float(rsi['rsi_14']), 2)
        except Exception:
            pass
    macd = (indicators or {}).get('macd')
    if isinstance(macd, dict) and 'histogram' in macd:
        try:
            key_indicators['momentum_indicators']['macd'] = {
                'histogram': round(float(macd['histogram']), 2),
                'trend': key_indicators['momentum_indicators'].get('direction','neutral')
            }
        except Exception:
            pass
    # Volume
    vol = (indicators or {}).get('volume') or {}
    if isinstance(vol, dict) and 'volume_ratio' in vol:
        try:
            key_indicators['volume_indicators'] = {
                'volume_ratio': round(float(vol['volume_ratio']), 2),
                'volume_trend': 'neutral'
            }
        except Exception:
            pass
    # Levels via TechnicalIndicators
    critical_levels: Dict[str, Any] = {}
    try:
        if stock_df is not None and not getattr(stock_df, 'empty', True):
            support_levels, resistance_levels = TechnicalIndicators.detect_support_resistance(stock_df)
            if support_levels:
                sl = sorted(set(float(x) for x in support_levels), reverse=True)
                critical_levels['support'] = [round(v, 2) for v in sl[:3]]
            if resistance_levels:
                rl = sorted(set(float(x) for x in resistance_levels))
                critical_levels['resistance'] = [round(v, 2) for v in rl[:3]]
    except Exception:
        pass
    # Conflicts via ContextEngineer
    from gemini.context_engineer import ContextEngineer
    try:
        ce = ContextEngineer()
        detected_conflicts = ce._comprehensive_conflict_analysis({'trend_indicators': key_indicators.get('trend_indicators',{}), 'momentum_indicators': key_indicators.get('momentum_indicators',{}), 'volume_indicators': key_indicators.get('volume_indicators',{})})
    except Exception:
        detected_conflicts = {"has_conflicts": False, "conflict_count": 0, "conflict_list": []}
    return {
        'analysis_focus': 'technical_indicators_summary',
        'key_indicators': key_indicators,
        'critical_levels': critical_levels,
        'conflict_analysis_needed': detected_conflicts.get('has_conflicts', False),
        'detected_conflicts': detected_conflicts,
    }


def build_new_prompt(symbol: str, period: int, interval: str, exchange: str, extra_context: str = "") -> str:
    """Build the indicator_summary prompt (new agent only) for a symbol using the production template."""
    # Instances
    ctx_engineer = ContextEngineer()
    prompt_mgr = PromptManager()

    async def _run() -> str:
        df = await _fetch_stock_df(symbol=symbol, exchange=exchange, interval=interval, period=period)
        if df is None or getattr(df, 'empty', True):
            raise ValueError(f"No data for {symbol}")

        # Compute deterministic indicators
        indicators = TechnicalIndicators.calculate_all_indicators_optimized(df, symbol)

        # Timeframe label (match old style where possible)
        if period == 365 and interval in ("day", "daily"):
            timeframe = "1yr, daily"
        else:
            timeframe = f"{period} days, {interval}"

        # Optional additional knowledge context
        knowledge_context = extra_context or ""

        # NEW prompt: run indicator agents orchestrator, adapt to curated shape, then structure with same template
        agent_result = await indicators_orchestrator.analyze_indicators_comprehensive(
            symbol=symbol, stock_data=df, indicators=indicators, context=knowledge_context
        )
        # Merge numeric indicators and levels locally (minimal curation)
        curated_new = _curate_for_prompt_merge(agent_result.unified_analysis, indicators, df)
        context_new = ctx_engineer.structure_context(curated_new, AnalysisType.INDICATOR_SUMMARY, symbol, timeframe, knowledge_context)
        # Strict JSON-only output: avoid adding solving line that may elicit chain-of-thought
        prompt_new = prompt_mgr.format_prompt("optimized_indicators_summary", context=context_new)

        return prompt_new

    # Run the async part
    import asyncio
    return asyncio.run(_run())


def write_results(out_dir: str, symbol: str, new_prompt: str, new_norm: str, indicators: Dict[str, Any], timeframe_label: str, company: str | None = None, sector: str | None = None, expected_behavior: str | None = None) -> None:
    # Save the raw prompt only, named as prompt_<SYMBOL>_<timestamp>.txt in the specified directory
    os.makedirs(out_dir, exist_ok=True)
    sym_safe = re.sub(r"[^A-Za-z0-9_\-]+", "_", symbol)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_path = os.path.join(out_dir, f"prompt_{sym_safe}_{ts}.txt")

    # Also emit an old-style analysis wrapper file into the legacy results directory for like-to-like comparison
    legacy_results_dir = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../../prompt testing/indicator_summary/multi_stock_test_results"))
    os.makedirs(legacy_results_dir, exist_ok=True)
    analysis_path = os.path.join(legacy_results_dir, f"prompt_analysis_{sym_safe}_{ts}.txt")

    # Helper for nested gets
    def safe_get(data, *keys):
        try:
            cur = data
            for k in keys:
                cur = cur[k]
            return cur
        except Exception:
            return None

    # Build the KEY TECHNICAL INDICATORS SUMMARY, mirroring the old format
    current_price = (
        safe_get(indicators, 'current_price')
        or safe_get(indicators, 'daily_metrics', 'current_price')
        or safe_get(indicators, 'enhanced_volume', 'comprehensive_analysis', 'daily_metrics', 'current_price')
    )
    if current_price is None:
        # Fallback to last close if available
        closes = safe_get(indicators, 'close')
        if isinstance(closes, list) and closes:
            try:
                current_price = float(closes[-1])
            except Exception:
                current_price = None
    if current_price is None:
        current_price = 0

    macd_hist = safe_get(indicators, 'macd', 'histogram')
    macd_signal = None
    try:
        if macd_hist is not None:
            macd_signal = 'bullish' if float(macd_hist) > 0 else 'bearish'
    except Exception:
        macd_signal = None

    key_indicators = {
        "current_price": current_price,
        "rsi_14": safe_get(indicators, 'rsi', 'rsi_14'),
        "rsi_status": safe_get(indicators, 'rsi', 'status'),
        "macd_signal": macd_signal,
        "sma_20": safe_get(indicators, 'moving_averages', 'sma_20'),
        "sma_50": safe_get(indicators, 'moving_averages', 'sma_50'),
        "sma_200": safe_get(indicators, 'moving_averages', 'sma_200'),
        "price_vs_sma200_pct": safe_get(indicators, 'moving_averages', 'price_to_sma_200'),
        "volume_ratio": safe_get(indicators, 'volume', 'volume_ratio'),
        "death_cross": safe_get(indicators, 'moving_averages', 'death_cross'),
        "golden_cross": safe_get(indicators, 'moving_averages', 'golden_cross'),
        "enhanced_levels_available": 'enhanced_levels' in indicators,
        "support_levels_count": len(safe_get(indicators, 'enhanced_levels', 'dynamic_support') or []),
        "resistance_levels_count": len(safe_get(indicators, 'enhanced_levels', 'dynamic_resistance') or []),
    }

    # Build wrapper content once
    import json as _json
    header = []
    header.append("PROMPT ANALYSIS FOR LLM\n")
    header.append("=" * 80 + "\n\n")
    header.append(f"Stock Symbol: {symbol}\n")
    header.append(f"Company: {company or 'N/A'}\n")
    header.append(f"Sector: {sector or 'N/A'}\n")
    header.append(f"Expected Behavior: {expected_behavior or 'N/A'}\n")
    header.append(f"Generated: {datetime.now().isoformat()}\n")
    header.append(f"Prompt Length: {len(new_prompt)} characters\n")
    header.append(f"Context Length: {len(new_norm)} characters\n\n")
    header.append("KEY TECHNICAL INDICATORS SUMMARY:\n")
    header.append("-" * 40 + "\n")
    header.append(_json.dumps(key_indicators, indent=2, default=str) + "\n\n")
    header.append("FINAL PROMPT SENT TO LLM:\n")
    header.append("-" * 40 + "\n")
    wrapper_text = "".join(header) + new_prompt

    # Write wrapper to both the new prompt file and the legacy analysis file (like-to-like format)
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(wrapper_text)
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write(wrapper_text)


def main():
    ap = argparse.ArgumentParser(description="indicator_summary prompt parity tester (old vs new)")
    ap.add_argument("--symbols", required=True, help="Comma-separated list of symbols")
    ap.add_argument("--period", type=int, default=365)
    ap.add_argument("--interval", default="day", help="Internal backend interval (e.g., day, minute, 5minute, 60minute)")
    ap.add_argument("--exchange", default="NSE")
    ap.add_argument("--context", default="", help="Optional extra static context to include")
    # By default, save prompts in the same directory as this test file
    DEFAULT_OUT = CURRENT_DIR
    ap.add_argument("--out-dir", default=DEFAULT_OUT, help="Directory to write prompt files (defaults to this directory)")
    ap.add_argument("--call-llm", action="store_true", help="Optionally call LLM (not saved in prompt file)")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    print(f"Testing indicator_summary prompt parity for {len(symbols)} symbol(s): {symbols}")

    all_ok = True

    for sym in symbols:
        print(f"\n=== {sym} ===")
        try:
            new_prompt = build_new_prompt(
                symbol=sym,
                period=args.period,
                interval=args.interval,
                exchange=args.exchange,
                extra_context=args.context,
            )
            # Normalize
            new_norm = normalize_prompt(new_prompt)
            print(f"  New prompt length (normalized): {len(new_norm)}")

            # Also write an old-style analysis file for like-to-like comparison
            # Recompute indicators deterministically to feed the wrapper summary
            import asyncio as _asyncio
            df = _asyncio.run(_fetch_stock_df(symbol=sym, exchange=args.exchange, interval=args.interval, period=args.period))
            indicators = TechnicalIndicators.calculate_all_indicators_optimized(df, sym)
            timeframe_label = "1yr, daily" if args.period == 365 and args.interval in ("day", "daily") else f"{args.period} days, {args.interval}"
            # Populate company/sector/expected behavior for known symbols
            META = {
                "RELIANCE": {"company": "Reliance Industries", "sector": "Energy/Petrochemicals", "expected": "large_cap_stable"},
                "TCS": {"company": "Tata Consultancy Services", "sector": "IT Services", "expected": "large_cap_growth"},
                "HDFCBANK": {"company": "HDFC Bank", "sector": "Banking", "expected": "large_cap_stable"},
                "ICICIBANK": {"company": "ICICI Bank", "sector": "Banking", "expected": "large_cap_volatile"},
                "ITC": {"company": "ITC Limited", "sector": "FMCG/Tobacco", "expected": "large_cap_defensive"},
            }
            meta = META.get(sym.upper(), {})
            write_results(
                args.out_dir,
                sym,
                new_prompt,
                new_norm,
                indicators,
                timeframe_label,
                meta.get("company"),
                meta.get("sector"),
                meta.get("expected"),
            )

            # Optionally call LLM for a full response (if API key available)
            llm_text = None
            if args.call_llm:
                try:
                    # Use backend Gemini client to send the prompt with code execution enabled
                    from gemini.gemini_core import GeminiCore  # may not exist as public; fallback via client
                except Exception:
                    pass
                try:
                    import asyncio
                    from gemini.gemini_client import GeminiClient as _GC
                    gc = _GC()
                    text, code_results, exec_results = asyncio.run(gc.core.call_llm_with_code_execution(new_prompt))
                    llm_text = text or ""

                    # Write LLM response next to prompts (out dir)
                    out_dir = args.out_dir
                    os.makedirs(out_dir, exist_ok=True)
                    ts_resp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    resp_path = os.path.join(out_dir, f"response_{sym}_{ts_resp}.txt")
                    with open(resp_path, 'w', encoding='utf-8') as rf:
                        rf.write("LLM ANALYSIS RESULTS\n")
                        rf.write("=" * 80 + "\n\n")
                        rf.write(f"Stock Symbol: {sym}\n")
                        rf.write(f"Response Time: {datetime.now().isoformat()}\n")
                        rf.write(f"Response Length: {len(llm_text)} characters\n")
                        if code_results:
                            rf.write(f"Mathematical Calculations: {len(code_results)} code snippets executed\n")
                        if exec_results:
                            rf.write(f"Calculation Results: {len(exec_results)} computational outputs\n")
                        rf.write("\nCOMPLETE LLM RESPONSE:\n")
                        rf.write("-" * 40 + "\n")
                        rf.write(llm_text)
                        rf.write("\n")
                        
                    print(f"  Saved LLM response to: {resp_path}")
                except Exception as llm_ex:
                    print(f"  ⚠️ LLM call failed: {llm_ex}")
                    llm_text = None

        except Exception as e:
            all_ok = False
            print(f"  ERROR: {e}")

    print("\nRESULT:")
    print("  Completed generating new prompts.")


if __name__ == "__main__":
    main()
