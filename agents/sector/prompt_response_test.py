#!/usr/bin/env python3
"""
Sector Synthesis Prompt/Response Test

Creates and saves the exact prompt used for sector synthesis (as built in the
current decision-making pipeline), and optionally calls the LLM to store the
response. Mirrors the construction logic in GeminiClient.synthesize_sector_summary.

Usage:
  python backend/agents/sector/prompt_response_test.py

Behavior:
- Builds SECTOR CONTEXT using SectorSynthesisProcessor's context builder
- Constructs the exact prompt string identical to GeminiClient.synthesize_sector_summary
- Saves the final prompt to disk
- If GEMINI_API_KEY is available, calls the LLM and saves the response to disk
"""

import os
import sys
import asyncio
import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

# Resolve paths to import project modules, similar to volume agent test
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..', '..')
backend_path = os.path.join(project_root, 'backend')

# Ensure paths are in Python path
sys.path.insert(0, project_root)
sys.path.insert(0, backend_path)

# Explicitly load backend/config/.env so env vars are available to this tester
try:
    import dotenv  # type: ignore
    env_path = os.path.join(backend_path, 'config', '.env')
    if os.path.exists(env_path):
        dotenv.load_dotenv(dotenv_path=env_path)
        print(f"📁 Loaded .env for sector tester from: {env_path}")
except Exception as _dotenv_ex:  # python-dotenv might not be installed; APIKeyManager also attempts to load
    print(f"ℹ️ dotenv load skipped: {_dotenv_ex}")

try:
    # Import refactored sector agent and Gemini utilities
    from agents.sector import SectorSynthesisProcessor, SectorBenchmarkingProvider
    from agents.sector.classifier import SectorClassifier
    from gemini.gemini_client import GeminiClient
    from gemini.prompt_manager import PromptManager
    from zerodha.client import ZerodhaDataClient
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(f"Script dir: {script_dir}")
    print(f"Project root: {project_root}")
    print(f"Backend path: {backend_path}")
    print("Python path (first 10):")
    for p in sys.path[:10]:
        print(f"  {p}")
    sys.exit(1)


@dataclass
class SectorTestConfig:
    symbol: str
    name: str
    sector_name: Optional[str]
    sector_outperformance_pct: Optional[float]
    market_outperformance_pct: Optional[float]
    sector_beta: Optional[float]
    market_beta: Optional[float]
    rotation_stage: Optional[str]
    rotation_momentum: Optional[float]

    def to_sector_data(self) -> Dict[str, Any]:
        return {
            "sector_name": self.sector_name,
            "sector_outperformance_pct": self.sector_outperformance_pct,
            "market_outperformance_pct": self.market_outperformance_pct,
            "sector_beta": self.sector_beta,
            "market_beta": self.market_beta,
            "rotation_stage": self.rotation_stage,
            "rotation_momentum": self.rotation_momentum,
        }


class SectorPromptResponseTester:
    def __init__(self, symbols: List[str], sectors: List[str], output_dir: Optional[str], no_llm: bool, period: int, exchange: str, metrics_json: Optional[str], concurrency: int):
        self.prompt_manager = PromptManager()
        # Detect API key upfront unless --no-llm. Only use GEMINI_API_KEY from .env (no rotation).
        self._api_key = None if no_llm else os.environ.get("GEMINI_API_KEY")
        self.client = GeminiClient(api_key=self._api_key) if (self._api_key and not no_llm) else None
        # Resolve results directory robustly to avoid duplicated "backend/agents/sector" segments
        # Priority:
        # 1) Absolute path -> use as-is
        # 2) Relative path starting with "backend/agents/sector" -> resolve from project_root
        # 3) Other relative paths -> resolve from this script's directory
        if output_dir:
            if os.path.isabs(output_dir):
                self.results_dir = output_dir
            else:
                normalized = os.path.normpath(output_dir)
                if normalized.startswith("backend/agents/sector"):
                    self.results_dir = os.path.normpath(os.path.join(project_root, normalized))
                else:
                    self.results_dir = os.path.normpath(os.path.join(script_dir, normalized))
        else:
            self.results_dir = os.path.join(script_dir, "sector_synthesis_test_results")
        os.makedirs(self.results_dir, exist_ok=True)

        self.period = period
        self.exchange = exchange
        self.concurrency = max(1, concurrency)

        # Initialize providers/clients
        self.benchmark_provider = SectorBenchmarkingProvider()
        self.sector_classifier = SectorClassifier()
        self.zerodha_client = ZerodhaDataClient()
        # Rotation cache
        self.rotation_map: Dict[str, Dict[str, Any]] = {}

        # Resolve symbols list; if sectors provided, add first symbol from each sector
        self.symbols = list(symbols)
        for sec in sectors:
            try:
                stocks = self.sector_classifier.get_sector_stocks(sec)
                if stocks:
                    first = stocks[0]
                    if first not in self.symbols:
                        self.symbols.append(first)
            except Exception:
                continue

        # Optional metrics override JSON
        self.metrics_override: Dict[str, Dict[str, Any]] = {}
        if metrics_json:
            try:
                with open(metrics_json, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.metrics_override = data
            except Exception as e:
                print(f"⚠️  Failed to load metrics override JSON: {e}")

        # Pre-constructed configs list not used; we build per symbol from real metrics
        self.tests: List[SectorTestConfig] = []

    @staticmethod
    def _build_exact_sector_prompt(prompt_manager: PromptManager, knowledge_context: str) -> str:
        """Recreate (concise) prompt assembly used by synthesize_sector_summary: explicit timeframes, rotation elevated."""
        import re
        ctx = knowledge_context or ""

        def extract(pattern: str):
            m = re.search(pattern, ctx, re.IGNORECASE)
            return m.group(1).strip() if m else None

        sector_out = extract(r"-\s*Sector\s*Outperformance:\s*([+-]?[0-9]+(?:\.[0-9]+)?)")
        market_out = extract(r"-\s*Market\s*Outperformance:\s*([+-]?[0-9]+(?:\.[0-9]+)?)")
        sector_beta = extract(r"-\s*Sector\s*Beta:\s*([+-]?[0-9]+(?:\.[0-9]+)?)")
        market_beta = extract(r"-\s*Market\s*Beta:\s*([+-]?[0-9]+(?:\.[0-9]+)?)")
        rotation_stage = extract(r"-\s*Rotation\s*Stage:\s*([A-Za-z]+)")
        rotation_mom = extract(r"-\s*Rotation\s*Momentum:\s*([+-]?[0-9]+(?:\.[0-9]+)?)")
        sector_name = extract(r"-\s*Sector:\s*(.+)")

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

        context = f"""
[Source: SectorContext]
Timeframes: Relative performance and beta = 12m; Rotation = 3m

Sector Metrics:
{chr(10).join(metrics_lines) if metrics_lines else 'N/A'}

Additional Context (if available):
{chr(10).join(additional_lines) if additional_lines else 'None'}
"""
        prompt = prompt_manager.format_prompt("sector_synthesis_template", context=context)
        prompt += prompt_manager.SOLVING_LINE
        return prompt

    def _write_prompt_file(self, symbol: str, prompt: str, knowledge_context: str) -> str:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.results_dir, f"sector_prompt_{symbol}_{ts}.txt")
        with open(path, 'w') as f:
            f.write("SECTOR SYNTHESIS PROMPT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Symbol: {symbol}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Prompt Length: {len(prompt)} characters\n\n")
            f.write("SECTOR KNOWLEDGE CONTEXT (Exact Block):\n")
            f.write("-" * 40 + "\n")
            f.write((knowledge_context or "").strip() + "\n\n")
            f.write("FINAL PROMPT SENT TO LLM (Exact):\n")
            f.write("-" * 40 + "\n")
            f.write(prompt)
        return path

    def _write_response_file(self, symbol: str, response: str, debug_note: Optional[str] = None) -> str:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.results_dir, f"sector_response_{symbol}_{ts}.txt")
        with open(path, 'w') as f:
            f.write("SECTOR SYNTHESIS RESPONSE\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Symbol: {symbol}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Response Length: {len(response) if response else 0} characters\n\n")
            if debug_note:
                f.write("DEBUG NOTE:\n")
                f.write("-" * 40 + "\n")
                f.write(debug_note.strip() + "\n\n")
            f.write("COMPLETE LLM RESPONSE:\n")
            f.write("-" * 40 + "\n")
            f.write(response or "<no response>")
        return path


    def _pct(self, value: Optional[float]) -> Optional[float]:
        try:
            if value is None:
                return None
            return round(float(value) * 100.0, 2)
        except Exception:
            return None

    def _safe_float(self, value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    def _infer_rotation_stage(self, relative_strength: Optional[float], momentum: Optional[float]) -> Optional[str]:
        try:
            if relative_strength is None or momentum is None:
                return None
            rs = float(relative_strength)
            mom = float(momentum)
            if rs >= 0 and mom >= 0:
                return "Leading"
            if rs < 0 and mom >= 0:
                return "Improving"
            if rs >= 0 and mom < 0:
                return "Weakening"
            return "Lagging"
        except Exception:
            return None

    def _build_sector_context_exact(self, symbol: str, sector_data: Optional[Dict[str, Any]], base_context: str) -> str:
        """Exact reproduction of SectorSynthesisProcessor._build_sector_context logic."""
        lines = []
        extras = []
        if sector_data:
            so = sector_data.get("sector_outperformance_pct")
            mo = sector_data.get("market_outperformance_pct")
            sb = sector_data.get("sector_beta")
            mb = sector_data.get("market_beta")
            rot = sector_data.get("rotation_stage")
            rot_mom = sector_data.get("rotation_momentum")
            sector_name = sector_data.get("sector_name")

            if so is not None:
                lines.append(f"- Sector Outperformance: {so}")
            if mo is not None:
                lines.append(f"- Market Outperformance: {mo}")
            if sb is not None:
                lines.append(f"- Sector Beta: {sb}")
            if mb is not None:
                extras.append(f"- Market Beta: {mb}")
            if sector_name:
                extras.append(f"- Sector: {sector_name}")
            if rot:
                extras.append(f"- Rotation Stage: {rot}")
            if rot_mom is not None:
                extras.append(f"- Rotation Momentum: {rot_mom}")

        header = [f"SECTOR CONTEXT for {symbol}".strip()]
        body = "\n".join(lines + extras)
        kc = "\n".join(["\n".join(header), body]).strip()
        if base_context:
            return f"{kc}\n\n{base_context}" if kc else base_context
        return kc

    async def run(self):
        print("🚀 Sector Synthesis Prompt/Response Test")
        print(f"Results directory: {self.results_dir}")
        print("==" * 40)

        api_key_present = bool(self._api_key)
        if api_key_present:
            print("✅ GEMINI_API_KEY detected — will call LLM and save responses")
        else:
            print("⚠️  No GEMINI_API_KEY detected — will only save prompts (no API calls)")

        # Authenticate Zerodha (best-effort)
        try:
            print("🔗 Authenticating with Zerodha...")
            if not self.zerodha_client.authenticate():
                print("⚠️  Zerodha authentication failed — attempting unauthenticated data access (if supported)")
        except Exception as e:
            print(f"⚠️  Zerodha auth error: {e}")

        # Pre-compute sector rotation analysis once (3M window)
        try:
            rotation = await self.benchmark_provider.analyze_sector_rotation_async(timeframe="3M")
            perf = (rotation or {}).get('sector_performance') or {}
            # Build rotation_map: sector_code -> {stage, momentum}
            for sec_code, pdata in perf.items():
                rel = pdata.get('relative_strength')
                mom = pdata.get('momentum')
                stage = self._infer_rotation_stage(rel, mom)
                self.rotation_map[sec_code] = {"stage": stage, "momentum": mom}
            print(f"📊 Loaded rotation context for {len(self.rotation_map)} sectors")
        except Exception as e:
            print(f"⚠️  Rotation analysis unavailable: {e}")
            self.rotation_map = {}

        # Build tasks for all symbols
        semaphore = asyncio.Semaphore(self.concurrency)
        results_summary: List[Tuple[str, str, Optional[str], Optional[str]]] = []  # (symbol, sector, prompt_path, response_path)

        async def process_symbol(symbol: str, index: int):
            nonlocal results_summary
            print(f"\n[{index}/{len(self.symbols)}] Processing {symbol}...")
            try:
                # Fetch stock data (period days, daily interval)
                stock_data = None
                if hasattr(self.zerodha_client, 'get_historical_data_async'):
                    stock_data = await self.zerodha_client.get_historical_data_async(
                        symbol=symbol, exchange=self.exchange, interval="day", period=self.period
                    )
                else:
                    loop = asyncio.get_event_loop()
                    stock_data = await loop.run_in_executor(
                        None,
                        self.zerodha_client.get_historical_data,
                        symbol, self.exchange, "day", None, None, self.period
                    )
                if stock_data is None or getattr(stock_data, 'empty', False):
                    print(f"❌ No data for {symbol}; skipping")
                    results_summary.append((symbol, "", None, None))
                    return
                if 'date' in stock_data.columns:
                    stock_data = stock_data.set_index('date')

                # Compute sector benchmarking (async)
                bench = await self.benchmark_provider.get_comprehensive_benchmarking_async(symbol, stock_data)
                sector_info = (bench or {}).get('sector_info') or {}
                sector_name = sector_info.get('sector_name') if sector_info else None
                sector_code = sector_info.get('sector') if sector_info else None
                sector_metrics = (bench or {}).get('sector_benchmarking') or {}
                market_metrics = (bench or {}).get('market_benchmarking') or {}

                # Map to sector synthesis metrics
                metrics = {
                    "sector_name": sector_name,
                    "sector_outperformance_pct": self._pct(sector_metrics.get('sector_outperformance') or sector_metrics.get('sector_excess_return')),
                    "market_outperformance_pct": self._pct(market_metrics.get('excess_return')),
                    "sector_beta": self._safe_float(sector_metrics.get('sector_beta')),
                    "market_beta": self._safe_float(market_metrics.get('beta')),
                    # Rotation extras inferred from rotation map (3M)
                    "rotation_stage": (self.rotation_map.get(sector_code, {}) or {}).get("stage"),
                    "rotation_momentum": (self.rotation_map.get(sector_code, {}) or {}).get("momentum"),
                }

                # Apply metrics overrides if provided
                if symbol in self.metrics_override:
                    metrics.update(self.metrics_override[symbol])

                cfg = SectorTestConfig(
                    symbol=symbol, name=symbol, sector_name=metrics.get('sector_name'),
                    sector_outperformance_pct=metrics.get('sector_outperformance_pct'),
                    market_outperformance_pct=metrics.get('market_outperformance_pct'),
                    sector_beta=metrics.get('sector_beta'), market_beta=metrics.get('market_beta'),
                    rotation_stage=metrics.get('rotation_stage'), rotation_momentum=metrics.get('rotation_momentum')
                )

                # Build sector knowledge context using exact logic
                sector_kc = self._build_sector_context_exact(symbol, cfg.to_sector_data(), base_context="")
                exact_prompt = self._build_exact_sector_prompt(self.prompt_manager, sector_kc)

                # Write files
                prompt_path = self._write_prompt_file(symbol, exact_prompt, sector_kc)
                response_path = None

                if api_key_present and self.client is not None:
                    print(f"🚀 Sending prompt to LLM for {symbol}...")
                    # Retry up to 3 times if empty response
                    attempts = 3
                    response_text = ""
                    debug_note = None
                    for attempt in range(1, attempts + 1):
                        try:
                            response_text = await self.client.synthesize_sector_summary(sector_kc)
                            print(response_text)
                        except Exception as llm_ex:  # Defensive: capture unexpected errors
                            debug_note = f"LLM call raised exception on attempt {attempt}: {llm_ex}"
                            response_text = ""
                        if response_text and response_text.strip():
                            break
                        if attempt < attempts:
                            wait = 2 ** attempt
                            print(f"⚠️  Empty response (attempt {attempt}/{attempts}). Retrying in {wait}s...")
                            await asyncio.sleep(wait)
                    if not response_text or not response_text.strip():
                        debug_extra = "; ".join(filter(None, [debug_note, "No text returned from LLM"]))
                        debug_note = f"LLM returned empty response after {attempts} attempt(s). {debug_extra}. Possible causes: missing/invalid GEMINI_API_KEY, rate limits (429), or network/auth errors."
                    response_path = self._write_response_file(symbol, response_text, debug_note=debug_note)
                    print(f"✅ Response written: {response_path}")
                else:
                    print("⏩ Skipping LLM call (prompt-only mode)")

                results_summary.append((symbol, sector_name or "", prompt_path, response_path))

            except Exception as e:
                print(f"❌ Error processing {symbol}: {e}")
                results_summary.append((symbol, "", None, None))

        async def worker(symbol: str, index: int):
            async with semaphore:
                await process_symbol(symbol, index)

        tasks = [worker(sym, i + 1) for i, sym in enumerate(self.symbols)]
        await asyncio.gather(*tasks)

        # Completed without summary file per user preference
        print("\n✅ Completed. Prompts saved to:", self.results_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Sector Synthesis Prompt/Response Tester")
    parser.add_argument("--symbols", type=str, default="RELIANCE,INFY,HDFCBANK", help="Comma-separated stock symbols")
    parser.add_argument("--sectors", type=str, default="", help="Comma-separated sector codes to include (will pick first stock per sector)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--no-llm", action="store_true", help="Do not call LLM; save prompts only")
    parser.add_argument("--period", type=int, default=365, help="Historical period in days for stock data")
    parser.add_argument("--exchange", type=str, default="NSE", help="Exchange (default: NSE)")
    parser.add_argument("--metrics-json", type=str, default=None, help="Path to JSON to override/fill metrics per symbol")
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent items to process")
    return parser.parse_args()


def main():
    args = parse_args()
    symbols = [s.strip() for s in (args.symbols or "").split(",") if s.strip()]
    sectors = [s.strip() for s in (args.sectors or "").split(",") if s.strip()]
    tester = SectorPromptResponseTester(
        symbols=symbols,
        sectors=sectors,
        output_dir=args.output_dir,
        no_llm=args.no_llm,
        period=args.period,
        exchange=args.exchange,
        metrics_json=args.metrics_json,
        concurrency=args.concurrency
    )
    asyncio.run(tester.run())


if __name__ == "__main__":
    main()
